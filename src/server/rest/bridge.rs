//! REST handlers for SY bridge status and management.

use crate::types::TenantId;
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::Serialize;

use crate::server::state::AppState;

/// Bridge status response.
#[derive(Serialize)]
pub struct BridgeStatusResponse {
    pub enabled: bool,
    pub client_state: String,
    pub server_state: String,
    pub sy_endpoint: Option<String>,
    pub heartbeat_interval_secs: u64,
}

/// GET /bridge/status — get bridge connection status.
pub async fn status(
    State(state): State<AppState>,
    Extension(_tenant_id): Extension<TenantId>,
) -> Json<BridgeStatusResponse> {
    let client_state = match &state.bridge_client {
        Some(client) => format!("{:?}", client.connection_state().await),
        None => "disabled".into(),
    };

    let server_state = match &state.bridge_server {
        Some(server) => format!("{:?}", server.connection_state().await),
        None => "disabled".into(),
    };

    Json(BridgeStatusResponse {
        enabled: state.config.bridge.enabled,
        client_state,
        server_state,
        sy_endpoint: state.config.bridge.sy_endpoint.clone(),
        heartbeat_interval_secs: state.config.bridge.heartbeat_interval_secs,
    })
}

/// POST /bridge/connect — manually trigger bridge connection to SY.
pub async fn connect(
    State(state): State<AppState>,
    Extension(_tenant_id): Extension<TenantId>,
) -> Result<Json<BridgeStatusResponse>, (StatusCode, String)> {
    let client = state.bridge_client.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Bridge is not enabled. Set [bridge] enabled = true in ifran.toml".into(),
        )
    })?;

    client
        .connect()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;

    // Also start the server if present
    if let Some(server) = &state.bridge_server {
        let grpc_service = crate::bridge::server::IfranBridgeService::new(
            state.job_manager.clone(),
            state.backends.clone(),
            state.model_manager.clone(),
        );
        server
            .start(&state.config.server.grpc_bind, grpc_service)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    // Announce capabilities
    let hardware = crate::hardware::detect::detect().ok();
    let (gpu_count, total_gpu_mem) = hardware
        .as_ref()
        .map(|hw| {
            (
                hw.gpus.len() as u32,
                hw.gpus.iter().map(|g| g.memory_total_mb).sum::<u64>(),
            )
        })
        .unwrap_or((0, 0));

    let instance_id =
        std::env::var("IFRAN_INSTANCE_ID").unwrap_or_else(|_| state.config.server.bind.clone());

    // Collect dynamic capability information from backends and model manager
    let backend_ids = state.backends.list_backends();
    let backends: Vec<String> = backend_ids.iter().map(|id| id.0.clone()).collect();

    let mut all_formats = std::collections::BTreeSet::new();
    for id in &backend_ids {
        if let Some(b) = state.backends.get(id) {
            for fmt in b.supported_formats() {
                all_formats.insert(format!("{fmt:?}").to_lowercase());
            }
        }
    }
    let supported_formats: Vec<String> = all_formats.into_iter().collect();

    let loaded = state.model_manager.list_loaded(None).await;
    let loaded_models: Vec<String> = loaded.iter().map(|m| m.model_name.clone()).collect();

    let supported_quants: Vec<String> = vec![
        "f32", "f16", "bf16", "q8_0", "q6k", "q5km", "q5ks", "q4km", "q4ks", "q4_0", "q3km",
        "q3ks", "q2k", "iq4xs", "iq3xxs",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let capabilities = crate::bridge::protocol::Capabilities {
        instance_id,
        version: env!("CARGO_PKG_VERSION").into(),
        gpu_count,
        total_gpu_memory_mb: total_gpu_mem,
        supported_methods: vec![
            "lora".into(),
            "qlora".into(),
            "full".into(),
            "dpo".into(),
            "rlhf".into(),
            "distillation".into(),
        ],
        backends,
        loaded_models,
        supported_formats,
        supported_quants,
    };

    client
        .announce(capabilities)
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;

    Ok(Json(BridgeStatusResponse {
        enabled: true,
        client_state: format!("{:?}", client.connection_state().await),
        server_state: match &state.bridge_server {
            Some(s) => format!("{:?}", s.connection_state().await),
            None => "disabled".into(),
        },
        sy_endpoint: state.config.bridge.sy_endpoint.clone(),
        heartbeat_interval_secs: state.config.bridge.heartbeat_interval_secs,
    }))
}

/// POST /bridge/heartbeat — send a one-off heartbeat to SY (useful for debugging).
pub async fn heartbeat(
    State(state): State<AppState>,
    Extension(_tenant_id): Extension<TenantId>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let server = state.bridge_server.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Bridge is not enabled. Set [bridge] enabled = true in ifran.toml".into(),
        )
    })?;

    let loaded_models = state.model_manager.list_loaded(None).await.len() as u32;
    let active_jobs = state.job_manager.running_count().await as u32;

    let gpu_free = crate::hardware::detect::detect()
        .ok()
        .map(|hw| hw.gpus.iter().map(|g| g.memory_free_mb).sum::<u64>())
        .unwrap_or(0);

    let hb = server.build_heartbeat(loaded_models, gpu_free, active_jobs);

    // If client is connected, report progress for any running jobs
    if let Some(client) = &state.bridge_client {
        client
            .report_progress("heartbeat", "alive", 0, 0.0)
            .await
            .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;
    }

    Ok(Json(serde_json::json!({
        "instance_id": hb.instance_id,
        "timestamp": hb.timestamp,
        "loaded_models": hb.loaded_models,
        "gpu_memory_free_mb": hb.gpu_memory_free_mb,
        "active_training_jobs": hb.active_training_jobs,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_status_response_serializes() {
        let resp = BridgeStatusResponse {
            enabled: true,
            client_state: "Connected".into(),
            server_state: "Listening".into(),
            sy_endpoint: Some("https://sy.example.com".into()),
            heartbeat_interval_secs: 30,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["enabled"], true);
        assert_eq!(json["client_state"], "Connected");
        assert_eq!(json["server_state"], "Listening");
        assert_eq!(json["sy_endpoint"], "https://sy.example.com");
        assert_eq!(json["heartbeat_interval_secs"], 30);
    }

    #[test]
    fn bridge_status_response_disabled() {
        let resp = BridgeStatusResponse {
            enabled: false,
            client_state: "disabled".into(),
            server_state: "disabled".into(),
            sy_endpoint: None,
            heartbeat_interval_secs: 10,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["enabled"], false);
        assert!(json["sy_endpoint"].is_null());
        assert_eq!(json["heartbeat_interval_secs"], 10);
    }

    #[test]
    fn bridge_status_response_roundtrip() {
        let resp = BridgeStatusResponse {
            enabled: true,
            client_state: "Reconnecting".into(),
            server_state: "disabled".into(),
            sy_endpoint: Some("ws://10.0.0.1:9090".into()),
            heartbeat_interval_secs: 60,
        };
        let serialized = serde_json::to_string(&resp).unwrap();
        assert!(serialized.contains("Reconnecting"));
        assert!(serialized.contains("ws://10.0.0.1:9090"));
    }
}
