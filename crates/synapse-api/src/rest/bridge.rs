//! REST handlers for SY bridge status and management.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use serde::Serialize;

use crate::state::AppState;

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
pub async fn status(State(state): State<AppState>) -> Json<BridgeStatusResponse> {
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
) -> Result<Json<BridgeStatusResponse>, (StatusCode, String)> {
    let client = state
        .bridge_client
        .as_ref()
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Bridge is not enabled".into()))?;

    client
        .connect()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;

    // Also start the server if present
    if let Some(server) = &state.bridge_server {
        server
            .start(&state.config.server.grpc_bind)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    // Announce capabilities
    let hardware = synapse_core::hardware::detect::detect().ok();
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
        std::env::var("SYNAPSE_INSTANCE_ID").unwrap_or_else(|_| state.config.server.bind.clone());

    let capabilities = synapse_bridge::protocol::Capabilities {
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
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let server = state
        .bridge_server
        .as_ref()
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Bridge is not enabled".into()))?;

    let loaded_models = state.model_manager.list_loaded().await.len() as u32;
    let active_jobs = state.job_manager.running_count().await as u32;

    let gpu_free = synapse_core::hardware::detect::detect()
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
