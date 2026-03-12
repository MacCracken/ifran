//! REST handlers for system endpoints (health, version, GPU status).

use crate::state::AppState;
use axum::Json;
use axum::extract::State;

/// GET /health — simple liveness probe.
pub async fn health() -> &'static str {
    "ok"
}

/// GET /system/status — detailed system information.
pub async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let loaded_models = state.model_manager.list_loaded().await;
    let backends = state.backends.list_backends();

    let hardware = synapse_core::hardware::detect::detect()
        .map(|hw| {
            serde_json::json!({
                "cpu": {
                    "model": hw.cpu.model_name,
                    "cores": hw.cpu.physical_cores,
                    "threads": hw.cpu.logical_cores,
                    "memory_total_mb": hw.cpu.total_memory_mb,
                    "memory_available_mb": hw.cpu.available_memory_mb,
                },
                "gpus": hw.gpus.iter().map(|g| serde_json::json!({
                    "index": g.index,
                    "name": g.name,
                    "memory_total_mb": g.memory_total_mb,
                    "memory_free_mb": g.memory_free_mb,
                })).collect::<Vec<_>>(),
            })
        })
        .unwrap_or(serde_json::json!({"error": "detection failed"}));

    let bridge = serde_json::json!({
        "enabled": state.config.bridge.enabled,
        "client_state": match &state.bridge_client {
            Some(c) => format!("{:?}", c.connection_state().await),
            None => "disabled".into(),
        },
        "server_state": match &state.bridge_server {
            Some(s) => format!("{:?}", s.connection_state().await),
            None => "disabled".into(),
        },
    });

    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "loaded_models": loaded_models.len(),
        "registered_backends": backends.iter().map(|b| &b.0).collect::<Vec<_>>(),
        "hardware": hardware,
        "bridge": bridge,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn health_returns_ok() {
        let result = health().await;
        assert_eq!(result, "ok");
    }

    #[tokio::test]
    async fn status_returns_expected_fields() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = synapse_core::config::SynapseConfig {
            server: synapse_core::config::ServerConfig {
                bind: "127.0.0.1:0".into(),
                grpc_bind: "127.0.0.1:0".into(),
            },
            storage: synapse_core::config::StorageConfig {
                models_dir: tmp.path().join("models"),
                database: tmp.path().join("test.db"),
                cache_dir: tmp.path().join("cache"),
            },
            backends: synapse_core::config::BackendsConfig {
                default: "llamacpp".into(),
                enabled: vec!["llamacpp".into()],
            },
            training: synapse_core::config::TrainingConfig {
                executor: "subprocess".into(),
                trainer_image: None,
                max_concurrent_jobs: 2,
                checkpoints_dir: tmp.path().join("checkpoints"),
            },
            bridge: synapse_core::config::BridgeConfig {
                sy_endpoint: None,
                enabled: false,
                heartbeat_interval_secs: 10,
            },
            hardware: synapse_core::config::HardwareConfig {
                gpu_memory_reserve_mb: 512,
            },
        };

        let state = AppState::new(config).unwrap();
        let Json(json) = status(State(state)).await;

        assert!(json["version"].is_string());
        assert_eq!(json["loaded_models"], 0);
        assert!(json["registered_backends"].is_array());
        assert!(json["hardware"].is_object());
        assert!(json["bridge"].is_object());
        assert_eq!(json["bridge"]["enabled"], false);
        assert_eq!(json["bridge"]["client_state"], "disabled");
        assert_eq!(json["bridge"]["server_state"], "disabled");
    }
}
