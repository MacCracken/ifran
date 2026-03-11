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
