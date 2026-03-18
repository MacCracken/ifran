//! REST handlers for fleet node management.

use crate::state::AppState;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::Deserialize;
use synapse_core::fleet::manager::RegisterNodeRequest;

/// POST /fleet/nodes — register a new node.
pub async fn register_node(
    State(state): State<AppState>,
    Json(body): Json<RegisterNodeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let node = state.fleet_manager.register(body).await;
    Ok(Json(serde_json::to_value(&node).unwrap()))
}

/// Heartbeat request body.
#[derive(Debug, Deserialize)]
pub struct HeartbeatBody {
    pub gpu_utilization_pct: Option<f32>,
    pub gpu_memory_used_mb: Option<u64>,
    pub gpu_temperature_c: Option<f32>,
}

/// POST /fleet/nodes/{id}/heartbeat — process a heartbeat.
pub async fn heartbeat(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<HeartbeatBody>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .fleet_manager
        .heartbeat(
            &id,
            body.gpu_utilization_pct,
            body.gpu_memory_used_mb,
            body.gpu_temperature_c,
        )
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e))?;
    Ok(StatusCode::NO_CONTENT)
}

/// GET /fleet/nodes — list all nodes.
pub async fn list_nodes(State(state): State<AppState>) -> Json<serde_json::Value> {
    let nodes = state.fleet_manager.list_nodes().await;
    Json(serde_json::json!({ "nodes": nodes }))
}

/// GET /fleet/stats — fleet statistics.
pub async fn fleet_stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    let stats = state.fleet_manager.stats().await;
    Json(serde_json::to_value(&stats).unwrap())
}

/// DELETE /fleet/nodes/{id} — remove a node.
pub async fn remove_node(State(state): State<AppState>, Path(id): Path<String>) -> StatusCode {
    if state.fleet_manager.remove(&id).await {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heartbeat_body_deserialize() {
        let json = r#"{"gpu_utilization_pct": 85.5, "gpu_memory_used_mb": 20000}"#;
        let body: HeartbeatBody = serde_json::from_str(json).unwrap();
        assert_eq!(body.gpu_utilization_pct, Some(85.5));
        assert_eq!(body.gpu_memory_used_mb, Some(20000));
        assert!(body.gpu_temperature_c.is_none());
    }

    #[test]
    fn heartbeat_body_empty() {
        let json = r#"{}"#;
        let body: HeartbeatBody = serde_json::from_str(json).unwrap();
        assert!(body.gpu_utilization_pct.is_none());
    }
}
