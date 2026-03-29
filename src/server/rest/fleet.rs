//! REST handlers for fleet node management.

use crate::fleet::manager::{FleetNode, NodeHealth, RegisterNodeRequest};
use crate::server::state::AppState;
use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use serde::Deserialize;

use super::pagination::{PaginatedResponse, PaginationQuery};

/// POST /fleet/nodes — register a new node.
pub async fn register_node(
    State(state): State<AppState>,
    Json(body): Json<RegisterNodeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let node = state
        .fleet_manager
        .register(body)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let value = serde_json::to_value(&node)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(value))
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
        .map_err(|e| {
            let status = match &e {
                crate::types::IfranError::ValidationError(_) => StatusCode::BAD_REQUEST,
                _ => StatusCode::NOT_FOUND,
            };
            (status, e.to_string())
        })?;
    Ok(StatusCode::NO_CONTENT)
}

/// Query parameters for listing fleet nodes.
#[derive(Debug, Deserialize)]
pub struct ListNodesQuery {
    #[serde(flatten)]
    pub page: PaginationQuery,
    /// Optional health filter, e.g. `?health=online`.
    pub health: Option<NodeHealth>,
}

/// GET /fleet/nodes — list all nodes.
pub async fn list_nodes(
    State(state): State<AppState>,
    Query(query): Query<ListNodesQuery>,
) -> Json<PaginatedResponse<FleetNode>> {
    let nodes = state.fleet_manager.list_nodes().await;
    let filtered: Vec<FleetNode> = nodes
        .into_iter()
        .filter(|n| query.health.is_none() || Some(n.health) == query.health)
        .collect();
    Json(PaginatedResponse::from_slice(
        &filtered,
        &query.page,
        |node| node.clone(),
    ))
}

/// GET /fleet/stats — fleet statistics.
pub async fn fleet_stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    let stats = state.fleet_manager.stats().await;
    Json(serde_json::json!({
        "total_nodes": stats.total_nodes,
        "online": stats.online,
        "suspect": stats.suspect,
        "offline": stats.offline,
        "total_gpus": stats.total_gpus,
        "total_gpu_memory_mb": stats.total_gpu_memory_mb,
    }))
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

    #[test]
    fn heartbeat_body_all_fields() {
        let json = r#"{"gpu_utilization_pct": 50.0, "gpu_memory_used_mb": 8000, "gpu_temperature_c": 72.5}"#;
        let body: HeartbeatBody = serde_json::from_str(json).unwrap();
        assert_eq!(body.gpu_utilization_pct, Some(50.0));
        assert_eq!(body.gpu_memory_used_mb, Some(8000));
        assert_eq!(body.gpu_temperature_c, Some(72.5));
    }

    #[test]
    fn register_node_request_deserialize() {
        let json = r#"{
            "id": "node-42",
            "endpoint": "http://10.0.0.42:8420",
            "gpu_count": 4,
            "total_gpu_memory_mb": 98304
        }"#;
        let req: RegisterNodeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.id, "node-42");
        assert_eq!(req.endpoint, "http://10.0.0.42:8420");
        assert_eq!(req.gpu_count, 4);
        assert_eq!(req.total_gpu_memory_mb, 98304);
    }

    #[test]
    fn register_node_request_missing_fields() {
        let json = r#"{"id": "node-1"}"#;
        let result = serde_json::from_str::<RegisterNodeRequest>(json);
        assert!(result.is_err());
    }

    #[test]
    fn list_nodes_query_deserialize() {
        let json = r#"{"health": "online"}"#;
        let q: ListNodesQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.health, Some(NodeHealth::Online));
    }

    #[test]
    fn list_nodes_query_no_filter() {
        let json = r#"{}"#;
        let q: ListNodesQuery = serde_json::from_str(json).unwrap();
        assert!(q.health.is_none());
    }

    use crate::server::test_helpers::helpers::test_state;

    #[tokio::test]
    async fn list_nodes_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let query = ListNodesQuery {
            page: PaginationQuery {
                limit: super::super::pagination::DEFAULT_LIMIT,
                offset: 0,
            },
            health: None,
        };
        let result = list_nodes(State(state), Query(query)).await;
        assert!(result.0.data.is_empty());
    }

    #[tokio::test]
    async fn fleet_stats_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let result = fleet_stats(State(state)).await;
        let json = result.0;
        assert_eq!(json["total_nodes"], 0);
        assert_eq!(json["online"], 0);
        assert_eq!(json["suspect"], 0);
        assert_eq!(json["offline"], 0);
    }

    #[tokio::test]
    async fn remove_node_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let result = remove_node(State(state), Path("nonexistent".into())).await;
        assert_eq!(result, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn heartbeat_node_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let body = HeartbeatBody {
            gpu_utilization_pct: Some(50.0),
            gpu_memory_used_mb: Some(4000),
            gpu_temperature_c: None,
        };
        let result = heartbeat(State(state), Path("nonexistent".into()), Json(body)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn register_and_list_node() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = RegisterNodeRequest {
            id: "test-node-1".into(),
            endpoint: "http://127.0.0.1:9000".into(),
            gpu_count: 2,
            total_gpu_memory_mb: 24576,
        };
        let reg_result = register_node(State(state.clone()), Json(req)).await;
        assert!(reg_result.is_ok());

        let query = ListNodesQuery {
            page: PaginationQuery {
                limit: super::super::pagination::DEFAULT_LIMIT,
                offset: 0,
            },
            health: None,
        };
        let list = list_nodes(State(state.clone()), Query(query)).await;
        assert_eq!(list.0.data.len(), 1);
        assert_eq!(list.0.data[0].id, "test-node-1");

        // Stats should show 1 node online
        let stats = fleet_stats(State(state)).await;
        assert_eq!(stats.0["total_nodes"], 1);
        assert_eq!(stats.0["online"], 1);
    }

    #[tokio::test]
    async fn list_nodes_with_health_filter() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = RegisterNodeRequest {
            id: "healthy-node".into(),
            endpoint: "http://127.0.0.1:9001".into(),
            gpu_count: 1,
            total_gpu_memory_mb: 8192,
        };
        let _ = register_node(State(state.clone()), Json(req))
            .await
            .unwrap();

        // Filter for online — should find it
        let query_online = ListNodesQuery {
            page: PaginationQuery {
                limit: super::super::pagination::DEFAULT_LIMIT,
                offset: 0,
            },
            health: Some(NodeHealth::Online),
        };
        let online = list_nodes(State(state.clone()), Query(query_online)).await;
        assert_eq!(online.0.data.len(), 1);

        // Filter for offline — should be empty
        let query_offline = ListNodesQuery {
            page: PaginationQuery {
                limit: super::super::pagination::DEFAULT_LIMIT,
                offset: 0,
            },
            health: Some(NodeHealth::Offline),
        };
        let offline = list_nodes(State(state), Query(query_offline)).await;
        assert!(offline.0.data.is_empty());
    }
}
