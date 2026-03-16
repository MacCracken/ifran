//! REST handlers for pipeline lineage tracking.

use crate::state::AppState;
use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::Deserialize;
use synapse_types::TenantId;
use synapse_types::lineage::{LineageNode, PipelineStage};

#[derive(Deserialize)]
pub struct RecordRequest {
    pub stage: PipelineStage,
    pub name: String,
    pub artifact_ref: String,
    #[serde(default)]
    pub parent_ids: Vec<uuid::Uuid>,
    #[serde(default = "default_metadata")]
    pub metadata: serde_json::Value,
}

fn default_metadata() -> serde_json::Value {
    serde_json::json!({})
}

#[derive(Deserialize)]
pub struct ListQuery {
    pub stage: Option<PipelineStage>,
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

fn default_limit() -> u32 {
    100
}

/// POST /lineage — record a new lineage node.
pub async fn record_node(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Json(req): Json<RecordRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let store = state.lineage_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Lineage store not initialized".into(),
    ))?;

    let node = LineageNode {
        id: uuid::Uuid::new_v4(),
        stage: req.stage,
        name: req.name,
        artifact_ref: req.artifact_ref,
        parent_ids: req.parent_ids,
        metadata: req.metadata,
        created_at: chrono::Utc::now(),
    };

    let store = store.lock().await;
    store.record(&node, &tenant_id).map_err(|e| {
        tracing::error!(error = %e, "internal error");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error".into(),
        )
    })?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "id": node.id.to_string(),
            "stage": node.stage,
            "name": node.name,
        })),
    ))
}

/// GET /lineage — list lineage nodes, optionally filtered by stage.
pub async fn list_nodes(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let store = state.lineage_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Lineage store not initialized".into(),
    ))?;

    let store = store.lock().await;
    let safe_limit = query.limit.min(1000);
    let nodes = store
        .list(&tenant_id, query.stage, safe_limit, query.offset)
        .map_err(|e| {
            tracing::error!(error = %e, "internal error");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".into(),
            )
        })?;

    Ok(Json(nodes.iter().map(node_to_json).collect()))
}

/// GET /lineage/:id — get a lineage node with its parents.
pub async fn get_node(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<uuid::Uuid>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.lineage_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Lineage store not initialized".into(),
    ))?;

    let store = store.lock().await;
    let node = store
        .get(id, &tenant_id)
        .map_err(|_| (StatusCode::NOT_FOUND, "Not found".into()))?;

    Ok(Json(node_to_json(&node)))
}

/// GET /lineage/:id/ancestry — get full ancestry chain.
pub async fn get_ancestry(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<uuid::Uuid>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let store = state.lineage_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Lineage store not initialized".into(),
    ))?;

    let store = store.lock().await;
    let nodes = store.get_ancestry(id, &tenant_id).map_err(|e| {
        tracing::error!(error = %e, "internal error");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error".into(),
        )
    })?;

    Ok(Json(nodes.iter().map(node_to_json).collect()))
}

fn node_to_json(node: &LineageNode) -> serde_json::Value {
    serde_json::json!({
        "id": node.id.to_string(),
        "stage": node.stage,
        "name": node.name,
        "artifact_ref": node.artifact_ref,
        "parent_ids": node.parent_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
        "metadata": node.metadata,
        "created_at": node.created_at.to_rfc3339(),
    })
}
