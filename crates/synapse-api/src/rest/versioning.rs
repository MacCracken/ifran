//! REST handlers for model versioning.

use crate::state::AppState;
use axum::Json;
use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use serde::Deserialize;
use synapse_types::TenantId;
use synapse_types::versioning::ModelVersion;

use super::pagination::{PaginatedResponse, PaginationQuery};

#[derive(Deserialize)]
pub struct CreateVersionRequest {
    pub model_family: String,
    pub version_tag: String,
    pub model_id: Option<uuid::Uuid>,
    pub training_job_id: Option<uuid::Uuid>,
    pub parent_version_id: Option<uuid::Uuid>,
    pub consumer: Option<String>,
    pub notes: Option<String>,
}

#[derive(Deserialize)]
pub struct FamilyQuery {
    pub family: Option<String>,
    #[serde(flatten)]
    pub page: PaginationQuery,
}

/// POST /versions -- create a new model version.
pub async fn create_version(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Json(req): Json<CreateVersionRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let store = state.version_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Version store not initialized".into(),
    ))?;

    let version = ModelVersion {
        id: uuid::Uuid::new_v4(),
        model_family: req.model_family,
        version_tag: req.version_tag,
        model_id: req.model_id,
        training_job_id: req.training_job_id,
        parent_version_id: req.parent_version_id,
        consumer: req.consumer,
        notes: req.notes,
        created_at: chrono::Utc::now(),
    };

    let store = store.lock().await;
    store.create(&version, &tenant_id).map_err(|e| {
        tracing::error!(error = %e, "internal error");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error".into(),
        )
    })?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "id": version.id.to_string(),
            "model_family": version.model_family,
            "version_tag": version.version_tag,
        })),
    ))
}

/// GET /versions -- list versions, optionally by family.
pub async fn list_versions(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Query(query): Query<FamilyQuery>,
) -> Result<Json<PaginatedResponse<serde_json::Value>>, (StatusCode, String)> {
    let store = state.version_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Version store not initialized".into(),
    ))?;

    let page = &query.page;
    let safe_limit = page.safe_limit();
    let store = store.lock().await;
    let versions = match query.family {
        Some(ref f) => store.list_by_family(f, &tenant_id, safe_limit, page.offset),
        None => store.list(&tenant_id, safe_limit, page.offset),
    }
    .map_err(|e| {
        tracing::error!(error = %e, "internal error");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error".into(),
        )
    })?;

    let data: Vec<serde_json::Value> = versions.iter().map(version_to_json).collect();
    let total = data.len();
    Ok(Json(PaginatedResponse::pre_sliced(
        data,
        total,
        safe_limit,
        page.offset,
    )))
}

/// GET /versions/:id -- get a specific version.
pub async fn get_version(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<uuid::Uuid>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.version_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Version store not initialized".into(),
    ))?;

    let store = store.lock().await;
    let version = store
        .get(id, &tenant_id)
        .map_err(|_| (StatusCode::NOT_FOUND, "Not found".into()))?;

    Ok(Json(version_to_json(&version)))
}

/// GET /versions/:id/lineage -- get version lineage chain.
pub async fn get_lineage(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<uuid::Uuid>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let store = state.version_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Version store not initialized".into(),
    ))?;

    let store = store.lock().await;
    let chain = store.get_lineage(id, &tenant_id).map_err(|e| {
        tracing::error!(error = %e, "internal error");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error".into(),
        )
    })?;

    Ok(Json(chain.iter().map(version_to_json).collect()))
}

fn version_to_json(v: &ModelVersion) -> serde_json::Value {
    serde_json::json!({
        "id": v.id.to_string(),
        "model_family": v.model_family,
        "version_tag": v.version_tag,
        "model_id": v.model_id.map(|id| id.to_string()),
        "training_job_id": v.training_job_id.map(|id| id.to_string()),
        "parent_version_id": v.parent_version_id.map(|id| id.to_string()),
        "consumer": v.consumer,
        "notes": v.notes,
        "created_at": v.created_at.to_rfc3339(),
    })
}
