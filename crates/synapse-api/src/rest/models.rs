//! REST handlers for model management (list, get, remove).

use crate::state::AppState;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;

/// GET /models — list all models in the catalog.
pub async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db = state.db.lock().await;
    let models = db.list().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id.to_string(),
                "name": m.name,
                "repo_id": m.repo_id,
                "format": format!("{:?}", m.format).to_lowercase(),
                "quant": format!("{:?}", m.quant),
                "size_bytes": m.size_bytes,
                "parameter_count": m.parameter_count,
                "architecture": m.architecture,
                "local_path": m.local_path,
                "pulled_at": m.pulled_at.to_rfc3339(),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({ "data": data })))
}

/// GET /models/:id — get a specific model by ID.
pub async fn get_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db = state.db.lock().await;
    let model = if let Ok(uuid) = uuid::Uuid::parse_str(&id) {
        db.get(uuid)
    } else {
        db.get_by_name(&id)
    };

    let model = model.map_err(|_| StatusCode::NOT_FOUND)?;

    Ok(Json(serde_json::json!({
        "id": model.id.to_string(),
        "name": model.name,
        "repo_id": model.repo_id,
        "format": format!("{:?}", model.format).to_lowercase(),
        "quant": format!("{:?}", model.quant),
        "size_bytes": model.size_bytes,
        "parameter_count": model.parameter_count,
        "architecture": model.architecture,
        "local_path": model.local_path,
        "sha256": model.sha256,
        "pulled_at": model.pulled_at.to_rfc3339(),
    })))
}

/// DELETE /models/:id — remove a model from the catalog and disk.
pub async fn delete_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    let db = state.db.lock().await;
    let model = if let Ok(uuid) = uuid::Uuid::parse_str(&id) {
        db.get(uuid)
    } else {
        db.get_by_name(&id)
    };

    let model = model.map_err(|_| StatusCode::NOT_FOUND)?;

    // Remove files
    let local_path = std::path::Path::new(&model.local_path);
    if let Some(model_dir) = local_path.parent() {
        if model_dir.exists() {
            let _ = std::fs::remove_dir_all(model_dir);
        }
    }

    db.delete(model.id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(StatusCode::NO_CONTENT)
}
