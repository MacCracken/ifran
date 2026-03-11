//! REST handlers for the model marketplace.

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use synapse_types::marketplace::{MarketplaceEntry, MarketplaceQuery};
use synapse_types::model::ModelFormat;

use crate::state::AppState;

/// Response for a marketplace entry.
#[derive(Serialize)]
pub struct MarketplaceEntryResponse {
    pub model_name: String,
    pub description: Option<String>,
    pub format: ModelFormat,
    pub size_bytes: u64,
    pub publisher_instance: String,
    pub download_url: String,
    pub sha256: Option<String>,
    pub tags: Vec<String>,
    pub published_at: String,
}

/// Query params for marketplace search.
#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: Option<String>,
    pub format: Option<ModelFormat>,
    pub max_size: Option<u64>,
}

/// Request to publish a model.
#[derive(Deserialize)]
pub struct PublishRequest {
    pub model_name: String,
}

/// Request to pull a model from a remote marketplace entry.
#[derive(Deserialize)]
pub struct PullRequest {
    pub model_name: String,
    pub source_url: String,
    pub expected_sha256: Option<String>,
}

/// GET /marketplace/search — search marketplace entries.
pub async fn search(
    State(state): State<AppState>,
    Query(params): Query<SearchQuery>,
) -> Result<Json<Vec<MarketplaceEntryResponse>>, (StatusCode, String)> {
    let query = MarketplaceQuery {
        search: params.q,
        format: params.format,
        tags: None,
        max_size_bytes: params.max_size,
    };

    let catalog = state.marketplace_catalog.lock().await;
    let entries = catalog
        .search(&query)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(entries.iter().map(entry_to_response).collect()))
}

/// GET /marketplace/entries — list all marketplace entries.
pub async fn list_entries(
    State(state): State<AppState>,
) -> Result<Json<Vec<MarketplaceEntryResponse>>, (StatusCode, String)> {
    let catalog = state.marketplace_catalog.lock().await;
    let entries = catalog
        .list()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(entries.iter().map(entry_to_response).collect()))
}

/// POST /marketplace/publish — publish a local model to the marketplace.
pub async fn publish(
    State(state): State<AppState>,
    Json(req): Json<PublishRequest>,
) -> Result<(StatusCode, Json<MarketplaceEntryResponse>), (StatusCode, String)> {
    // Look up the model in the local DB
    let db = state.db.lock().await;
    let model = db
        .get_by_name(&req.model_name)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let instance_id =
        std::env::var("SYNAPSE_INSTANCE_ID").unwrap_or_else(|_| state.config.server.bind.clone());
    let base_url = format!("http://{}", state.config.server.bind);

    let entry = synapse_core::marketplace::publisher::create_entry(&model, &instance_id, &base_url)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let catalog = state.marketplace_catalog.lock().await;
    catalog
        .publish(&entry)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(entry_to_response(&entry))))
}

/// DELETE /marketplace/entries/:name — unpublish a model.
pub async fn unpublish(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let catalog = state.marketplace_catalog.lock().await;
    catalog
        .unpublish(&name)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

/// GET /marketplace/download/:name — stream a published model file.
pub async fn download(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<axum::response::Response<Body>, (StatusCode, String)> {
    // Decode URL-encoded name (__ back to /)
    let model_name = name.replace("__", "/");

    // Look up the model in the local DB to find the file path
    let db = state.db.lock().await;
    let model = db
        .get_by_name(&model_name)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let path = std::path::Path::new(&model.local_path);
    if !path.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Model file not found on disk: {}", model.local_path),
        ));
    }

    let file_bytes = tokio::fs::read(path)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| model_name.clone());

    let response = axum::response::Response::builder()
        .header("content-type", "application/octet-stream")
        .header(
            "content-disposition",
            format!("attachment; filename=\"{filename}\""),
        )
        .header("content-length", file_bytes.len().to_string())
        .body(Body::from(file_bytes))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(response)
}

/// POST /marketplace/pull — download a model from a remote marketplace entry.
pub async fn pull(
    State(state): State<AppState>,
    Json(req): Json<PullRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let client = synapse_core::pull::downloader::build_client()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Determine destination path
    let dest_dir = state.config.storage.models_dir.clone();
    let safe_name = req.model_name.replace('/', "__");
    let dest = dest_dir.join(&safe_name);

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    // Download the file
    let progress = synapse_core::pull::progress::ProgressTracker::new(16);
    let download_req = synapse_core::pull::downloader::DownloadRequest {
        url: req.source_url.clone(),
        dest: dest.clone(),
        model_name: req.model_name.clone(),
        expected_sha256: req.expected_sha256.clone(),
    };

    synapse_core::pull::downloader::download(&client, &download_req, &progress)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "model_name": req.model_name,
            "path": dest.to_string_lossy(),
            "verified": req.expected_sha256.is_some(),
        })),
    ))
}

fn entry_to_response(entry: &MarketplaceEntry) -> MarketplaceEntryResponse {
    MarketplaceEntryResponse {
        model_name: entry.model_name.clone(),
        description: entry.description.clone(),
        format: entry.format,
        size_bytes: entry.size_bytes,
        publisher_instance: entry.publisher_instance.clone(),
        download_url: entry.download_url.clone(),
        sha256: entry.sha256.clone(),
        tags: entry.tags.clone(),
        published_at: entry.published_at.to_rfc3339(),
    }
}
