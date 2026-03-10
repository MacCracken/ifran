//! REST handlers for the model marketplace.

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

fn entry_to_response(entry: &MarketplaceEntry) -> MarketplaceEntryResponse {
    MarketplaceEntryResponse {
        model_name: entry.model_name.clone(),
        description: entry.description.clone(),
        format: entry.format,
        size_bytes: entry.size_bytes,
        publisher_instance: entry.publisher_instance.clone(),
        download_url: entry.download_url.clone(),
        tags: entry.tags.clone(),
        published_at: entry.published_at.to_rfc3339(),
    }
}
