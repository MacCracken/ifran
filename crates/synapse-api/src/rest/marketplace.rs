//! REST handlers for the model marketplace.

use axum::body::Body;
use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use synapse_types::TenantId;
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
    Extension(tenant_id): Extension<TenantId>,
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
        .search(&query, &tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(entries.iter().map(entry_to_response).collect()))
}

/// GET /marketplace/entries — list all marketplace entries.
pub async fn list_entries(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
) -> Result<Json<Vec<MarketplaceEntryResponse>>, (StatusCode, String)> {
    let catalog = state.marketplace_catalog.lock().await;
    let entries = catalog
        .list(&tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(entries.iter().map(entry_to_response).collect()))
}

/// POST /marketplace/publish — publish a local model to the marketplace.
pub async fn publish(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Json(req): Json<PublishRequest>,
) -> Result<(StatusCode, Json<MarketplaceEntryResponse>), (StatusCode, String)> {
    // Look up the model in the local DB
    let db = state.db.lock().await;
    let model = db
        .get_by_name(&req.model_name, &tenant_id)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let instance_id =
        std::env::var("SYNAPSE_INSTANCE_ID").unwrap_or_else(|_| state.config.server.bind.clone());
    let base_url = format!("http://{}", state.config.server.bind);

    let entry = synapse_core::marketplace::publisher::create_entry(&model, &instance_id, &base_url)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let catalog = state.marketplace_catalog.lock().await;
    catalog
        .publish(&entry, &tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(entry_to_response(&entry))))
}

/// DELETE /marketplace/entries/:name — unpublish a model.
pub async fn unpublish(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(name): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let catalog = state.marketplace_catalog.lock().await;
    catalog
        .unpublish(&name, &tenant_id)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

/// GET /marketplace/download/:name — stream a published model file.
pub async fn download(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(name): Path<String>,
) -> Result<axum::response::Response<Body>, (StatusCode, String)> {
    // Decode URL-encoded name (__ back to /)
    let model_name = name.replace("__", "/");

    // Look up the model in the local DB to find the file path
    let db = state.db.lock().await;
    let model = db
        .get_by_name(&model_name, &tenant_id)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    // Open file and get metadata atomically — avoids TOCTOU race where
    // file could be deleted between exists() check and open()
    let path = std::path::Path::new(&model.local_path);

    let file = tokio::fs::File::open(path).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            (
                StatusCode::NOT_FOUND,
                format!("Model file not found on disk: {}", model.local_path),
            )
        } else {
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        }
    })?;

    let metadata = file
        .metadata()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| model_name.clone());

    // Stream file instead of loading entirely into memory
    let stream = tokio_util::io::ReaderStream::new(file);
    let body = Body::from_stream(stream);

    let response = axum::response::Response::builder()
        .header("content-type", "application/octet-stream")
        .header(
            "content-disposition",
            format!("attachment; filename=\"{filename}\""),
        )
        .header("content-length", metadata.len().to_string())
        .body(body)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(response)
}

/// POST /marketplace/pull — download a model from a remote marketplace entry.
pub async fn pull(
    State(state): State<AppState>,
    Extension(_tenant_id): Extension<TenantId>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use axum::extract::Extension;
    use synapse_core::config::*;
    use synapse_types::marketplace::MarketplaceEntry;
    use synapse_types::model::{ModelFormat, QuantLevel};

    fn test_state(tmp: &tempfile::TempDir) -> AppState {
        let config = SynapseConfig {
            server: ServerConfig {
                bind: "127.0.0.1:0".into(),
                grpc_bind: "127.0.0.1:0".into(),
            },
            storage: StorageConfig {
                models_dir: tmp.path().join("models"),
                database: tmp.path().join("test.db"),
                cache_dir: tmp.path().join("cache"),
            },
            backends: BackendsConfig {
                default: "llamacpp".into(),
                enabled: vec!["llamacpp".into()],
            },
            training: TrainingConfig {
                executor: "subprocess".into(),
                trainer_image: None,
                max_concurrent_jobs: 2,
                checkpoints_dir: tmp.path().join("checkpoints"),
            },
            bridge: BridgeConfig {
                sy_endpoint: None,
                enabled: false,
                heartbeat_interval_secs: 10,
            },
            hardware: HardwareConfig {
                gpu_memory_reserve_mb: 512,
            },
            security: SecurityConfig::default(),
        };
        AppState::new(config).unwrap()
    }

    fn test_entry() -> MarketplaceEntry {
        MarketplaceEntry {
            model_name: "test-model".into(),
            description: Some("A test model".into()),
            format: ModelFormat::Gguf,
            quant: QuantLevel::Q4KM,
            size_bytes: 4_000_000_000,
            parameter_count: Some(7_000_000_000),
            architecture: Some("llama".into()),
            publisher_instance: "node-1".into(),
            download_url: "http://node-1:8420/marketplace/download/test-model".into(),
            sha256: Some("abc123".into()),
            tags: vec!["llama".into(), "chat".into()],
            published_at: chrono::Utc::now(),
            eval_scores: None,
        }
    }

    #[test]
    fn search_query_deserialize() {
        let json = r#"{"q": "llama", "format": "gguf", "max_size": 10000000000}"#;
        let q: SearchQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.q.unwrap(), "llama");
        assert_eq!(q.max_size, Some(10_000_000_000));
    }

    #[test]
    fn search_query_all_optional() {
        let json = r#"{}"#;
        let q: SearchQuery = serde_json::from_str(json).unwrap();
        assert!(q.q.is_none());
        assert!(q.format.is_none());
        assert!(q.max_size.is_none());
    }

    #[test]
    fn publish_request_deserialize() {
        let json = r#"{"model_name": "my-model"}"#;
        let req: PublishRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_name, "my-model");
    }

    #[test]
    fn pull_request_deserialize() {
        let json = r#"{
            "model_name": "remote-model",
            "source_url": "http://peer:8420/marketplace/download/remote-model"
        }"#;
        let req: PullRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_name, "remote-model");
        assert!(req.expected_sha256.is_none());
    }

    #[test]
    fn pull_request_with_sha256() {
        let json = r#"{
            "model_name": "remote-model",
            "source_url": "http://peer:8420/marketplace/download/remote-model",
            "expected_sha256": "deadbeef"
        }"#;
        let req: PullRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.expected_sha256.unwrap(), "deadbeef");
    }

    #[test]
    fn marketplace_entry_response_serializes() {
        let resp = MarketplaceEntryResponse {
            model_name: "test-model".into(),
            description: Some("desc".into()),
            format: ModelFormat::Gguf,
            size_bytes: 4_000_000_000,
            publisher_instance: "node-1".into(),
            download_url: "http://node-1/download".into(),
            sha256: Some("abc".into()),
            tags: vec!["llama".into()],
            published_at: chrono::Utc::now().to_rfc3339(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["model_name"], "test-model");
        assert_eq!(json["size_bytes"], 4_000_000_000u64);
        assert_eq!(json["tags"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn entry_to_response_conversion() {
        let entry = test_entry();
        let resp = entry_to_response(&entry);
        assert_eq!(resp.model_name, entry.model_name);
        assert_eq!(resp.description, entry.description);
        assert_eq!(resp.format, entry.format);
        assert_eq!(resp.size_bytes, entry.size_bytes);
        assert_eq!(resp.publisher_instance, entry.publisher_instance);
        assert_eq!(resp.download_url, entry.download_url);
        assert_eq!(resp.sha256, entry.sha256);
        assert_eq!(resp.tags, entry.tags);
    }

    #[tokio::test]
    async fn list_entries_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let result = list_entries(State(state), Extension(TenantId::default_tenant()))
            .await
            .unwrap();
        assert!(result.0.is_empty());
    }

    #[tokio::test]
    async fn list_entries_with_data() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        // Publish directly via catalog
        {
            let catalog = state.marketplace_catalog.lock().await;
            catalog
                .publish(&test_entry(), &TenantId::default_tenant())
                .unwrap();
        }

        let result = list_entries(State(state), Extension(TenantId::default_tenant()))
            .await
            .unwrap();
        assert_eq!(result.0.len(), 1);
        assert_eq!(result.0[0].model_name, "test-model");
    }

    #[tokio::test]
    async fn search_no_query() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        {
            let catalog = state.marketplace_catalog.lock().await;
            catalog
                .publish(&test_entry(), &TenantId::default_tenant())
                .unwrap();
        }

        let params = SearchQuery {
            q: None,
            format: None,
            max_size: None,
        };
        let result = search(
            State(state),
            Extension(TenantId::default_tenant()),
            Query(params),
        )
        .await
        .unwrap();
        assert_eq!(result.0.len(), 1);
    }

    #[tokio::test]
    async fn search_with_query() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        {
            let catalog = state.marketplace_catalog.lock().await;
            catalog
                .publish(&test_entry(), &TenantId::default_tenant())
                .unwrap();
        }

        let params = SearchQuery {
            q: Some("test".into()),
            format: None,
            max_size: None,
        };
        let result = search(
            State(state),
            Extension(TenantId::default_tenant()),
            Query(params),
        )
        .await
        .unwrap();
        assert_eq!(result.0.len(), 1);
    }

    #[tokio::test]
    async fn search_no_match() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        {
            let catalog = state.marketplace_catalog.lock().await;
            catalog
                .publish(&test_entry(), &TenantId::default_tenant())
                .unwrap();
        }

        let params = SearchQuery {
            q: Some("nonexistent-xyz".into()),
            format: None,
            max_size: None,
        };
        let result = search(
            State(state),
            Extension(TenantId::default_tenant()),
            Query(params),
        )
        .await
        .unwrap();
        assert!(result.0.is_empty());
    }

    #[tokio::test]
    async fn unpublish_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        {
            let catalog = state.marketplace_catalog.lock().await;
            catalog
                .publish(&test_entry(), &TenantId::default_tenant())
                .unwrap();
        }

        let result = unpublish(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Path("test-model".into()),
        )
        .await
        .unwrap();
        assert_eq!(result, StatusCode::NO_CONTENT);

        // Verify it's gone
        let entries = list_entries(State(state), Extension(TenantId::default_tenant()))
            .await
            .unwrap();
        assert!(entries.0.is_empty());
    }

    #[tokio::test]
    async fn unpublish_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = unpublish(
            State(state),
            Extension(TenantId::default_tenant()),
            Path("nonexistent".into()),
        )
        .await;
        assert_eq!(result.unwrap_err().0, StatusCode::NOT_FOUND);
    }
}
