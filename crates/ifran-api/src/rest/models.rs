//! REST handlers for model management (list, get, remove).

use crate::rest::error::ApiErrorResponse;
use crate::state::AppState;
use axum::Json;
use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use ifran_types::TenantId;

use super::pagination::{PaginatedResponse, PaginationQuery};

/// GET /models — list models with pagination.
pub async fn list_models(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Query(page): Query<PaginationQuery>,
) -> Result<Json<PaginatedResponse<serde_json::Value>>, ApiErrorResponse> {
    let safe_limit = page.safe_limit();
    let paged = state
        .db
        .list(&tenant_id, safe_limit, page.offset)
        .map_err(|e| ApiErrorResponse::internal(e.to_string()))?;

    let data: Vec<serde_json::Value> = paged
        .items
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id.to_string(),
                "name": m.name,
                "repo_id": m.repo_id,
                "format": serde_json::to_value(m.format).unwrap_or(serde_json::Value::Null),
                "quant": serde_json::to_value(m.quant).unwrap_or(serde_json::Value::Null),
                "size_bytes": m.size_bytes,
                "parameter_count": m.parameter_count,
                "architecture": m.architecture,
                "local_path": m.local_path,
                "pulled_at": m.pulled_at.to_rfc3339(),
            })
        })
        .collect();

    Ok(Json(PaginatedResponse::pre_sliced(
        data,
        paged.total,
        safe_limit,
        page.offset,
    )))
}

/// GET /models/:id — get a specific model by ID.
pub async fn get_model(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiErrorResponse> {
    let model = if let Ok(uuid) = uuid::Uuid::parse_str(&id) {
        state.db.get(uuid, &tenant_id)
    } else {
        state.db.get_by_name(&id, &tenant_id)
    };

    let model = model.map_err(|_| {
        ApiErrorResponse::not_found("Model", &id).with_hint(
            "Use GET /models to list available models, or pull one with 'ifran pull <model>'",
        )
    })?;

    Ok(Json(serde_json::json!({
        "id": model.id.to_string(),
        "name": model.name,
        "repo_id": model.repo_id,
        "format": serde_json::to_value(model.format).unwrap_or(serde_json::Value::Null),
        "quant": serde_json::to_value(model.quant).unwrap_or(serde_json::Value::Null),
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
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiErrorResponse> {
    let model = if let Ok(uuid) = uuid::Uuid::parse_str(&id) {
        state.db.get(uuid, &tenant_id)
    } else {
        state.db.get_by_name(&id, &tenant_id)
    };

    let model = model.map_err(|_| {
        ApiErrorResponse::not_found("Model", &id).with_hint(
            "Use GET /models to list available models, or pull one with 'ifran pull <model>'",
        )
    })?;

    // Delete from database first — if this fails, filesystem stays consistent.
    // Reversing the order (FS first, then DB) risks orphaned DB records pointing
    // to deleted files.
    state
        .db
        .delete(model.id, &tenant_id)
        .map_err(|e| ApiErrorResponse::internal(e.to_string()))?;

    // Clean up files from disk (best-effort after successful DB delete)
    let local_path = std::path::Path::new(&model.local_path);
    if let Some(model_dir) = local_path.parent() {
        if model_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(model_dir) {
                tracing::warn!(path = %model_dir.display(), error = %e, "Failed to remove model files from disk");
            }
        }
    }

    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rest::pagination::PaginationQuery;
    use crate::state::AppState;
    use axum::extract::Query;
    use ifran_core::config::*;
    use ifran_core::storage::db::ModelDatabase;
    use ifran_types::model::{ModelFormat, ModelInfo, QuantLevel};

    fn test_state(tmp: &tempfile::TempDir) -> AppState {
        let config = IfranConfig {
            server: ServerConfig {
                bind: "127.0.0.1:0".into(),
                grpc_bind: "127.0.0.1:0".into(),
                ws_bind: None,
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
                job_eviction_ttl_secs: 86400,
            },
            bridge: BridgeConfig {
                sy_endpoint: None,
                enabled: false,
                heartbeat_interval_secs: 10,
            },
            hardware: HardwareConfig {
                gpu_memory_reserve_mb: 512,
                telemetry_interval_secs: 0,
            },
            security: SecurityConfig::default(),
            budget: BudgetConfig::default(),
            fleet: FleetConfig::default(),
        };
        AppState::new(config).unwrap()
    }

    fn insert_model(db_path: &std::path::Path) -> ModelInfo {
        let db = ModelDatabase::open(db_path).unwrap();
        let model = ModelInfo {
            id: uuid::Uuid::new_v4(),
            name: "test-model".into(),
            repo_id: Some("org/test-model".into()),
            format: ModelFormat::Gguf,
            quant: QuantLevel::Q4KM,
            size_bytes: 4_000_000_000,
            parameter_count: Some(7_000_000_000),
            architecture: Some("llama".into()),
            license: Some("MIT".into()),
            local_path: "/nonexistent/models/test-model/model.gguf".into(),
            sha256: Some("abc123".into()),
            pulled_at: chrono::Utc::now(),
        };
        db.insert(&model, &ifran_types::TenantId::default_tenant())
            .unwrap();
        model
    }

    fn default_page() -> Query<PaginationQuery> {
        Query(PaginationQuery {
            limit: 50,
            offset: 0,
        })
    }

    #[tokio::test]
    async fn list_models_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let result = list_models(
            State(state),
            Extension(TenantId::default_tenant()),
            default_page(),
        )
        .await
        .unwrap();
        assert_eq!(result.0.data.len(), 0);
    }

    #[tokio::test]
    async fn list_models_with_data() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        insert_model(&tmp.path().join("test.db"));

        let result = list_models(
            State(state),
            Extension(TenantId::default_tenant()),
            default_page(),
        )
        .await
        .unwrap();
        let data = &result.0.data;
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["name"], "test-model");
        assert_eq!(data[0]["format"], "gguf");
        assert!(data[0]["size_bytes"].is_number());
        assert!(data[0]["pulled_at"].is_string());
    }

    #[tokio::test]
    async fn get_model_by_name() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        insert_model(&tmp.path().join("test.db"));

        let result = get_model(
            State(state),
            Extension(TenantId::default_tenant()),
            Path("test-model".into()),
        )
        .await
        .unwrap();
        assert_eq!(result.0["name"], "test-model");
        assert_eq!(result.0["repo_id"], "org/test-model");
        assert_eq!(result.0["sha256"], "abc123");
        assert_eq!(result.0["architecture"], "llama");
    }

    #[tokio::test]
    async fn get_model_by_uuid() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let model = insert_model(&tmp.path().join("test.db"));

        let result = get_model(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(model.id.to_string()),
        )
        .await
        .unwrap();
        assert_eq!(result.0["name"], "test-model");
    }

    #[tokio::test]
    async fn get_model_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = get_model(
            State(state),
            Extension(TenantId::default_tenant()),
            Path("nonexistent".into()),
        )
        .await;
        assert_eq!(result.unwrap_err().status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_model_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let model = insert_model(&tmp.path().join("test.db"));

        let result = delete_model(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Path(model.id.to_string()),
        )
        .await
        .unwrap();
        assert_eq!(result, StatusCode::NO_CONTENT);

        // Verify model is gone
        let get_result = get_model(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(model.id.to_string()),
        )
        .await;
        assert_eq!(get_result.unwrap_err().status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_model_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = delete_model(
            State(state),
            Extension(TenantId::default_tenant()),
            Path("nonexistent".into()),
        )
        .await;
        assert_eq!(result.unwrap_err().status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn list_models_returns_all_fields() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        insert_model(&tmp.path().join("test.db"));

        let result = list_models(
            State(state),
            Extension(TenantId::default_tenant()),
            default_page(),
        )
        .await
        .unwrap();
        let model = &result.0.data[0];

        // Verify all expected fields are present
        assert!(model["id"].is_string());
        assert!(model["name"].is_string());
        assert!(model["repo_id"].is_string());
        assert!(model["format"].is_string());
        assert!(model["quant"].is_string());
        assert!(model["size_bytes"].is_number());
        assert!(model["parameter_count"].is_number());
        assert!(model["architecture"].is_string());
        assert!(model["local_path"].is_string());
        assert!(model["pulled_at"].is_string());
    }
}
