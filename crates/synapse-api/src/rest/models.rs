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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use synapse_core::config::*;
    use synapse_core::storage::db::ModelDatabase;
    use synapse_types::model::{ModelFormat, ModelInfo, QuantLevel};

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
        db.insert(&model).unwrap();
        model
    }

    #[tokio::test]
    async fn list_models_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let result = list_models(State(state)).await.unwrap();
        let json = result.0;
        assert_eq!(json["data"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn list_models_with_data() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        insert_model(&tmp.path().join("test.db"));

        let result = list_models(State(state)).await.unwrap();
        let data = result.0["data"].as_array().unwrap();
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

        let result = get_model(State(state), Path("test-model".into()))
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

        let result = get_model(State(state), Path(model.id.to_string()))
            .await
            .unwrap();
        assert_eq!(result.0["name"], "test-model");
    }

    #[tokio::test]
    async fn get_model_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = get_model(State(state), Path("nonexistent".into())).await;
        assert_eq!(result.unwrap_err(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_model_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let model = insert_model(&tmp.path().join("test.db"));

        let result = delete_model(State(state.clone()), Path(model.id.to_string()))
            .await
            .unwrap();
        assert_eq!(result, StatusCode::NO_CONTENT);

        // Verify model is gone
        let get_result = get_model(State(state), Path(model.id.to_string())).await;
        assert_eq!(get_result.unwrap_err(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_model_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = delete_model(State(state), Path("nonexistent".into())).await;
        assert_eq!(result.unwrap_err(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn list_models_returns_all_fields() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        insert_model(&tmp.path().join("test.db"));

        let result = list_models(State(state)).await.unwrap();
        let model = &result.0["data"][0];

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
