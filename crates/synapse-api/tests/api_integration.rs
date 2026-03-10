//! Integration tests for the Synapse REST API.
//!
//! These tests spin up the full Axum router with a temp SQLite database and
//! exercise the HTTP endpoints end-to-end.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use synapse_api::state::AppState;
use synapse_core::config::*;
use synapse_core::storage::db::ModelDatabase;
use synapse_types::model::{ModelFormat, ModelInfo, QuantLevel};
use tower::ServiceExt;

/// Create a test config pointing at a temp directory.
fn test_config(tmp: &tempfile::TempDir) -> SynapseConfig {
    SynapseConfig {
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
    }
}

/// Build a test router with temp state.
fn test_app() -> (axum::Router, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);
    (app, tmp)
}

/// Insert a test model into the database.
fn insert_test_model(db_path: &std::path::Path) -> ModelInfo {
    let db = ModelDatabase::open(db_path).unwrap();
    // Use a path under a non-existent subdirectory so the delete handler's
    // remove_dir_all on parent() is harmless.
    let model = ModelInfo {
        id: uuid::Uuid::new_v4(),
        name: "test-model".into(),
        repo_id: Some("test-org/test-model".into()),
        format: ModelFormat::Gguf,
        quant: QuantLevel::Q4KM,
        size_bytes: 4_000_000_000,
        parameter_count: Some(7_000_000_000),
        architecture: Some("llama".into()),
        license: Some("MIT".into()),
        local_path: "/nonexistent/models/test-model/test-model.gguf".into(),
        sha256: Some("abcdef1234567890".into()),
        pulled_at: chrono::Utc::now(),
    };
    db.insert(&model).unwrap();
    model
}

// -- Health & System --

#[tokio::test]
async fn health_returns_ok() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&body[..], b"ok");
}

#[tokio::test]
async fn system_status_returns_json() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/system/status").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["version"].is_string());
    assert!(json["loaded_models"].is_number());
    assert!(json["hardware"].is_object());
}

// -- Models --

#[tokio::test]
async fn list_models_empty() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/models").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn list_models_with_data() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    insert_test_model(&config.storage.database);
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    let resp = app
        .oneshot(Request::get("/models").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let models = json["data"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], "test-model");
}

#[tokio::test]
async fn get_model_by_name() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    insert_test_model(&config.storage.database);
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    let resp = app
        .oneshot(
            Request::get("/models/test-model")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["name"], "test-model");
    assert_eq!(json["format"], "gguf");
}

#[tokio::test]
async fn get_model_by_uuid() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    let model = insert_test_model(&config.storage.database);
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    let resp = app
        .oneshot(
            Request::get(&format!("/models/{}", model.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn get_model_not_found() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::get("/models/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_model() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    let model = insert_test_model(&config.storage.database);
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    let resp = app
        .oneshot(
            Request::delete(&format!("/models/{}", model.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn delete_model_not_found() {
    let (app, _tmp) = test_app();
    let fake_id = uuid::Uuid::new_v4();

    let resp = app
        .oneshot(
            Request::delete(&format!("/models/{fake_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// -- OpenAI-compatible --

#[tokio::test]
async fn openai_list_models() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    insert_test_model(&config.storage.database);
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    let resp = app
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "list");
    assert!(!json["data"].as_array().unwrap().is_empty());
}

// -- Training --

#[tokio::test]
async fn training_list_jobs_empty() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::get("/training/jobs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn training_create_job() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "base_model": "test-model",
        "dataset": {
            "path": "/tmp/test.jsonl",
            "format": "jsonl"
        },
        "method": "lora",
        "hyperparams": {
            "learning_rate": 2e-4,
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 10,
            "weight_decay": 0.01,
            "max_seq_length": 512
        },
        "auto_start": false
    });

    let resp = app
        .oneshot(
            Request::post("/training/jobs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["id"].is_string());
    assert_eq!(json["status"], "queued");
}

#[tokio::test]
async fn training_get_job_not_found() {
    let (app, _tmp) = test_app();
    let fake_id = uuid::Uuid::new_v4();

    let resp = app
        .oneshot(
            Request::get(&format!("/training/jobs/{fake_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// -- 404 for unknown routes --

#[tokio::test]
async fn unknown_route_returns_404() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::get("/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
