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
            Request::get(format!("/models/{}", model.id))
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
            Request::delete(format!("/models/{}", model.id))
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
            Request::delete(format!("/models/{fake_id}"))
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
        .oneshot(Request::get("/training/jobs").body(Body::empty()).unwrap())
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
            Request::get(format!("/training/jobs/{fake_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// -- Eval --

#[tokio::test]
async fn eval_list_runs_empty() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/eval/runs").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn eval_create_run() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "model_name": "test-model",
        "benchmarks": ["custom"],
        "sample_limit": 10,
    });

    let resp = app
        .oneshot(
            Request::post("/eval/runs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["run_id"].is_string());
    assert_eq!(json["status"], "queued");
    assert_eq!(json["model_name"], "test-model");
}

#[tokio::test]
async fn eval_get_run_not_found() {
    let (app, _tmp) = test_app();
    let fake_id = uuid::Uuid::new_v4();

    let resp = app
        .oneshot(
            Request::get(format!("/eval/runs/{fake_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// -- Marketplace --

#[tokio::test]
async fn marketplace_search_empty() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::get("/marketplace/search")
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
async fn marketplace_list_entries_empty() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::get("/marketplace/entries")
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
async fn marketplace_unpublish_not_found() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::delete("/marketplace/entries/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn marketplace_download_not_found() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::get("/marketplace/download/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn marketplace_publish_model() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    insert_test_model(&config.storage.database);
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    let body = serde_json::json!({"model_name": "test-model"});

    let resp = app
        .oneshot(
            Request::post("/marketplace/publish")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["model_name"], "test-model");
    assert!(json["download_url"].is_string());
}

// -- Distributed Training --

#[tokio::test]
async fn distributed_list_jobs_empty() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::get("/training/distributed/jobs")
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
async fn distributed_create_job() {
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
        "world_size": 2,
        "strategy": "data_parallel"
    });

    let resp = app
        .oneshot(
            Request::post("/training/distributed/jobs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["job_id"].is_string());
    assert_eq!(json["status"], "queued");
    assert_eq!(json["world_size"], 2);
}

#[tokio::test]
async fn distributed_get_job_not_found() {
    let (app, _tmp) = test_app();
    let fake_id = uuid::Uuid::new_v4();

    let resp = app
        .oneshot(
            Request::get(format!("/training/distributed/jobs/{fake_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// -- Distributed Training Full Workflow --

#[tokio::test]
async fn distributed_assign_worker() {
    let (app, _tmp) = test_app();

    // Create job first
    let create_body = serde_json::json!({
        "base_model": "test-model",
        "dataset": { "path": "/tmp/test.jsonl", "format": "jsonl" },
        "method": "lora",
        "hyperparams": {
            "learning_rate": 2e-4, "epochs": 1, "batch_size": 4,
            "gradient_accumulation_steps": 1, "warmup_steps": 0,
            "weight_decay": 0.0, "max_seq_length": 512
        },
        "world_size": 2,
        "strategy": "data_parallel"
    });

    let resp = app
        .clone()
        .oneshot(
            Request::post("/training/distributed/jobs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&create_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let job_id = json["job_id"].as_str().unwrap();

    // Assign worker
    let worker_body = serde_json::json!({
        "rank": 0,
        "instance_id": "node-1",
        "endpoint": "http://node-1:9000",
        "device_ids": [0]
    });

    let resp = app
        .oneshot(
            Request::post(format!("/training/distributed/jobs/{job_id}/workers"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&worker_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["workers"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn distributed_full_lifecycle() {
    let (app, _tmp) = test_app();

    // Create job
    let create_body = serde_json::json!({
        "base_model": "test-model",
        "dataset": { "path": "/tmp/test.jsonl", "format": "jsonl" },
        "method": "lora",
        "hyperparams": {
            "learning_rate": 2e-4, "epochs": 1, "batch_size": 4,
            "gradient_accumulation_steps": 1, "warmup_steps": 0,
            "weight_decay": 0.0, "max_seq_length": 512
        },
        "world_size": 2,
        "strategy": "data_parallel"
    });

    let resp = app
        .clone()
        .oneshot(
            Request::post("/training/distributed/jobs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&create_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let job_id = json["job_id"].as_str().unwrap().to_string();

    // Assign 2 workers
    for rank in 0..2u32 {
        let worker_body = serde_json::json!({
            "rank": rank,
            "instance_id": format!("node-{}", rank + 1),
            "endpoint": format!("http://node-{}:9000", rank + 1),
            "device_ids": [0]
        });
        app.clone()
            .oneshot(
                Request::post(format!("/training/distributed/jobs/{job_id}/workers"))
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&worker_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
    }

    // Start job
    let resp = app
        .clone()
        .oneshot(
            Request::post(format!("/training/distributed/jobs/{job_id}/start"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "running");

    // Complete workers
    for rank in 0..2u32 {
        app.clone()
            .oneshot(
                Request::post(format!(
                    "/training/distributed/jobs/{job_id}/workers/{rank}/complete"
                ))
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
    }

    // Verify completed
    let resp = app
        .oneshot(
            Request::get(format!("/training/distributed/jobs/{job_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "completed");
    assert_eq!(json["completed_workers"], 2);
}

#[tokio::test]
async fn distributed_fail_job() {
    let (app, _tmp) = test_app();

    let create_body = serde_json::json!({
        "base_model": "test-model",
        "dataset": { "path": "/tmp/test.jsonl", "format": "jsonl" },
        "method": "lora",
        "hyperparams": {
            "learning_rate": 2e-4, "epochs": 1, "batch_size": 4,
            "gradient_accumulation_steps": 1, "warmup_steps": 0,
            "weight_decay": 0.0, "max_seq_length": 512
        },
        "world_size": 2,
        "strategy": "data_parallel"
    });

    let resp = app
        .clone()
        .oneshot(
            Request::post("/training/distributed/jobs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&create_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let job_id = json["job_id"].as_str().unwrap();

    let resp = app
        .oneshot(
            Request::post(format!("/training/distributed/jobs/{job_id}/fail"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "failed");
}

#[tokio::test]
async fn marketplace_search_with_query() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    insert_test_model(&config.storage.database);
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    // Publish first
    let pub_body = serde_json::json!({"model_name": "test-model"});
    app.clone()
        .oneshot(
            Request::post("/marketplace/publish")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&pub_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Search with matching query
    let resp = app
        .clone()
        .oneshot(
            Request::get("/marketplace/search?q=test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(!json.as_array().unwrap().is_empty());

    // Search with non-matching query
    let resp = app
        .oneshot(
            Request::get("/marketplace/search?q=nonexistent")
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
async fn marketplace_list_after_publish() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    insert_test_model(&config.storage.database);
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    let pub_body = serde_json::json!({"model_name": "test-model"});
    app.clone()
        .oneshot(
            Request::post("/marketplace/publish")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&pub_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let resp = app
        .oneshot(
            Request::get("/marketplace/entries")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json.as_array().unwrap().len(), 1);
}

// -- Bridge --

#[tokio::test]
async fn bridge_status_disabled() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/bridge/status").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["enabled"], false);
    assert_eq!(json["client_state"], "disabled");
    assert_eq!(json["server_state"], "disabled");
}

#[tokio::test]
async fn bridge_connect_when_disabled() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::post("/bridge/connect")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn bridge_heartbeat_when_disabled() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::post("/bridge/heartbeat")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn bridge_status_enabled() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = test_config(&tmp);
    config.bridge.enabled = true;
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    let resp = app
        .oneshot(Request::get("/bridge/status").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["enabled"], true);
    assert_eq!(json["client_state"], "Disconnected");
    assert_eq!(json["server_state"], "Disconnected");
}

#[tokio::test]
async fn bridge_connect_when_enabled() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = test_config(&tmp);
    config.bridge.enabled = true;
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    let resp = app
        .oneshot(
            Request::post("/bridge/connect")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["enabled"], true);
    assert_eq!(json["client_state"], "Connected");
    assert_eq!(json["server_state"], "Connected");
}

#[tokio::test]
async fn bridge_heartbeat_when_enabled() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = test_config(&tmp);
    config.bridge.enabled = true;
    let state = AppState::new(config).unwrap();
    let app = synapse_api::router::build(state);

    // Connect first
    app.clone()
        .oneshot(
            Request::post("/bridge/connect")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let resp = app
        .oneshot(
            Request::post("/bridge/heartbeat")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["instance_id"].is_string());
    assert!(json["timestamp"].is_number());
    assert!(json["loaded_models"].is_number());
    assert!(json["gpu_memory_free_mb"].is_number());
    assert!(json["active_training_jobs"].is_number());
}

#[tokio::test]
async fn system_status_includes_bridge() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/system/status").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["bridge"].is_object());
    assert_eq!(json["bridge"]["enabled"], false);
}

// -- 404 for unknown routes --

#[tokio::test]
async fn unknown_route_returns_404() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/nonexistent").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
