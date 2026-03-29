//! Integration tests for the Ifran REST API.
//!
//! These tests spin up the full Axum router with a temp SQLite database and
//! exercise the HTTP endpoints end-to-end.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use ifran::config::*;
use ifran::server::state::AppState;
use ifran::storage::db::ModelDatabase;
use ifran::types::model::{ModelFormat, ModelInfo, QuantLevel};
use tower::ServiceExt;

/// Create a test config pointing at a temp directory.
fn test_config(tmp: &tempfile::TempDir) -> IfranConfig {
    IfranConfig {
        server: ServerConfig {
            bind: "127.0.0.1:0".into(),
            grpc_bind: "127.0.0.1:0".into(),
            ws_bind: None,
        },
        storage: StorageConfig {
            models_dir: tmp.path().join("models"),
            database: tmp.path().join("test.db"),
            cache_dir: tmp.path().join("cache"),
            backend: Default::default(),
            postgres_url: None,
            postgres_pool_size: 8,
        },
        backends: BackendsConfig {
            default: "llamacpp".into(),
            enabled: vec![],
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
    }
}

/// Build a test router with temp state.
fn test_app() -> (axum::Router, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    let state = AppState::new(config).unwrap();
    let app = ifran::server::router::build(state);
    (app, tmp)
}

/// Build a test router with multi-tenant auth enabled.
/// Multi-tenant mode does NOT use the IFRAN_API_KEY env var — it resolves
/// keys via TenantStore, so it's safe for parallel tests.
fn test_app_multi_tenant() -> (axum::Router, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = test_config(&tmp);
    config.security.multi_tenant = true;
    let state = AppState::new(config).unwrap();
    let app = ifran::server::router::build(state);
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
    db.insert(&model, &ifran::types::TenantId::default_tenant())
        .unwrap();
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
    let app = ifran::server::router::build(state);

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
    let app = ifran::server::router::build(state);

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
    let app = ifran::server::router::build(state);

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
    let app = ifran::server::router::build(state);

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
    let app = ifran::server::router::build(state);

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
    assert!(json["data"].as_array().unwrap().is_empty());
    assert!(json["pagination"]["total"].is_number());
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
    assert!(json["data"].as_array().unwrap().is_empty());
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
    assert!(json["data"].as_array().unwrap().is_empty());
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
    assert!(json["data"].as_array().unwrap().is_empty());
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
    let app = ifran::server::router::build(state);

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
    assert!(json["data"].as_array().unwrap().is_empty());
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
    let app = ifran::server::router::build(state);

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
    assert!(!json["data"].as_array().unwrap().is_empty());

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
    assert!(json["data"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn marketplace_list_after_publish() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    insert_test_model(&config.storage.database);
    let state = AppState::new(config).unwrap();
    let app = ifran::server::router::build(state);

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
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
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
    let app = ifran::server::router::build(state);

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
    let app = ifran::server::router::build(state);

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
    // No real SY server running, so client gracefully degrades
    assert_eq!(json["client_state"], "Degraded");
}

#[tokio::test]
async fn bridge_heartbeat_when_enabled() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = test_config(&tmp);
    config.bridge.enabled = true;
    let state = AppState::new(config).unwrap();
    let app = ifran::server::router::build(state);

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

// -- Inference error paths --

#[tokio::test]
async fn inference_no_model_loaded() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello"
    });

    let resp = app
        .oneshot(
            Request::post("/inference")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("is not loaded"));
}

#[tokio::test]
async fn inference_stream_no_model_loaded() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "stream": true
    });

    let resp = app
        .oneshot(
            Request::post("/inference/stream")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn inference_invalid_json() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::post("/inference")
                .header("content-type", "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();

    // Axum returns 422 for JSON parse errors
    assert!(
        resp.status() == StatusCode::UNPROCESSABLE_ENTITY
            || resp.status() == StatusCode::BAD_REQUEST
    );
}

// -- Eval extended --

#[tokio::test]
async fn eval_create_and_get_run() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "model_name": "test-model",
        "benchmarks": ["perplexity", "mmlu"],
        "sample_limit": 50,
    });

    let resp = app
        .clone()
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
    let run_id = json["run_id"].as_str().unwrap();
    assert_eq!(json["model_name"], "test-model");
    assert_eq!(json["benchmarks"].as_array().unwrap().len(), 2);

    // Get the specific run
    let resp = app
        .oneshot(
            Request::get(format!("/eval/runs/{run_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["run_id"], run_id);
}

#[tokio::test]
async fn eval_list_after_create() {
    let (app, _tmp) = test_app();

    // Create two runs
    for name in &["model-a", "model-b"] {
        let body = serde_json::json!({
            "model_name": name,
            "benchmarks": ["custom"],
        });
        app.clone()
            .oneshot(
                Request::post("/eval/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
    }

    let resp = app
        .oneshot(Request::get("/eval/runs").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn eval_create_with_dataset_path() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "model_name": "test-model",
        "benchmarks": ["custom"],
        "dataset_path": "/tmp/eval.jsonl",
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
    assert_eq!(json["model_name"], "test-model");
    // Status may be queued (background task spawned but may not have started)
    assert!(json["status"].is_string());
}

#[tokio::test]
async fn eval_create_all_benchmark_kinds() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "model_name": "test-model",
        "benchmarks": ["perplexity", "mmlu", "hella_swag", "human_eval", "custom"],
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
    assert_eq!(json["benchmarks"].as_array().unwrap().len(), 5);
}

// -- Bridge extended --

#[tokio::test]
async fn bridge_status_includes_heartbeat_interval() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/bridge/status").body(Body::empty()).unwrap())
        .await
        .unwrap();

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["heartbeat_interval_secs"], 10);
}

#[tokio::test]
async fn bridge_status_no_endpoint_when_disabled() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/bridge/status").body(Body::empty()).unwrap())
        .await
        .unwrap();

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["sy_endpoint"].is_null());
}

#[tokio::test]
async fn bridge_enabled_connect_and_status() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = test_config(&tmp);
    config.bridge.enabled = true;
    let state = AppState::new(config).unwrap();
    let app = ifran::server::router::build(state);

    // Connect
    let resp = app
        .clone()
        .oneshot(
            Request::post("/bridge/connect")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify status after connect
    let resp = app
        .oneshot(Request::get("/bridge/status").body(Body::empty()).unwrap())
        .await
        .unwrap();

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["enabled"], true);
    // No real SY server running, so client gracefully degrades
    assert_eq!(json["client_state"], "Degraded");
}

// -- Training extended error paths --

#[tokio::test]
async fn training_create_invalid_hyperparams() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "base_model": "test-model",
        "dataset": {
            "path": "/tmp/test.jsonl",
            "format": "jsonl"
        },
        "method": "lora",
        "hyperparams": {
            "learning_rate": 0.0,
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

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn training_create_and_list() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "base_model": "test-model",
        "dataset": { "path": "/tmp/test.jsonl", "format": "jsonl" },
        "method": "lora",
        "hyperparams": {
            "learning_rate": 2e-4, "epochs": 1, "batch_size": 4,
            "gradient_accumulation_steps": 1, "warmup_steps": 0,
            "weight_decay": 0.0, "max_seq_length": 512
        },
        "auto_start": false
    });

    // Create two jobs
    for _ in 0..2 {
        app.clone()
            .oneshot(
                Request::post("/training/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
    }

    let resp = app
        .oneshot(Request::get("/training/jobs").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 2);
    assert_eq!(json["pagination"]["total"], 2);
}

// -- Model edge cases --

#[tokio::test]
async fn get_model_invalid_uuid_falls_through_to_name() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(
            Request::get("/models/not-a-uuid")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should try name lookup and return 404
    // Handler may return BAD_REQUEST or NOT_FOUND for nonexistent distributed jobs
    assert!(resp.status() == StatusCode::NOT_FOUND || resp.status() == StatusCode::BAD_REQUEST);
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

// -- Method not allowed --

#[tokio::test]
async fn health_post_not_allowed() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::post("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert!(
        resp.status() == StatusCode::METHOD_NOT_ALLOWED || resp.status() == StatusCode::NOT_FOUND
    );
}

// -- OpenAI extended --

#[tokio::test]
async fn openai_completions_no_model_loaded() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let resp = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// -- Marketplace extended --

#[tokio::test]
async fn marketplace_publish_nonexistent_model() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({"model_name": "nonexistent"});

    let resp = app
        .oneshot(
            Request::post("/marketplace/publish")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(
        resp.status() == StatusCode::NOT_FOUND
            || resp.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// -- Distributed extended --

#[tokio::test]
async fn distributed_assign_worker_not_found() {
    let (app, _tmp) = test_app();
    let fake_id = uuid::Uuid::new_v4();

    let worker_body = serde_json::json!({
        "rank": 0,
        "instance_id": "node-1",
        "endpoint": "http://node-1:9000",
        "device_ids": [0]
    });

    let resp = app
        .oneshot(
            Request::post(format!("/training/distributed/jobs/{fake_id}/workers"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&worker_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Handler returns BAD_REQUEST for nonexistent distributed jobs
    assert!(resp.status() == StatusCode::NOT_FOUND || resp.status() == StatusCode::BAD_REQUEST);
}

// -- RLHF --

#[tokio::test]
async fn rlhf_create_session() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "name": "test-session",
        "model_name": "llama-7b"
    });

    let resp = app
        .oneshot(
            Request::post("/rlhf/sessions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["name"], "test-session");
    assert_eq!(json["model_name"], "llama-7b");
    assert!(json["id"].is_string());
}

#[tokio::test]
async fn rlhf_list_sessions_empty() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/rlhf/sessions").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn rlhf_create_and_list_sessions() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    let state = AppState::new(config).unwrap();
    let app = ifran::server::router::build(state);

    // Create a session
    let body = serde_json::json!({
        "name": "session-1",
        "model_name": "llama-7b"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/rlhf/sessions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    // List sessions
    let resp = app
        .oneshot(Request::get("/rlhf/sessions").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn rlhf_add_pairs_and_get() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    let state = AppState::new(config).unwrap();
    let app = ifran::server::router::build(state);

    // Create session
    let body = serde_json::json!({
        "name": "pair-test",
        "model_name": "llama-7b"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/rlhf/sessions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let resp_body = resp.into_body().collect().await.unwrap().to_bytes();
    let session: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
    let session_id = session["id"].as_str().unwrap();

    // Add pairs
    let pairs_body = serde_json::json!({
        "pairs": [
            {"prompt": "What is Rust?", "response_a": "A language", "response_b": "A systems language"},
            {"prompt": "What is Python?", "response_a": "A scripting language", "response_b": "A general purpose language"}
        ]
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post(format!("/rlhf/sessions/{session_id}/pairs"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&pairs_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["added"], 2);

    // Get next unannotated pair
    let resp = app
        .clone()
        .oneshot(
            Request::get(format!("/rlhf/sessions/{session_id}/pairs"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["prompt"].is_string());

    // Get session detail with stats
    let resp = app
        .clone()
        .oneshot(
            Request::get(format!("/rlhf/sessions/{session_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["stats"]["total_pairs"], 2);
    assert_eq!(json["stats"]["remaining"], 2);

    // Get stats
    let resp = app
        .oneshot(
            Request::get(format!("/rlhf/sessions/{session_id}/stats"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["total_pairs"], 2);
}

#[tokio::test]
async fn rlhf_annotate_pair() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    let state = AppState::new(config).unwrap();
    let app = ifran::server::router::build(state);

    // Create session + add pair
    let session_body = serde_json::json!({"name": "annotate-test", "model_name": "llama"});
    let resp = app
        .clone()
        .oneshot(
            Request::post("/rlhf/sessions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&session_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let resp_body = resp.into_body().collect().await.unwrap().to_bytes();
    let session: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
    let session_id = session["id"].as_str().unwrap();

    let pairs_body = serde_json::json!({
        "pairs": [{"prompt": "test", "response_a": "a", "response_b": "b"}]
    });
    app.clone()
        .oneshot(
            Request::post(format!("/rlhf/sessions/{session_id}/pairs"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&pairs_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Get the pair ID
    let resp = app
        .clone()
        .oneshot(
            Request::get(format!("/rlhf/sessions/{session_id}/pairs"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let pair: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let pair_id = pair["id"].as_str().unwrap();

    // Annotate
    let annotate_body = serde_json::json!({"preference": "response_a"});
    let resp = app
        .clone()
        .oneshot(
            Request::post(format!("/rlhf/pairs/{pair_id}/annotate"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&annotate_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Export session
    let resp = app
        .oneshot(
            Request::post(format!("/rlhf/sessions/{session_id}/export"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["format"], "dpo_jsonl");
    assert_eq!(json["count"], 1);
}

// -- RAG --

#[tokio::test]
async fn rag_create_pipeline() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "name": "test-rag",
        "embedding_model": "nomic-embed",
        "chunk_size": 256,
        "chunk_overlap": 32,
        "similarity_top_k": 3
    });

    let resp = app
        .oneshot(
            Request::post("/rag/pipelines")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["name"], "test-rag");
}

#[tokio::test]
async fn rag_list_pipelines_empty() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/rag/pipelines").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn rag_pipeline_lifecycle() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    let state = AppState::new(config).unwrap();
    let app = ifran::server::router::build(state);

    // Create pipeline
    let body = serde_json::json!({
        "name": "lifecycle-test",
        "embedding_model": "nomic-embed"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/rag/pipelines")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let resp_body = resp.into_body().collect().await.unwrap().to_bytes();
    let pipeline: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
    let pipeline_id = pipeline["id"].as_str().unwrap();

    // Get pipeline
    let resp = app
        .clone()
        .oneshot(
            Request::get(format!("/rag/pipelines/{pipeline_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["name"], "lifecycle-test");
    assert_eq!(json["chunk_size"], 512); // default

    // List pipelines
    let resp = app
        .clone()
        .oneshot(Request::get("/rag/pipelines").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 1);

    // Ingest a document
    let ingest_body = serde_json::json!({
        "filename": "test.txt",
        "content": "Rust is a systems programming language focused on safety and performance."
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post(format!("/rag/pipelines/{pipeline_id}/ingest"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&ingest_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["document_id"].is_string());
    assert!(json["chunks"].as_u64().unwrap() >= 1);

    // Query
    let query_body = serde_json::json!({
        "query": "What is Rust?",
        "pipeline_id": pipeline_id,
        "top_k": 3,
        "include_sources": true
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/rag/query")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&query_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["sources"].is_array());

    // Delete pipeline
    let resp = app
        .oneshot(
            Request::delete(format!("/rag/pipelines/{pipeline_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
}

// -- Eval extended --

#[tokio::test]
async fn eval_create_run_with_benchmarks() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "model_name": "llama-7b",
        "benchmarks": ["custom"],
        "sample_limit": 10
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
    assert_eq!(json["model_name"], "llama-7b");
    assert!(json["run_id"].is_string());
}

#[tokio::test]
async fn eval_list_runs() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/eval/runs").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["data"].is_array());
    assert!(json["pagination"].is_object());
}

#[tokio::test]
async fn eval_create_get_and_list_run() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = test_config(&tmp);
    let state = AppState::new(config).unwrap();
    let app = ifran::server::router::build(state);

    // Create
    let body = serde_json::json!({
        "model_name": "test-model",
        "benchmarks": ["perplexity", "mmlu"]
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/eval/runs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let resp_body = resp.into_body().collect().await.unwrap().to_bytes();
    let run: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
    let run_id = run["run_id"].as_str().unwrap();

    // Get
    let resp = app
        .oneshot(
            Request::get(format!("/eval/runs/{run_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["model_name"], "test-model");
}

#[tokio::test]
async fn eval_get_nonexistent_run() {
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

// -- Lineage --

#[tokio::test]
async fn lineage_record_and_get() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "stage": "dataset",
        "name": "train.jsonl",
        "artifact_ref": "/data/train.jsonl"
    });

    let resp = app
        .clone()
        .oneshot(
            Request::post("/lineage")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::CREATED);
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    let id = json["id"].as_str().unwrap().to_string();

    // GET it back
    let resp = app
        .oneshot(
            Request::get(format!("/lineage/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    assert_eq!(json["name"], "train.jsonl");
    assert_eq!(json["artifact_ref"], "/data/train.jsonl");
}

#[tokio::test]
async fn lineage_list_empty() {
    let (app, _tmp) = test_app();

    let resp = app
        .oneshot(Request::get("/lineage").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["data"].as_array().unwrap().is_empty());
    assert!(json["pagination"].is_object());
}

#[tokio::test]
async fn lineage_ancestry() {
    let (app, _tmp) = test_app();

    // Create node A (dataset)
    let body_a = serde_json::json!({
        "stage": "dataset",
        "name": "data",
        "artifact_ref": "/data"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/lineage")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_a).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json_a: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    let id_a = json_a["id"].as_str().unwrap().to_string();

    // Create node B (training) with parent A
    let body_b = serde_json::json!({
        "stage": "training",
        "name": "lora-run",
        "artifact_ref": "job-1",
        "parent_ids": [id_a]
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/lineage")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_b).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json_b: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    let id_b = json_b["id"].as_str().unwrap().to_string();

    // GET ancestry of B — should include B and A
    let resp = app
        .oneshot(
            Request::get(format!("/lineage/{id_b}/ancestry"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    assert_eq!(json.as_array().unwrap().len(), 2);
}

// -- Versions --

#[tokio::test]
async fn version_create_and_get() {
    let (app, _tmp) = test_app();

    let body = serde_json::json!({
        "model_family": "llama-8b",
        "version_tag": "v1"
    });

    let resp = app
        .clone()
        .oneshot(
            Request::post("/versions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::CREATED);
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    let id = json["id"].as_str().unwrap().to_string();
    assert_eq!(json["model_family"], "llama-8b");

    // GET it back
    let resp = app
        .oneshot(
            Request::get(format!("/versions/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    assert_eq!(json["version_tag"], "v1");
}

#[tokio::test]
async fn version_list_by_family() {
    let (app, _tmp) = test_app();

    // Create two versions in the same family
    for tag in ["v1", "v2"] {
        let body = serde_json::json!({
            "model_family": "llama-8b",
            "version_tag": tag
        });
        app.clone()
            .oneshot(
                Request::post("/versions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
    }

    // Create one in a different family
    let body = serde_json::json!({
        "model_family": "mistral-7b",
        "version_tag": "v1"
    });
    app.clone()
        .oneshot(
            Request::post("/versions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // List by family=llama-8b
    let resp = app
        .oneshot(
            Request::get("/versions?family=llama-8b")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn version_lineage_chain() {
    let (app, _tmp) = test_app();

    // Create v1 (no parent)
    let body_v1 = serde_json::json!({
        "model_family": "llama-8b",
        "version_tag": "v1"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/versions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_v1).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json_v1: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    let id_v1 = json_v1["id"].as_str().unwrap().to_string();

    // Create v2 with parent v1
    let body_v2 = serde_json::json!({
        "model_family": "llama-8b",
        "version_tag": "v2",
        "parent_version_id": id_v1
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/versions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_v2).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json_v2: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    let id_v2 = json_v2["id"].as_str().unwrap().to_string();

    // GET lineage of v2 — should return [v2, v1]
    let resp = app
        .oneshot(
            Request::get(format!("/versions/{id_v2}/lineage"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let rbody = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&rbody).unwrap();
    let chain = json.as_array().unwrap();
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0]["version_tag"], "v2");
    assert_eq!(chain[1]["version_tag"], "v1");
}

// -- Auth & Permission Tests --

// 1. Auth bypass on probe endpoints (uses multi-tenant to avoid env var races)

#[tokio::test]
async fn health_accessible_without_auth() {
    let (app, _tmp) = test_app_multi_tenant();
    // No Authorization header — should still get 200
    let resp = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn ready_accessible_without_auth() {
    let (app, _tmp) = test_app_multi_tenant();
    let resp = app
        .oneshot(Request::get("/ready").body(Body::empty()).unwrap())
        .await
        .unwrap();
    // ready returns 200 or 503 depending on DB state — just verify it's not 401
    assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn metrics_accessible_without_auth() {
    let (app, _tmp) = test_app_multi_tenant();
    let resp = app
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
}

// Single-tenant auth token extraction and constant-time comparison are tested
// via unit tests in src/server/middleware/auth.rs. Integration tests for
// single-tenant auth are intentionally omitted because they require setting
// the IFRAN_API_KEY process-wide env var, which races with parallel tests.

// 3. Multi-tenant auth

#[tokio::test]
async fn multi_tenant_rejects_without_token() {
    let (app, _tmp) = test_app_multi_tenant();

    let resp = app
        .oneshot(Request::get("/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn multi_tenant_rejects_unknown_key() {
    let (app, _tmp) = test_app_multi_tenant();
    let resp = app
        .oneshot(
            Request::get("/models")
                .header("authorization", "Bearer unknown-tenant-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// 4. Open access mode (no key configured)

#[tokio::test]
async fn open_access_when_no_key_configured() {
    // Ensure no key is set
    let (app, _tmp) = test_app();
    let resp = app
        .oneshot(Request::get("/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// 5. Request ID propagation

#[tokio::test]
async fn request_id_generated_when_missing() {
    let (app, _tmp) = test_app();
    let resp = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let request_id = resp.headers().get("x-request-id");
    assert!(request_id.is_some());
    // Should be a valid UUID
    let id_str = request_id.unwrap().to_str().unwrap();
    assert!(uuid::Uuid::parse_str(id_str).is_ok());
}

#[tokio::test]
async fn request_id_echoed_from_client() {
    let (app, _tmp) = test_app();
    let resp = app
        .oneshot(
            Request::get("/health")
                .header("x-request-id", "my-custom-id-123")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let request_id = resp
        .headers()
        .get("x-request-id")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(request_id, "my-custom-id-123");
}
