//! REST handlers for training job management (create, status, cancel, list).

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use synapse_types::training::{TrainingJobConfig, TrainingJobId, TrainingStatus};

use crate::state::AppState;

/// Response body for a training job.
#[derive(Debug, Serialize)]
pub struct JobResponse {
    pub id: TrainingJobId,
    pub status: TrainingStatus,
    pub current_step: u64,
    pub total_steps: u64,
    pub current_epoch: f32,
    pub current_loss: Option<f64>,
    pub progress_percent: f64,
    pub error: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

/// Request body for creating a training job.
#[derive(Deserialize)]
pub struct CreateJobRequest {
    #[serde(flatten)]
    pub config: TrainingJobConfig,
    /// Whether to start the job immediately (default: true).
    #[serde(default = "default_true")]
    pub auto_start: bool,
}

fn default_true() -> bool {
    true
}

/// POST /training/jobs — create (and optionally start) a training job.
pub async fn create_job(
    State(state): State<AppState>,
    Json(req): Json<CreateJobRequest>,
) -> Result<(StatusCode, Json<JobResponse>), (StatusCode, String)> {
    let id = state
        .job_manager
        .create_job(req.config)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    if req.auto_start {
        if let Err(e) = state.job_manager.start_job(id).await {
            tracing::warn!(job_id = %id, error = %e, "Auto-start failed for training job");
        }

        // Report to SY bridge if connected
        if let Some(client) = &state.bridge_client {
            let _ = client
                .report_progress(&id.to_string(), "running", 0, 0.0)
                .await;
        }
    }

    let job = state
        .job_manager
        .get_job(id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(job_to_response(&job))))
}

/// GET /training/jobs — list all training jobs.
pub async fn list_jobs(State(state): State<AppState>) -> Json<Vec<JobResponse>> {
    let jobs = state.job_manager.list_jobs(None).await;
    Json(jobs.iter().map(job_to_response).collect())
}

/// GET /training/jobs/:id — get a specific job's status.
pub async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<TrainingJobId>,
) -> Result<Json<JobResponse>, (StatusCode, String)> {
    let job = state
        .job_manager
        .get_job(id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(Json(job_to_response(&job)))
}

/// POST /training/jobs/:id/cancel — cancel a running or queued job.
pub async fn cancel_job(
    State(state): State<AppState>,
    Path(id): Path<TrainingJobId>,
) -> Result<Json<JobResponse>, (StatusCode, String)> {
    state
        .job_manager
        .cancel_job(id)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Report cancellation to SY bridge
    if let Some(client) = &state.bridge_client {
        let _ = client
            .report_progress(&id.to_string(), "cancelled", 0, 0.0)
            .await;
    }

    let job = state
        .job_manager
        .get_job(id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(job_to_response(&job)))
}

fn job_to_response(job: &synapse_train::job::status::JobState) -> JobResponse {
    JobResponse {
        id: job.id,
        status: job.status,
        current_step: job.current_step,
        total_steps: job.total_steps,
        current_epoch: job.current_epoch,
        current_loss: job.current_loss,
        progress_percent: job.progress_percent(),
        error: job.error.clone(),
        created_at: job.created_at.to_rfc3339(),
        started_at: job.started_at.map(|t| t.to_rfc3339()),
        completed_at: job.completed_at.map(|t| t.to_rfc3339()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use synapse_core::config::*;
    use synapse_types::training::*;

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

    fn test_job_config() -> TrainingJobConfig {
        TrainingJobConfig {
            base_model: "llama-7b".into(),
            dataset: DatasetConfig {
                path: "/data/train.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: Some(100),
            },
            method: TrainingMethod::Lora,
            hyperparams: HyperParams {
                learning_rate: 2e-4,
                epochs: 1,
                batch_size: 4,
                gradient_accumulation_steps: 1,
                warmup_steps: 0,
                weight_decay: 0.0,
                max_seq_length: 512,
            },
            output_name: Some("test-output".into()),
            lora: None,
            max_steps: None,
            time_budget_secs: None,
        }
    }

    #[test]
    fn create_job_request_deserialize() {
        let json = r#"{
            "base_model": "llama-7b",
            "dataset": {"path": "/data/train.jsonl", "format": "jsonl"},
            "method": "lora",
            "hyperparams": {"learning_rate": 0.0002, "epochs": 3, "batch_size": 4, "gradient_accumulation_steps": 1, "warmup_steps": 100, "weight_decay": 0.01, "max_seq_length": 512}
        }"#;
        let req: CreateJobRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.config.base_model, "llama-7b");
        assert!(req.auto_start); // default true
    }

    #[test]
    fn create_job_request_auto_start_false() {
        let json = r#"{
            "base_model": "llama-7b",
            "dataset": {"path": "/data/train.jsonl", "format": "jsonl"},
            "method": "lora",
            "hyperparams": {"learning_rate": 0.0002, "epochs": 3, "batch_size": 4, "gradient_accumulation_steps": 1, "warmup_steps": 100, "weight_decay": 0.01, "max_seq_length": 512},
            "auto_start": false
        }"#;
        let req: CreateJobRequest = serde_json::from_str(json).unwrap();
        assert!(!req.auto_start);
    }

    #[test]
    fn job_response_serializes() {
        let resp = JobResponse {
            id: uuid::Uuid::new_v4(),
            status: TrainingStatus::Running,
            current_step: 50,
            total_steps: 100,
            current_epoch: 1.5,
            current_loss: Some(0.42),
            progress_percent: 50.0,
            error: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: Some(chrono::Utc::now().to_rfc3339()),
            completed_at: None,
        };

        let json = serde_json::to_value(&resp).unwrap();
        assert!(json["id"].is_string());
        assert_eq!(json["current_step"], 50);
        assert_eq!(json["total_steps"], 100);
        assert_eq!(json["progress_percent"], 50.0);
        assert!(json["completed_at"].is_null());
    }

    #[tokio::test]
    async fn create_job_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateJobRequest {
            config: test_job_config(),
            auto_start: false,
        };

        let result = create_job(State(state), Json(req)).await.unwrap();
        assert_eq!(result.0, StatusCode::CREATED);
        assert_eq!(result.1.status, TrainingStatus::Queued);
        assert_eq!(result.1.current_step, 0);
    }

    #[tokio::test]
    async fn create_job_with_auto_start() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateJobRequest {
            config: test_job_config(),
            auto_start: true,
        };

        let result = create_job(State(state), Json(req)).await.unwrap();
        assert_eq!(result.0, StatusCode::CREATED);
        assert_eq!(result.1.status, TrainingStatus::Running);
    }

    #[tokio::test]
    async fn list_jobs_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let Json(jobs) = list_jobs(State(state)).await;
        assert!(jobs.is_empty());
    }

    #[tokio::test]
    async fn list_jobs_with_data() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        // Create two jobs
        let req1 = CreateJobRequest {
            config: test_job_config(),
            auto_start: false,
        };
        let req2 = CreateJobRequest {
            config: test_job_config(),
            auto_start: false,
        };
        let _ = create_job(State(state.clone()), Json(req1)).await.unwrap();
        let _ = create_job(State(state.clone()), Json(req2)).await.unwrap();

        let Json(jobs) = list_jobs(State(state)).await;
        assert_eq!(jobs.len(), 2);
    }

    #[tokio::test]
    async fn get_job_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateJobRequest {
            config: test_job_config(),
            auto_start: false,
        };
        let (_, Json(created)) = create_job(State(state.clone()), Json(req)).await.unwrap();

        let result = get_job(State(state), Path(created.id)).await.unwrap();
        assert_eq!(result.id, created.id);
        assert_eq!(result.status, TrainingStatus::Queued);
    }

    #[tokio::test]
    async fn get_job_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = get_job(State(state), Path(uuid::Uuid::new_v4())).await;
        assert_eq!(result.unwrap_err().0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn cancel_job_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateJobRequest {
            config: test_job_config(),
            auto_start: false,
        };
        let (_, Json(created)) = create_job(State(state.clone()), Json(req)).await.unwrap();

        let result = cancel_job(State(state), Path(created.id)).await.unwrap();
        assert_eq!(result.status, TrainingStatus::Cancelled);
    }

    #[tokio::test]
    async fn job_to_response_conversion() {
        let config = test_job_config();
        let id = uuid::Uuid::new_v4();
        let mut job = synapse_train::job::status::JobState::new(id, config, 1000);
        job.start();
        job.update_progress(500, 1.5, 0.35);

        let resp = job_to_response(&job);
        assert_eq!(resp.id, id);
        assert_eq!(resp.status, TrainingStatus::Running);
        assert_eq!(resp.current_step, 500);
        assert_eq!(resp.total_steps, 1000);
        assert_eq!(resp.current_epoch, 1.5);
        assert_eq!(resp.current_loss, Some(0.35));
        assert_eq!(resp.progress_percent, 50.0);
        assert!(resp.started_at.is_some());
        assert!(resp.completed_at.is_none());
    }
}
