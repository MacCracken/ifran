//! REST handlers for training job management (create, status, cancel, list, stream).

use std::convert::Infallible;

use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::response::sse::{Event, Sse};
use futures::Stream;
use serde::{Deserialize, Serialize};
use synapse_core::training_events::TrainingEvent;
use synapse_types::TenantId;
use synapse_types::training::{TrainingJobConfig, TrainingJobId, TrainingStatus};

use super::pagination::{PaginatedResponse, PaginationQuery};

use crate::rest::error::ApiErrorResponse;
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
    Extension(tenant_id): Extension<TenantId>,
    Json(req): Json<CreateJobRequest>,
) -> Result<(StatusCode, Json<JobResponse>), ApiErrorResponse> {
    let base_model = req.config.base_model.clone();

    let id = state
        .job_manager
        .create_job(req.config, tenant_id.clone())
        .await
        .map_err(|e| ApiErrorResponse::bad_request("INVALID_CONFIG", e.to_string()))?;

    if req.auto_start {
        if let Err(e) = state.job_manager.start_job(id, &tenant_id).await {
            tracing::warn!(job_id = %id, error = %e, "Auto-start failed for training job");
        }

        // Emit local training event
        state.training_event_bus.emit(TrainingEvent::JobStarted {
            job_id: id.to_string(),
            model: base_model,
            timestamp: chrono::Utc::now(),
        });

        // Also report to SY bridge if connected
        if let Some(client) = &state.bridge_client {
            let _ = client
                .report_progress(&id.to_string(), "running", 0, 0.0)
                .await;
        }
    }

    let job = state
        .job_manager
        .get_job(id, &tenant_id)
        .await
        .map_err(|e| ApiErrorResponse::internal(e.to_string()))?;

    Ok((StatusCode::CREATED, Json(job_to_response(&job))))
}

/// Query parameters for listing training jobs.
#[derive(Debug, Deserialize)]
pub struct ListJobsQuery {
    #[serde(flatten)]
    pub page: PaginationQuery,
    /// Optional status filter, e.g. `?status=running`.
    pub status: Option<TrainingStatus>,
}

/// GET /training/jobs — list training jobs with pagination.
pub async fn list_jobs(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Query(query): Query<ListJobsQuery>,
) -> Json<PaginatedResponse<JobResponse>> {
    let jobs = state.job_manager.list_jobs(query.status, &tenant_id).await;
    Json(PaginatedResponse::from_slice(
        &jobs,
        &query.page,
        job_to_response,
    ))
}

/// GET /training/jobs/:id — get a specific job's status.
pub async fn get_job(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<TrainingJobId>,
) -> Result<Json<JobResponse>, ApiErrorResponse> {
    let job = state
        .job_manager
        .get_job(id, &tenant_id)
        .await
        .map_err(|e| ApiErrorResponse::not_found("Training job", &e.to_string()))?;
    Ok(Json(job_to_response(&job)))
}

/// POST /training/jobs/:id/cancel — cancel a running or queued job.
pub async fn cancel_job(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<TrainingJobId>,
) -> Result<Json<JobResponse>, ApiErrorResponse> {
    state
        .job_manager
        .cancel_job(id, &tenant_id)
        .await
        .map_err(|e| ApiErrorResponse::bad_request("CANCEL_FAILED", e.to_string()))?;

    // Emit local cancellation event
    state.training_event_bus.emit(TrainingEvent::JobCancelled {
        job_id: id.to_string(),
        timestamp: chrono::Utc::now(),
    });

    // Also report cancellation to SY bridge
    if let Some(client) = &state.bridge_client {
        let _ = client
            .report_progress(&id.to_string(), "cancelled", 0, 0.0)
            .await;
    }

    let job = state
        .job_manager
        .get_job(id, &tenant_id)
        .await
        .map_err(|e| ApiErrorResponse::internal(e.to_string()))?;
    Ok(Json(job_to_response(&job)))
}

/// GET /training/jobs/:id/stream — SSE stream of job progress updates.
///
/// Polls the job every 2 seconds and emits JSON events until the job reaches a
/// terminal state (Completed, Failed, Cancelled). Returns 404 if the job does
/// not exist.
pub async fn stream_job(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<TrainingJobId>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiErrorResponse> {
    // Verify the job exists before starting the stream.
    state
        .job_manager
        .get_job(id, &tenant_id)
        .await
        .map_err(|e| ApiErrorResponse::not_found("Training job", &e.to_string()))?;

    let stream = async_stream::stream! {
        loop {
            let job = match state.job_manager.get_job(id, &tenant_id).await {
                Ok(j) => j,
                Err(e) => {
                    let payload = serde_json::json!({ "error": e.to_string() });
                    yield Ok(Event::default().data(payload.to_string()));
                    break;
                }
            };

            let payload = serde_json::json!({
                "status": job.status,
                "step": job.current_step,
                "total_steps": job.total_steps,
                "epoch": job.current_epoch,
                "loss": job.current_loss,
                "progress_percent": job.progress_percent(),
                "error": job.error,
            });
            yield Ok(Event::default().data(payload.to_string()));

            if job.is_terminal() {
                break;
            }

            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    };

    Ok(Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keep-alive"),
    ))
}

/// Response for training job metrics summary.
#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub job_id: TrainingJobId,
    pub current_step: u64,
    pub total_steps: u64,
    pub current_loss: Option<f64>,
    pub current_epoch: f32,
    pub progress_percent: f64,
    pub checkpoints: Vec<synapse_types::training::CheckpointInfo>,
}

/// Load checkpoints for a job from disk.
async fn load_checkpoints(
    state: &AppState,
    id: TrainingJobId,
) -> Result<Vec<synapse_types::training::CheckpointInfo>, ApiErrorResponse> {
    let checkpoints_dir = state.config.training.checkpoints_dir.clone();
    tokio::task::spawn_blocking(move || {
        synapse_train::checkpoint::store::CheckpointStore::new(checkpoints_dir).list(id)
    })
    .await
    .map_err(|e| ApiErrorResponse::internal(format!("Task failed: {e}")))?
    .map_err(|e| ApiErrorResponse::internal(e.to_string()))
}

/// GET /training/jobs/:id/checkpoints — list checkpoints for a training job.
pub async fn list_checkpoints(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<TrainingJobId>,
) -> Result<Json<Vec<synapse_types::training::CheckpointInfo>>, ApiErrorResponse> {
    state
        .job_manager
        .get_job(id, &tenant_id)
        .await
        .map_err(|e| ApiErrorResponse::not_found("Training job", &e.to_string()))?;

    Ok(Json(load_checkpoints(&state, id).await?))
}

/// GET /training/jobs/:id/metrics — get training metrics summary for a job.
pub async fn get_metrics(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<TrainingJobId>,
) -> Result<Json<MetricsResponse>, ApiErrorResponse> {
    let job = state
        .job_manager
        .get_job(id, &tenant_id)
        .await
        .map_err(|e| ApiErrorResponse::not_found("Training job", &e.to_string()))?;

    let checkpoints = load_checkpoints(&state, id).await?;

    Ok(Json(MetricsResponse {
        job_id: job.id,
        current_step: job.current_step,
        total_steps: job.total_steps,
        current_loss: job.current_loss,
        current_epoch: job.current_epoch,
        progress_percent: job.progress_percent(),
        checkpoints,
    }))
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
    use crate::rest::pagination::PaginationQuery;
    use crate::state::AppState;
    use axum::extract::{Extension, Query};
    use synapse_core::config::*;
    use synapse_types::TenantId;
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

        let result = create_job(
            State(state),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();
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

        let result = create_job(
            State(state),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();
        assert_eq!(result.0, StatusCode::CREATED);
        assert_eq!(result.1.status, TrainingStatus::Running);
    }

    #[tokio::test]
    async fn list_jobs_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let Json(resp) = list_jobs(
            State(state),
            Extension(TenantId::default_tenant()),
            Query(ListJobsQuery {
                page: PaginationQuery {
                    limit: 50,
                    offset: 0,
                },
                status: None,
            }),
        )
        .await;
        assert!(resp.data.is_empty());
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
        let _ = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req1),
        )
        .await
        .unwrap();
        let _ = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req2),
        )
        .await
        .unwrap();

        let Json(resp) = list_jobs(
            State(state),
            Extension(TenantId::default_tenant()),
            Query(ListJobsQuery {
                page: PaginationQuery {
                    limit: 50,
                    offset: 0,
                },
                status: None,
            }),
        )
        .await;
        assert_eq!(resp.data.len(), 2);
    }

    #[tokio::test]
    async fn get_job_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateJobRequest {
            config: test_job_config(),
            auto_start: false,
        };
        let (_, Json(created)) = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();

        let result = get_job(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(created.id),
        )
        .await
        .unwrap();
        assert_eq!(result.id, created.id);
        assert_eq!(result.status, TrainingStatus::Queued);
    }

    #[tokio::test]
    async fn get_job_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = get_job(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(uuid::Uuid::new_v4()),
        )
        .await;
        assert_eq!(result.unwrap_err().status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn cancel_job_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateJobRequest {
            config: test_job_config(),
            auto_start: false,
        };
        let (_, Json(created)) = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();

        let result = cancel_job(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(created.id),
        )
        .await
        .unwrap();
        assert_eq!(result.status, TrainingStatus::Cancelled);
    }

    #[tokio::test]
    async fn job_to_response_conversion() {
        let config = test_job_config();
        let id = uuid::Uuid::new_v4();
        let mut job =
            synapse_train::job::status::JobState::new(id, TenantId::default_tenant(), config, 1000);
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

    #[tokio::test]
    async fn stream_job_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = stream_job(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(uuid::Uuid::new_v4()),
        )
        .await;
        assert_eq!(result.unwrap_err().status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn stream_job_returns_sse() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        // Create a job then cancel it so the stream terminates immediately.
        let req = CreateJobRequest {
            config: test_job_config(),
            auto_start: false,
        };
        let (_, Json(created)) = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();
        let _ = cancel_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Path(created.id),
        )
        .await
        .unwrap();

        let app = crate::rest::router::build(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/training/jobs/{}/stream", created.id))
                    .header("Authorization", "Bearer test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let content_type = response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        assert!(
            content_type.contains("text/event-stream"),
            "expected SSE content type, got: {content_type}"
        );

        // Read the body and verify it contains SSE event data.
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8_lossy(&body);
        assert!(
            body_str.contains("cancelled"),
            "SSE body should contain cancelled status, got: {body_str}"
        );
    }

    #[test]
    fn metrics_response_serializes() {
        let resp = MetricsResponse {
            job_id: uuid::Uuid::new_v4(),
            current_step: 500,
            total_steps: 1000,
            current_loss: Some(0.35),
            current_epoch: 1.5,
            progress_percent: 50.0,
            checkpoints: vec![],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["current_step"], 500);
        assert_eq!(json["total_steps"], 1000);
        assert_eq!(json["progress_percent"], 50.0);
        assert!(json["checkpoints"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_checkpoints_job_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = list_checkpoints(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(uuid::Uuid::new_v4()),
        )
        .await;
        assert_eq!(result.unwrap_err().status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn list_checkpoints_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        // Create a job first
        let req = CreateJobRequest {
            config: test_job_config(),
            auto_start: false,
        };
        let (_, Json(created)) = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();

        let Json(checkpoints) = list_checkpoints(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(created.id),
        )
        .await
        .unwrap();
        assert!(checkpoints.is_empty());
    }

    #[tokio::test]
    async fn get_metrics_job_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = get_metrics(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(uuid::Uuid::new_v4()),
        )
        .await;
        assert_eq!(result.unwrap_err().status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_metrics_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateJobRequest {
            config: test_job_config(),
            auto_start: false,
        };
        let (_, Json(created)) = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();

        let Json(metrics) = get_metrics(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(created.id),
        )
        .await
        .unwrap();
        assert_eq!(metrics.job_id, created.id);
        assert_eq!(metrics.current_step, 0);
        assert!(metrics.checkpoints.is_empty());
    }
}
