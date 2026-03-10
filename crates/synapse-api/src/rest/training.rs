//! REST handlers for training job management (create, status, cancel, list).

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use synapse_types::training::{TrainingJobConfig, TrainingJobId, TrainingStatus};

use crate::state::AppState;

/// Response body for a training job.
#[derive(Serialize)]
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
        let _ = state.job_manager.start_job(id).await;
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
