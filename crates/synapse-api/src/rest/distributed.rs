//! REST handlers for distributed training job management.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use synapse_types::distributed::*;
use synapse_types::training::{TrainingJobConfig, TrainingStatus};

use crate::state::AppState;

/// Response for a distributed training job.
#[derive(Serialize)]
pub struct DistributedJobResponse {
    pub job_id: DistributedJobId,
    pub coordinator: String,
    pub world_size: u32,
    pub strategy: DistributedStrategy,
    pub workers: Vec<WorkerAssignment>,
    pub status: TrainingStatus,
    pub aggregate_loss: Option<f64>,
    pub completed_workers: u32,
}

/// Request to create a distributed training job.
#[derive(Deserialize)]
pub struct CreateDistributedJobRequest {
    #[serde(flatten)]
    pub base_config: TrainingJobConfig,
    pub world_size: u32,
    pub strategy: DistributedStrategy,
}

/// Request to assign a worker to a distributed job.
#[derive(Deserialize)]
pub struct AssignWorkerRequest {
    pub rank: u32,
    pub instance_id: String,
    pub endpoint: String,
    #[serde(default)]
    pub device_ids: Vec<u32>,
}

/// Request to trigger checkpoint aggregation.
#[derive(Deserialize)]
pub struct AggregateRequest {
    pub output_dir: String,
    #[serde(default = "default_method")]
    pub method: String,
}

fn default_method() -> String {
    "average".into()
}

/// POST /training/distributed/jobs — create a distributed training job.
pub async fn create_job(
    State(state): State<AppState>,
    Json(req): Json<CreateDistributedJobRequest>,
) -> Result<(StatusCode, Json<DistributedJobResponse>), (StatusCode, String)> {
    let config = DistributedTrainingConfig {
        base_config: req.base_config,
        world_size: req.world_size,
        strategy: req.strategy,
    };

    let instance_id =
        std::env::var("SYNAPSE_INSTANCE_ID").unwrap_or_else(|_| state.config.server.bind.clone());

    let job_id = state
        .distributed_coordinator
        .create_job(config, &instance_id)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let job = state
        .distributed_coordinator
        .get_job(job_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(job_to_response(&job))))
}

/// GET /training/distributed/jobs — list all distributed jobs.
pub async fn list_jobs(State(state): State<AppState>) -> Json<Vec<DistributedJobResponse>> {
    let jobs = state.distributed_coordinator.list_jobs().await;
    Json(jobs.iter().map(job_to_response).collect())
}

/// GET /training/distributed/jobs/:id — get a distributed job.
pub async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<DistributedJobId>,
) -> Result<Json<DistributedJobResponse>, (StatusCode, String)> {
    let job = state
        .distributed_coordinator
        .get_job(id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(Json(job_to_response(&job)))
}

/// POST /training/distributed/jobs/:id/workers — assign a worker.
pub async fn assign_worker(
    State(state): State<AppState>,
    Path(id): Path<DistributedJobId>,
    Json(req): Json<AssignWorkerRequest>,
) -> Result<Json<DistributedJobResponse>, (StatusCode, String)> {
    let worker = WorkerAssignment {
        rank: req.rank,
        instance_id: req.instance_id,
        endpoint: req.endpoint,
        device_ids: req.device_ids,
    };

    state
        .distributed_coordinator
        .assign_worker(id, worker.clone())
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Notify SY bridge of worker assignment for cross-node coordination
    if let Some(client) = &state.bridge_client {
        let _ = client
            .request_worker_assignment(
                &id.to_string(),
                worker.rank,
                &worker.endpoint,
                &worker.device_ids,
            )
            .await;
    }

    let job = state
        .distributed_coordinator
        .get_job(id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(job_to_response(&job)))
}

/// POST /training/distributed/jobs/:id/start — start the distributed job.
pub async fn start_job(
    State(state): State<AppState>,
    Path(id): Path<DistributedJobId>,
) -> Result<Json<DistributedJobResponse>, (StatusCode, String)> {
    state
        .distributed_coordinator
        .start_job(id)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let job = state
        .distributed_coordinator
        .get_job(id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(job_to_response(&job)))
}

/// POST /training/distributed/jobs/:id/workers/:rank/complete — mark a worker as completed.
pub async fn worker_completed(
    State(state): State<AppState>,
    Path((id, rank)): Path<(DistributedJobId, u32)>,
) -> Result<Json<DistributedJobResponse>, (StatusCode, String)> {
    state
        .distributed_coordinator
        .worker_completed(id, rank)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let job = state
        .distributed_coordinator
        .get_job(id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Sync checkpoint via SY bridge
    if let Some(client) = &state.bridge_client {
        let checkpoint_path = format!(
            "{}/worker-{}",
            state.config.training.checkpoints_dir.display(),
            rank
        );
        let _ = client
            .sync_checkpoint(&id.to_string(), rank, &checkpoint_path)
            .await;
    }

    // If all workers completed, report completion to SY
    if job.status == TrainingStatus::Completed {
        if let Some(client) = &state.bridge_client {
            let _ = client
                .report_progress(&id.to_string(), "completed", 0, 0.0)
                .await;
        }
    }

    Ok(Json(job_to_response(&job)))
}

/// POST /training/distributed/jobs/:id/fail — fail the distributed job.
pub async fn fail_job(
    State(state): State<AppState>,
    Path(id): Path<DistributedJobId>,
) -> Result<Json<DistributedJobResponse>, (StatusCode, String)> {
    state
        .distributed_coordinator
        .fail_job(id)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let job = state
        .distributed_coordinator
        .get_job(id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(job_to_response(&job)))
}

/// POST /training/distributed/jobs/:id/aggregate — trigger checkpoint aggregation.
pub async fn aggregate(
    State(_state): State<AppState>,
    Path(id): Path<DistributedJobId>,
    Json(req): Json<AggregateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    use synapse_train::distributed::aggregator::{
        AggregationMethod, AggregationPlan, worker_checkpoint_dir,
    };

    let method = match req.method.as_str() {
        "weighted_average" => AggregationMethod::WeightedAverage,
        _ => AggregationMethod::Average,
    };

    let job = _state
        .distributed_coordinator
        .get_job(id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let base_dir = std::path::Path::new(&req.output_dir);
    let worker_dirs: Vec<_> = (0..job.config.world_size)
        .map(|rank| worker_checkpoint_dir(base_dir, rank))
        .collect();

    let plan = AggregationPlan::new(method, worker_dirs, base_dir.to_path_buf())
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let command = plan.build_command();

    Ok(Json(serde_json::json!({
        "job_id": id,
        "method": req.method,
        "command": command,
        "output_dir": req.output_dir,
    })))
}

fn job_to_response(job: &DistributedJobState) -> DistributedJobResponse {
    DistributedJobResponse {
        job_id: job.job_id,
        coordinator: job.coordinator.clone(),
        world_size: job.config.world_size,
        strategy: job.config.strategy,
        workers: job.workers.clone(),
        status: job.status,
        aggregate_loss: job.aggregate_loss,
        completed_workers: job.completed_workers,
    }
}
