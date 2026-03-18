//! REST handlers for distributed training job management.

use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use synapse_types::TenantId;
use synapse_types::distributed::*;
use synapse_types::training::{TrainingJobConfig, TrainingStatus};

use crate::state::AppState;

/// Response for a distributed training job.
#[derive(Debug, Serialize)]
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
    Extension(_tenant_id): Extension<TenantId>, // TODO: tenant-scope distributed jobs
    Json(req): Json<CreateDistributedJobRequest>,
) -> Result<(StatusCode, Json<DistributedJobResponse>), (StatusCode, String)> {
    let config = DistributedTrainingConfig {
        base_config: req.base_config,
        world_size: req.world_size,
        strategy: req.strategy,
        placement_policy: None,
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
pub async fn list_jobs(
    State(state): State<AppState>,
    Extension(_tenant_id): Extension<TenantId>, // TODO: tenant-scope distributed jobs
) -> Json<Vec<DistributedJobResponse>> {
    let jobs = state.distributed_coordinator.list_jobs().await;
    Json(jobs.iter().map(job_to_response).collect())
}

/// GET /training/distributed/jobs/:id — get a distributed job.
pub async fn get_job(
    State(state): State<AppState>,
    Extension(_tenant_id): Extension<TenantId>, // TODO: tenant-scope distributed jobs
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
    Extension(_tenant_id): Extension<TenantId>, // TODO: tenant-scope distributed jobs
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
    Extension(_tenant_id): Extension<TenantId>, // TODO: tenant-scope distributed jobs
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
    Extension(_tenant_id): Extension<TenantId>, // TODO: tenant-scope distributed jobs
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
    Extension(_tenant_id): Extension<TenantId>, // TODO: tenant-scope distributed jobs
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
    Extension(_tenant_id): Extension<TenantId>, // TODO: tenant-scope distributed jobs
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use axum::extract::Extension;
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
                max_samples: None,
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
            output_name: None,
            lora: None,
            max_steps: None,
            time_budget_secs: None,
        }
    }

    #[test]
    fn create_distributed_job_request_deserialize() {
        let json = r#"{
            "base_model": "llama-7b",
            "dataset": {"path": "/data/train.jsonl", "format": "jsonl"},
            "method": "lora",
            "hyperparams": {"learning_rate": 0.0002, "epochs": 1, "batch_size": 4, "gradient_accumulation_steps": 1, "warmup_steps": 0, "weight_decay": 0.0, "max_seq_length": 512},
            "world_size": 4,
            "strategy": "data_parallel"
        }"#;
        let req: CreateDistributedJobRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.world_size, 4);
        assert_eq!(req.strategy, DistributedStrategy::DataParallel);
        assert_eq!(req.base_config.base_model, "llama-7b");
    }

    #[test]
    fn assign_worker_request_deserialize() {
        let json = r#"{
            "rank": 1,
            "instance_id": "node-2",
            "endpoint": "http://node-2:9000"
        }"#;
        let req: AssignWorkerRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.rank, 1);
        assert_eq!(req.instance_id, "node-2");
        assert!(req.device_ids.is_empty()); // default
    }

    #[test]
    fn assign_worker_request_with_devices() {
        let json = r#"{
            "rank": 0,
            "instance_id": "node-1",
            "endpoint": "http://node-1:9000",
            "device_ids": [0, 1]
        }"#;
        let req: AssignWorkerRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.device_ids, vec![0, 1]);
    }

    #[test]
    fn aggregate_request_deserialize_defaults() {
        let json = r#"{"output_dir": "/output/merged"}"#;
        let req: AggregateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.output_dir, "/output/merged");
        assert_eq!(req.method, "average");
    }

    #[test]
    fn aggregate_request_with_method() {
        let json = r#"{"output_dir": "/output/merged", "method": "weighted_average"}"#;
        let req: AggregateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "weighted_average");
    }

    #[test]
    fn distributed_job_response_serializes() {
        let resp = DistributedJobResponse {
            job_id: uuid::Uuid::new_v4(),
            coordinator: "node-1".into(),
            world_size: 4,
            strategy: DistributedStrategy::DataParallel,
            workers: vec![],
            status: TrainingStatus::Queued,
            aggregate_loss: None,
            completed_workers: 0,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["world_size"], 4);
        assert_eq!(json["strategy"], "data_parallel");
        assert_eq!(json["completed_workers"], 0);
        assert!(json["aggregate_loss"].is_null());
    }

    #[tokio::test]
    async fn create_distributed_job_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateDistributedJobRequest {
            base_config: test_job_config(),
            world_size: 2,
            strategy: DistributedStrategy::DataParallel,
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
        assert_eq!(result.1.world_size, 2);
        assert_eq!(result.1.workers.len(), 0);
    }

    #[tokio::test]
    async fn list_distributed_jobs_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let Json(jobs) = list_jobs(State(state), Extension(TenantId::default_tenant())).await;
        assert!(jobs.is_empty());
    }

    #[tokio::test]
    async fn list_distributed_jobs_with_data() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req1 = CreateDistributedJobRequest {
            base_config: test_job_config(),
            world_size: 2,
            strategy: DistributedStrategy::DataParallel,
        };
        let req2 = CreateDistributedJobRequest {
            base_config: test_job_config(),
            world_size: 4,
            strategy: DistributedStrategy::ModelParallel,
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

        let Json(jobs) = list_jobs(State(state), Extension(TenantId::default_tenant())).await;
        assert_eq!(jobs.len(), 2);
    }

    #[tokio::test]
    async fn get_distributed_job_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateDistributedJobRequest {
            base_config: test_job_config(),
            world_size: 2,
            strategy: DistributedStrategy::DataParallel,
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
            Path(created.job_id),
        )
        .await
        .unwrap();
        assert_eq!(result.job_id, created.job_id);
        assert_eq!(result.status, TrainingStatus::Queued);
    }

    #[tokio::test]
    async fn get_distributed_job_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let result = get_job(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(uuid::Uuid::new_v4()),
        )
        .await;
        assert_eq!(result.unwrap_err().0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn assign_worker_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateDistributedJobRequest {
            base_config: test_job_config(),
            world_size: 2,
            strategy: DistributedStrategy::DataParallel,
        };
        let (_, Json(created)) = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();

        let worker_req = AssignWorkerRequest {
            rank: 0,
            instance_id: "node-1".into(),
            endpoint: "http://node-1:9000".into(),
            device_ids: vec![0],
        };
        let result = assign_worker(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(created.job_id),
            Json(worker_req),
        )
        .await
        .unwrap();
        assert_eq!(result.workers.len(), 1);
        assert_eq!(result.workers[0].rank, 0);
    }

    #[tokio::test]
    async fn start_distributed_job_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateDistributedJobRequest {
            base_config: test_job_config(),
            world_size: 2,
            strategy: DistributedStrategy::DataParallel,
        };
        let (_, Json(created)) = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();

        // Assign both workers
        for rank in 0..2 {
            let worker_req = AssignWorkerRequest {
                rank,
                instance_id: format!("node-{}", rank + 1),
                endpoint: format!("http://node-{}:9000", rank + 1),
                device_ids: vec![0],
            };
            let _ = assign_worker(
                State(state.clone()),
                Extension(TenantId::default_tenant()),
                Path(created.job_id),
                Json(worker_req),
            )
            .await
            .unwrap();
        }

        let result = start_job(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(created.job_id),
        )
        .await
        .unwrap();
        assert_eq!(result.status, TrainingStatus::Running);
    }

    #[tokio::test]
    async fn start_job_insufficient_workers() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateDistributedJobRequest {
            base_config: test_job_config(),
            world_size: 2,
            strategy: DistributedStrategy::DataParallel,
        };
        let (_, Json(created)) = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();

        // Only assign 1 of 2 workers
        let worker_req = AssignWorkerRequest {
            rank: 0,
            instance_id: "node-1".into(),
            endpoint: "http://node-1:9000".into(),
            device_ids: vec![0],
        };
        let _ = assign_worker(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Path(created.job_id),
            Json(worker_req),
        )
        .await
        .unwrap();

        let result = start_job(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(created.job_id),
        )
        .await;
        assert_eq!(result.unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn fail_distributed_job_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateDistributedJobRequest {
            base_config: test_job_config(),
            world_size: 2,
            strategy: DistributedStrategy::DataParallel,
        };
        let (_, Json(created)) = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();

        let result = fail_job(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(created.job_id),
        )
        .await
        .unwrap();
        assert_eq!(result.status, TrainingStatus::Failed);
    }

    #[tokio::test]
    async fn worker_completed_lifecycle() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let req = CreateDistributedJobRequest {
            base_config: test_job_config(),
            world_size: 2,
            strategy: DistributedStrategy::DataParallel,
        };
        let (_, Json(created)) = create_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await
        .unwrap();

        for rank in 0..2 {
            let worker_req = AssignWorkerRequest {
                rank,
                instance_id: format!("node-{}", rank + 1),
                endpoint: format!("http://node-{}:9000", rank + 1),
                device_ids: vec![0],
            };
            let _ = assign_worker(
                State(state.clone()),
                Extension(TenantId::default_tenant()),
                Path(created.job_id),
                Json(worker_req),
            )
            .await
            .unwrap();
        }
        let _ = start_job(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Path(created.job_id),
        )
        .await
        .unwrap();

        // First worker completes
        let result = worker_completed(
            State(state.clone()),
            Extension(TenantId::default_tenant()),
            Path((created.job_id, 0)),
        )
        .await
        .unwrap();
        assert_eq!(result.completed_workers, 1);
        assert_eq!(result.status, TrainingStatus::Running);

        // Second worker completes — job should be done
        let result = worker_completed(
            State(state),
            Extension(TenantId::default_tenant()),
            Path((created.job_id, 1)),
        )
        .await
        .unwrap();
        assert_eq!(result.completed_workers, 2);
        assert_eq!(result.status, TrainingStatus::Completed);
    }

    #[tokio::test]
    async fn job_to_response_conversion() {
        let job = DistributedJobState {
            job_id: uuid::Uuid::new_v4(),
            config: DistributedTrainingConfig {
                base_config: test_job_config(),
                world_size: 4,
                strategy: DistributedStrategy::PipelineParallel,
                placement_policy: None,
            },
            coordinator: "node-1".into(),
            workers: vec![WorkerAssignment {
                rank: 0,
                instance_id: "node-1".into(),
                endpoint: "http://node-1:9000".into(),
                device_ids: vec![0, 1],
            }],
            status: TrainingStatus::Running,
            aggregate_loss: Some(0.42),
            completed_workers: 1,
        };

        let resp = job_to_response(&job);
        assert_eq!(resp.job_id, job.job_id);
        assert_eq!(resp.coordinator, "node-1");
        assert_eq!(resp.world_size, 4);
        assert_eq!(resp.strategy, DistributedStrategy::PipelineParallel);
        assert_eq!(resp.workers.len(), 1);
        assert_eq!(resp.status, TrainingStatus::Running);
        assert_eq!(resp.aggregate_loss, Some(0.42));
        assert_eq!(resp.completed_workers, 1);
    }
}
