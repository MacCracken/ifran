//! Distributed training coordinator — manages worker lifecycle and job state.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

use ifran_types::IfranError;
use ifran_types::distributed::*;
use ifran_types::error::Result;
use ifran_types::training::TrainingStatus;
use majra::barrier::{AsyncBarrierSet, BarrierResult};

/// Coordinates a distributed training job across multiple Ifran instances.
///
/// MVP: Tracks worker assignments and aggregates status.
/// Future: Handles checkpoint synchronization and gradient averaging via gRPC.
pub struct DistributedCoordinator {
    jobs: Arc<RwLock<HashMap<DistributedJobId, DistributedJobState>>>,
    barriers: AsyncBarrierSet,
}

impl DistributedCoordinator {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            barriers: AsyncBarrierSet::new(),
        }
    }

    /// Create a new distributed training job scoped to a tenant.
    pub async fn create_job(
        &self,
        config: DistributedTrainingConfig,
        coordinator_instance: &str,
        tenant_id: &str,
    ) -> Result<DistributedJobId> {
        let job_id = uuid::Uuid::new_v4();
        let state = DistributedJobState {
            job_id,
            config,
            coordinator: coordinator_instance.to_string(),
            tenant_id: tenant_id.to_string(),
            workers: Vec::new(),
            status: TrainingStatus::Queued,
            aggregate_loss: None,
            completed_workers: 0,
        };
        self.jobs.write().await.insert(job_id, state);
        Ok(job_id)
    }

    /// Get the state of a distributed job, verifying tenant ownership.
    pub async fn get_job(
        &self,
        job_id: DistributedJobId,
        tenant_id: &str,
    ) -> Result<DistributedJobState> {
        let jobs = self.jobs.read().await;
        Ok(lookup_ref(&jobs, job_id, tenant_id)?.clone())
    }

    /// List distributed jobs for a specific tenant.
    pub async fn list_jobs(&self, tenant_id: &str) -> Vec<DistributedJobState> {
        self.jobs
            .read()
            .await
            .values()
            .filter(|j| j.tenant_id == tenant_id)
            .cloned()
            .collect()
    }

    /// Assign a worker to a distributed job.
    pub async fn assign_worker(
        &self,
        job_id: DistributedJobId,
        worker: WorkerAssignment,
        tenant_id: &str,
    ) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = lookup_mut(&mut jobs, job_id, tenant_id)?;

        if worker.rank as usize >= state.config.world_size as usize {
            return Err(IfranError::DistributedError(format!(
                "Worker rank {} exceeds world_size {}",
                worker.rank, state.config.world_size
            )));
        }

        // Don't allow duplicate ranks
        if state.workers.iter().any(|w| w.rank == worker.rank) {
            return Err(IfranError::DistributedError(format!(
                "Rank {} already assigned",
                worker.rank
            )));
        }

        state.workers.push(worker);
        Ok(())
    }

    /// Start the distributed job (all workers must be assigned).
    pub async fn start_job(&self, job_id: DistributedJobId, tenant_id: &str) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = lookup_mut(&mut jobs, job_id, tenant_id)?;

        if state.workers.len() != state.config.world_size as usize {
            return Err(IfranError::DistributedError(format!(
                "Expected {} workers, got {}",
                state.config.world_size,
                state.workers.len()
            )));
        }

        let participants: HashSet<String> = state
            .workers
            .iter()
            .map(|w| format!("worker-{}", w.rank))
            .collect();
        self.barriers.create(job_id.to_string(), participants);

        state.status = TrainingStatus::Running;
        Ok(())
    }

    /// Report that a worker has completed its portion.
    ///
    /// Guard: once all workers have reported, further calls are no-ops
    /// to prevent over-counting from duplicate completion reports.
    pub async fn worker_completed(
        &self,
        job_id: DistributedJobId,
        rank: u32,
        tenant_id: &str,
    ) -> Result<()> {
        let barrier_key = job_id.to_string();
        let participant = format!("worker-{rank}");

        let released = match self.barriers.arrive(&barrier_key, &participant) {
            BarrierResult::Released => true,
            BarrierResult::Waiting { .. } => false,
            BarrierResult::Unknown => {
                let jobs = self.jobs.read().await;
                let state = lookup_ref(&jobs, job_id, tenant_id)?;
                if state.completed_workers >= state.config.world_size {
                    return Ok(());
                }
                return Err(IfranError::DistributedError(format!(
                    "No barrier found for job {job_id}"
                )));
            }
            _ => false,
        };

        let mut jobs = self.jobs.write().await;
        let state = lookup_mut(&mut jobs, job_id, tenant_id)?;

        if state.completed_workers < state.config.world_size {
            state.completed_workers += 1;
        }

        if released {
            state.status = TrainingStatus::Completed;
            drop(jobs);
            self.barriers.complete(&barrier_key);
        }

        Ok(())
    }

    /// Update aggregate loss for a job.
    pub async fn update_aggregate_loss(
        &self,
        job_id: DistributedJobId,
        loss: f64,
        tenant_id: &str,
    ) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = lookup_mut(&mut jobs, job_id, tenant_id)?;
        state.aggregate_loss = Some(loss);
        Ok(())
    }

    /// Collect checkpoint paths from all workers for synchronization.
    pub async fn collect_checkpoint_paths(
        &self,
        job_id: DistributedJobId,
        base_output_dir: &std::path::Path,
        tenant_id: &str,
    ) -> Result<Vec<std::path::PathBuf>> {
        let jobs = self.jobs.read().await;
        let state = lookup_ref(&jobs, job_id, tenant_id)?;

        let paths: Vec<std::path::PathBuf> = state
            .workers
            .iter()
            .map(|w| super::aggregator::worker_checkpoint_dir(base_output_dir, w.rank))
            .collect();

        Ok(paths)
    }

    /// Automatically place workers using fleet node resources and a placement policy.
    ///
    /// This enables distributed training without SecureYeoman — workers are
    /// assigned directly to fleet nodes based on available GPU resources.
    pub async fn auto_place(
        &self,
        job_id: DistributedJobId,
        nodes: &[super::placement::NodeResources],
        policy: &dyn super::placement::PlacementPolicy,
        tenant_id: &str,
    ) -> Result<Vec<WorkerAssignment>> {
        let jobs = self.jobs.read().await;
        let state = lookup_ref(&jobs, job_id, tenant_id)?;

        if !state.workers.is_empty() {
            return Err(IfranError::DistributedError(
                "Workers already assigned — cannot auto-place".into(),
            ));
        }

        let world_size = state.config.world_size;
        // Default to 1 GPU per worker if not specified
        let gpus_per_worker = 1u32;

        drop(jobs); // Release read lock before acquiring write lock

        let assignments = policy.place(world_size, gpus_per_worker, nodes)?;

        // Assign each worker
        for worker in &assignments {
            self.assign_worker(job_id, worker.clone(), tenant_id)
                .await?;
        }

        Ok(assignments)
    }

    /// Fail a distributed job.
    pub async fn fail_job(&self, job_id: DistributedJobId, tenant_id: &str) -> Result<()> {
        let barrier_key = job_id.to_string();

        let mut jobs = self.jobs.write().await;
        let state = lookup_mut(&mut jobs, job_id, tenant_id)?;
        state.status = TrainingStatus::Failed;

        let worker_ids: Vec<String> = state
            .workers
            .iter()
            .map(|w| format!("worker-{}", w.rank))
            .collect();
        drop(jobs);

        for wid in &worker_ids {
            let _ = self.barriers.force(&barrier_key, wid);
        }
        self.barriers.complete(&barrier_key);

        Ok(())
    }
}

/// Look up a job by ID and verify tenant ownership (shared reference).
fn lookup_ref<'a>(
    jobs: &'a HashMap<DistributedJobId, DistributedJobState>,
    job_id: DistributedJobId,
    tenant_id: &str,
) -> Result<&'a DistributedJobState> {
    let state = jobs.get(&job_id).ok_or_else(|| {
        IfranError::DistributedError(format!("Distributed job {job_id} not found"))
    })?;
    if state.tenant_id != tenant_id {
        return Err(IfranError::DistributedError(format!(
            "Distributed job {job_id} not found"
        )));
    }
    Ok(state)
}

/// Look up a job by ID and verify tenant ownership (mutable reference).
fn lookup_mut<'a>(
    jobs: &'a mut HashMap<DistributedJobId, DistributedJobState>,
    job_id: DistributedJobId,
    tenant_id: &str,
) -> Result<&'a mut DistributedJobState> {
    let state = jobs.get_mut(&job_id).ok_or_else(|| {
        IfranError::DistributedError(format!("Distributed job {job_id} not found"))
    })?;
    if state.tenant_id != tenant_id {
        return Err(IfranError::DistributedError(format!(
            "Distributed job {job_id} not found"
        )));
    }
    Ok(state)
}

impl Default for DistributedCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ifran_types::training::*;

    const TENANT: &str = "default";
    const TENANT_B: &str = "acme";

    fn test_config() -> DistributedTrainingConfig {
        DistributedTrainingConfig {
            base_config: TrainingJobConfig {
                base_model: "test-model".into(),
                method: TrainingMethod::Lora,
                dataset: DatasetConfig {
                    path: "/tmp/data.jsonl".into(),
                    format: DatasetFormat::Jsonl,
                    split: None,
                    max_samples: None,
                },
                output_name: None,
                lora: None,
                max_steps: None,
                time_budget_secs: None,
                hyperparams: HyperParams {
                    learning_rate: 2e-4,
                    epochs: 1,
                    batch_size: 4,
                    gradient_accumulation_steps: 1,
                    warmup_steps: 0,
                    weight_decay: 0.0,
                    max_seq_length: 512,
                },
            },
            world_size: 2,
            strategy: DistributedStrategy::DataParallel,
            placement_policy: None,
        }
    }

    #[tokio::test]
    async fn create_and_get_job() {
        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();
        let state = coord.get_job(job_id, TENANT).await.unwrap();
        assert_eq!(state.coordinator, "node-1");
        assert_eq!(state.status, TrainingStatus::Queued);
        assert_eq!(state.tenant_id, TENANT);
    }

    #[tokio::test]
    async fn tenant_isolation() {
        let coord = DistributedCoordinator::new();
        let job_a = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();
        let job_b = coord
            .create_job(test_config(), "node-1", TENANT_B)
            .await
            .unwrap();

        // Each tenant sees only their own jobs
        assert_eq!(coord.list_jobs(TENANT).await.len(), 1);
        assert_eq!(coord.list_jobs(TENANT_B).await.len(), 1);

        // Cross-tenant access is denied
        assert!(coord.get_job(job_a, TENANT_B).await.is_err());
        assert!(coord.get_job(job_b, TENANT).await.is_err());
    }

    #[tokio::test]
    async fn assign_workers_and_start() {
        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();

        coord
            .assign_worker(
                job_id,
                WorkerAssignment {
                    rank: 0,
                    instance_id: "node-1".into(),
                    endpoint: "http://node-1:9000".into(),
                    device_ids: vec![0],
                },
                TENANT,
            )
            .await
            .unwrap();

        // Can't start yet — only 1 of 2 workers assigned
        assert!(coord.start_job(job_id, TENANT).await.is_err());

        coord
            .assign_worker(
                job_id,
                WorkerAssignment {
                    rank: 1,
                    instance_id: "node-2".into(),
                    endpoint: "http://node-2:9000".into(),
                    device_ids: vec![0],
                },
                TENANT,
            )
            .await
            .unwrap();

        coord.start_job(job_id, TENANT).await.unwrap();
        assert_eq!(
            coord.get_job(job_id, TENANT).await.unwrap().status,
            TrainingStatus::Running
        );
    }

    #[tokio::test]
    async fn duplicate_rank_rejected() {
        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();

        let worker = WorkerAssignment {
            rank: 0,
            instance_id: "node-1".into(),
            endpoint: "http://node-1:9000".into(),
            device_ids: vec![0],
        };
        coord
            .assign_worker(job_id, worker.clone(), TENANT)
            .await
            .unwrap();
        assert!(coord.assign_worker(job_id, worker, TENANT).await.is_err());
    }

    #[tokio::test]
    async fn worker_completion_lifecycle() {
        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();

        for rank in 0..2 {
            coord
                .assign_worker(
                    job_id,
                    WorkerAssignment {
                        rank,
                        instance_id: format!("node-{}", rank + 1),
                        endpoint: format!("http://node-{}:9000", rank + 1),
                        device_ids: vec![0],
                    },
                    TENANT,
                )
                .await
                .unwrap();
        }
        coord.start_job(job_id, TENANT).await.unwrap();

        coord.worker_completed(job_id, 0, TENANT).await.unwrap();
        assert_eq!(
            coord.get_job(job_id, TENANT).await.unwrap().status,
            TrainingStatus::Running
        );

        coord.worker_completed(job_id, 1, TENANT).await.unwrap();
        assert_eq!(
            coord.get_job(job_id, TENANT).await.unwrap().status,
            TrainingStatus::Completed
        );
    }

    #[tokio::test]
    async fn fail_job() {
        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();
        coord.fail_job(job_id, TENANT).await.unwrap();
        assert_eq!(
            coord.get_job(job_id, TENANT).await.unwrap().status,
            TrainingStatus::Failed
        );
    }

    #[tokio::test]
    async fn list_jobs() {
        let coord = DistributedCoordinator::new();
        coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();
        coord
            .create_job(test_config(), "node-2", TENANT)
            .await
            .unwrap();
        assert_eq!(coord.list_jobs(TENANT).await.len(), 2);
    }

    #[tokio::test]
    async fn update_aggregate_loss() {
        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();
        coord
            .update_aggregate_loss(job_id, 0.42, TENANT)
            .await
            .unwrap();
        let job = coord.get_job(job_id, TENANT).await.unwrap();
        assert_eq!(job.aggregate_loss, Some(0.42));
    }

    #[tokio::test]
    async fn collect_checkpoint_paths() {
        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();

        for rank in 0..2 {
            coord
                .assign_worker(
                    job_id,
                    WorkerAssignment {
                        rank,
                        instance_id: format!("node-{}", rank + 1),
                        endpoint: format!("http://node-{}:9000", rank + 1),
                        device_ids: vec![0],
                    },
                    TENANT,
                )
                .await
                .unwrap();
        }

        let paths = coord
            .collect_checkpoint_paths(job_id, std::path::Path::new("/output"), TENANT)
            .await
            .unwrap();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], std::path::PathBuf::from("/output/worker-0"));
        assert_eq!(paths[1], std::path::PathBuf::from("/output/worker-1"));
    }

    #[tokio::test]
    async fn rank_exceeds_world_size() {
        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();
        let result = coord
            .assign_worker(
                job_id,
                WorkerAssignment {
                    rank: 5, // world_size is 2
                    instance_id: "node-x".into(),
                    endpoint: "http://node-x:9000".into(),
                    device_ids: vec![0],
                },
                TENANT,
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_nonexistent_job() {
        let coord = DistributedCoordinator::new();
        assert!(coord.get_job(uuid::Uuid::new_v4(), TENANT).await.is_err());
    }

    #[tokio::test]
    async fn auto_place_with_gpu_affinity() {
        use crate::distributed::placement::*;

        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();

        let nodes = vec![NodeResources {
            node_id: "node-1".into(),
            endpoint: "http://node-1:8420".into(),
            available_gpu_ids: vec![0, 1, 2, 3],
            available_gpu_memory_mb: 96000,
            gpu_utilization_pct: Some(10.0),
            cost_per_gpu_hour: None,
        }];

        let policy = GpuAffinityPolicy;
        let assignments = coord
            .auto_place(job_id, &nodes, &policy, TENANT)
            .await
            .unwrap();

        assert_eq!(assignments.len(), 2); // world_size = 2
        assert_eq!(assignments[0].rank, 0);
        assert_eq!(assignments[1].rank, 1);

        // Job should now have 2 workers assigned
        let job = coord.get_job(job_id, TENANT).await.unwrap();
        assert_eq!(job.workers.len(), 2);
    }

    #[tokio::test]
    async fn auto_place_rejects_if_workers_already_assigned() {
        use crate::distributed::placement::*;

        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();

        // Manually assign a worker first
        coord
            .assign_worker(
                job_id,
                WorkerAssignment {
                    rank: 0,
                    instance_id: "node-1".into(),
                    endpoint: "http://node-1:8420".into(),
                    device_ids: vec![0],
                },
                TENANT,
            )
            .await
            .unwrap();

        let nodes = vec![NodeResources {
            node_id: "node-1".into(),
            endpoint: "http://node-1:8420".into(),
            available_gpu_ids: vec![0, 1, 2, 3],
            available_gpu_memory_mb: 96000,
            gpu_utilization_pct: None,
            cost_per_gpu_hour: None,
        }];

        let result = coord
            .auto_place(job_id, &nodes, &GpuAffinityPolicy, TENANT)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn cross_tenant_operations_denied() {
        let coord = DistributedCoordinator::new();
        let job_id = coord
            .create_job(test_config(), "node-1", TENANT)
            .await
            .unwrap();

        // Assign worker with wrong tenant
        let worker = WorkerAssignment {
            rank: 0,
            instance_id: "node-1".into(),
            endpoint: "http://node-1:9000".into(),
            device_ids: vec![0],
        };
        assert!(coord.assign_worker(job_id, worker, TENANT_B).await.is_err());

        // Start with wrong tenant
        assert!(coord.start_job(job_id, TENANT_B).await.is_err());

        // Fail with wrong tenant
        assert!(coord.fail_job(job_id, TENANT_B).await.is_err());

        // Worker completed with wrong tenant
        assert!(coord.worker_completed(job_id, 0, TENANT_B).await.is_err());
    }
}
