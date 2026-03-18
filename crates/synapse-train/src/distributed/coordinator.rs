//! Distributed training coordinator — manages worker lifecycle and job state.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use synapse_types::SynapseError;
use synapse_types::distributed::*;
use synapse_types::error::Result;
use synapse_types::training::TrainingStatus;

/// Coordinates a distributed training job across multiple Synapse instances.
///
/// MVP: Tracks worker assignments and aggregates status.
/// Future: Handles checkpoint synchronization and gradient averaging via gRPC.
pub struct DistributedCoordinator {
    jobs: Arc<RwLock<HashMap<DistributedJobId, DistributedJobState>>>,
}

impl DistributedCoordinator {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new distributed training job.
    pub async fn create_job(
        &self,
        config: DistributedTrainingConfig,
        coordinator_instance: &str,
    ) -> Result<DistributedJobId> {
        let job_id = uuid::Uuid::new_v4();
        let state = DistributedJobState {
            job_id,
            config,
            coordinator: coordinator_instance.to_string(),
            workers: Vec::new(),
            status: TrainingStatus::Queued,
            aggregate_loss: None,
            completed_workers: 0,
        };
        self.jobs.write().await.insert(job_id, state);
        Ok(job_id)
    }

    /// Get the state of a distributed job.
    pub async fn get_job(&self, job_id: DistributedJobId) -> Result<DistributedJobState> {
        self.jobs.read().await.get(&job_id).cloned().ok_or_else(|| {
            SynapseError::DistributedError(format!("Distributed job {job_id} not found"))
        })
    }

    /// List all distributed jobs.
    pub async fn list_jobs(&self) -> Vec<DistributedJobState> {
        self.jobs.read().await.values().cloned().collect()
    }

    /// Assign a worker to a distributed job.
    pub async fn assign_worker(
        &self,
        job_id: DistributedJobId,
        worker: WorkerAssignment,
    ) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = jobs.get_mut(&job_id).ok_or_else(|| {
            SynapseError::DistributedError(format!("Distributed job {job_id} not found"))
        })?;

        if worker.rank as usize >= state.config.world_size as usize {
            return Err(SynapseError::DistributedError(format!(
                "Worker rank {} exceeds world_size {}",
                worker.rank, state.config.world_size
            )));
        }

        // Don't allow duplicate ranks
        if state.workers.iter().any(|w| w.rank == worker.rank) {
            return Err(SynapseError::DistributedError(format!(
                "Rank {} already assigned",
                worker.rank
            )));
        }

        state.workers.push(worker);
        Ok(())
    }

    /// Start the distributed job (all workers must be assigned).
    pub async fn start_job(&self, job_id: DistributedJobId) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = jobs.get_mut(&job_id).ok_or_else(|| {
            SynapseError::DistributedError(format!("Distributed job {job_id} not found"))
        })?;

        if state.workers.len() != state.config.world_size as usize {
            return Err(SynapseError::DistributedError(format!(
                "Expected {} workers, got {}",
                state.config.world_size,
                state.workers.len()
            )));
        }

        state.status = TrainingStatus::Running;
        Ok(())
    }

    /// Report that a worker has completed its portion.
    ///
    /// Guard: once all workers have reported, further calls are no-ops
    /// to prevent over-counting from duplicate completion reports.
    pub async fn worker_completed(&self, job_id: DistributedJobId, _rank: u32) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = jobs.get_mut(&job_id).ok_or_else(|| {
            SynapseError::DistributedError(format!("Distributed job {job_id} not found"))
        })?;

        // Guard against duplicate completions pushing count past world_size
        if state.completed_workers >= state.config.world_size {
            return Ok(());
        }

        state.completed_workers += 1;

        // All workers done → job complete
        if state.completed_workers >= state.config.world_size {
            state.status = TrainingStatus::Completed;
        }

        Ok(())
    }

    /// Update aggregate loss for a job.
    pub async fn update_aggregate_loss(&self, job_id: DistributedJobId, loss: f64) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = jobs.get_mut(&job_id).ok_or_else(|| {
            SynapseError::DistributedError(format!("Distributed job {job_id} not found"))
        })?;
        state.aggregate_loss = Some(loss);
        Ok(())
    }

    /// Collect checkpoint paths from all workers for synchronization.
    pub async fn collect_checkpoint_paths(
        &self,
        job_id: DistributedJobId,
        base_output_dir: &std::path::Path,
    ) -> Result<Vec<std::path::PathBuf>> {
        let jobs = self.jobs.read().await;
        let state = jobs.get(&job_id).ok_or_else(|| {
            SynapseError::DistributedError(format!("Distributed job {job_id} not found"))
        })?;

        let paths: Vec<std::path::PathBuf> = state
            .workers
            .iter()
            .map(|w| super::aggregator::worker_checkpoint_dir(base_output_dir, w.rank))
            .collect();

        Ok(paths)
    }

    /// Fail a distributed job.
    pub async fn fail_job(&self, job_id: DistributedJobId) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = jobs.get_mut(&job_id).ok_or_else(|| {
            SynapseError::DistributedError(format!("Distributed job {job_id} not found"))
        })?;
        state.status = TrainingStatus::Failed;
        Ok(())
    }
}

impl Default for DistributedCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_types::training::*;

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
        let job_id = coord.create_job(test_config(), "node-1").await.unwrap();
        let state = coord.get_job(job_id).await.unwrap();
        assert_eq!(state.coordinator, "node-1");
        assert_eq!(state.status, TrainingStatus::Queued);
    }

    #[tokio::test]
    async fn assign_workers_and_start() {
        let coord = DistributedCoordinator::new();
        let job_id = coord.create_job(test_config(), "node-1").await.unwrap();

        coord
            .assign_worker(
                job_id,
                WorkerAssignment {
                    rank: 0,
                    instance_id: "node-1".into(),
                    endpoint: "http://node-1:9000".into(),
                    device_ids: vec![0],
                },
            )
            .await
            .unwrap();

        // Can't start yet — only 1 of 2 workers assigned
        assert!(coord.start_job(job_id).await.is_err());

        coord
            .assign_worker(
                job_id,
                WorkerAssignment {
                    rank: 1,
                    instance_id: "node-2".into(),
                    endpoint: "http://node-2:9000".into(),
                    device_ids: vec![0],
                },
            )
            .await
            .unwrap();

        coord.start_job(job_id).await.unwrap();
        assert_eq!(
            coord.get_job(job_id).await.unwrap().status,
            TrainingStatus::Running
        );
    }

    #[tokio::test]
    async fn duplicate_rank_rejected() {
        let coord = DistributedCoordinator::new();
        let job_id = coord.create_job(test_config(), "node-1").await.unwrap();

        let worker = WorkerAssignment {
            rank: 0,
            instance_id: "node-1".into(),
            endpoint: "http://node-1:9000".into(),
            device_ids: vec![0],
        };
        coord.assign_worker(job_id, worker.clone()).await.unwrap();
        assert!(coord.assign_worker(job_id, worker).await.is_err());
    }

    #[tokio::test]
    async fn worker_completion_lifecycle() {
        let coord = DistributedCoordinator::new();
        let job_id = coord.create_job(test_config(), "node-1").await.unwrap();

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
                )
                .await
                .unwrap();
        }
        coord.start_job(job_id).await.unwrap();

        coord.worker_completed(job_id, 0).await.unwrap();
        assert_eq!(
            coord.get_job(job_id).await.unwrap().status,
            TrainingStatus::Running
        );

        coord.worker_completed(job_id, 1).await.unwrap();
        assert_eq!(
            coord.get_job(job_id).await.unwrap().status,
            TrainingStatus::Completed
        );
    }

    #[tokio::test]
    async fn fail_job() {
        let coord = DistributedCoordinator::new();
        let job_id = coord.create_job(test_config(), "node-1").await.unwrap();
        coord.fail_job(job_id).await.unwrap();
        assert_eq!(
            coord.get_job(job_id).await.unwrap().status,
            TrainingStatus::Failed
        );
    }

    #[tokio::test]
    async fn list_jobs() {
        let coord = DistributedCoordinator::new();
        coord.create_job(test_config(), "node-1").await.unwrap();
        coord.create_job(test_config(), "node-2").await.unwrap();
        assert_eq!(coord.list_jobs().await.len(), 2);
    }

    #[tokio::test]
    async fn update_aggregate_loss() {
        let coord = DistributedCoordinator::new();
        let job_id = coord.create_job(test_config(), "node-1").await.unwrap();
        coord.update_aggregate_loss(job_id, 0.42).await.unwrap();
        let job = coord.get_job(job_id).await.unwrap();
        assert_eq!(job.aggregate_loss, Some(0.42));
    }

    #[tokio::test]
    async fn collect_checkpoint_paths() {
        let coord = DistributedCoordinator::new();
        let job_id = coord.create_job(test_config(), "node-1").await.unwrap();

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
                )
                .await
                .unwrap();
        }

        let paths = coord
            .collect_checkpoint_paths(job_id, std::path::Path::new("/output"))
            .await
            .unwrap();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], std::path::PathBuf::from("/output/worker-0"));
        assert_eq!(paths[1], std::path::PathBuf::from("/output/worker-1"));
    }

    #[tokio::test]
    async fn rank_exceeds_world_size() {
        let coord = DistributedCoordinator::new();
        let job_id = coord.create_job(test_config(), "node-1").await.unwrap();
        let result = coord
            .assign_worker(
                job_id,
                WorkerAssignment {
                    rank: 5, // world_size is 2
                    instance_id: "node-x".into(),
                    endpoint: "http://node-x:9000".into(),
                    device_ids: vec![0],
                },
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_nonexistent_job() {
        let coord = DistributedCoordinator::new();
        assert!(coord.get_job(uuid::Uuid::new_v4()).await.is_err());
    }
}
