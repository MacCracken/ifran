//! Training job manager — creates, tracks, and manages training job lifecycles.

use crate::executor::{ExecutorKind, TrainingExecutor};
use crate::job::status::JobState;
use std::collections::HashMap;
use std::sync::Arc;
use synapse_types::SynapseError;
use synapse_types::error::Result;
use synapse_types::training::{TrainingJobConfig, TrainingJobId, TrainingStatus};
use tokio::sync::RwLock;
use tracing::info;

/// Manages all training jobs.
pub struct JobManager {
    jobs: Arc<RwLock<HashMap<TrainingJobId, JobState>>>,
    executor: Arc<dyn TrainingExecutor>,
    max_concurrent: usize,
}

impl JobManager {
    pub fn new(
        executor_kind: ExecutorKind,
        trainer_image: Option<String>,
        max_concurrent: usize,
    ) -> Self {
        let executor: Arc<dyn TrainingExecutor> = match executor_kind {
            ExecutorKind::Docker => Arc::new(crate::executor::docker::DockerExecutor::new(
                trainer_image.unwrap_or_else(|| "ghcr.io/maccracken/synapse-trainer:latest".into()),
            )),
            ExecutorKind::Subprocess => {
                Arc::new(crate::executor::subprocess::SubprocessExecutor::new())
            }
        };

        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            executor,
            max_concurrent,
        }
    }

    /// Create and enqueue a new training job.
    pub async fn create_job(&self, config: TrainingJobConfig) -> Result<TrainingJobId> {
        let id = uuid::Uuid::new_v4();
        let total_steps = estimate_total_steps(&config);
        let state = JobState::new(id, config, total_steps);

        info!(job_id = %id, "Created training job");
        self.jobs.write().await.insert(id, state);
        Ok(id)
    }

    /// Start a queued job.
    pub async fn start_job(&self, id: TrainingJobId) -> Result<()> {
        let running_count = self.running_count().await;
        if running_count >= self.max_concurrent {
            return Err(SynapseError::TrainingError(format!(
                "Max concurrent jobs ({}) reached",
                self.max_concurrent
            )));
        }

        let mut jobs = self.jobs.write().await;
        let state = jobs
            .get_mut(&id)
            .ok_or_else(|| SynapseError::TrainingError(format!("Job {id} not found")))?;

        if state.status != TrainingStatus::Queued {
            return Err(SynapseError::TrainingError(format!(
                "Job {id} is {:?}, not Queued",
                state.status
            )));
        }

        state.start();

        let config = state.config.clone();
        let jobs_ref = self.jobs.clone();
        let executor = self.executor.clone();

        // Spawn the training in background
        tokio::spawn(async move {
            let result = executor.run(&config, id).await;
            let mut jobs = jobs_ref.write().await;
            if let Some(state) = jobs.get_mut(&id) {
                match result {
                    Ok(()) => state.complete(),
                    Err(e) => state.fail(e.to_string()),
                }
            }
        });

        Ok(())
    }

    /// Cancel a running or queued job.
    pub async fn cancel_job(&self, id: TrainingJobId) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = jobs
            .get_mut(&id)
            .ok_or_else(|| SynapseError::TrainingError(format!("Job {id} not found")))?;

        if state.is_terminal() {
            return Err(SynapseError::TrainingError(format!(
                "Job {id} already in terminal state {:?}",
                state.status
            )));
        }

        self.executor.cancel(id).await?;
        state.cancel();
        info!(job_id = %id, "Cancelled training job");
        Ok(())
    }

    /// Get a job's current state.
    pub async fn get_job(&self, id: TrainingJobId) -> Result<JobState> {
        self.jobs
            .read()
            .await
            .get(&id)
            .cloned()
            .ok_or_else(|| SynapseError::TrainingError(format!("Job {id} not found")))
    }

    /// List all jobs, optionally filtered by status.
    pub async fn list_jobs(&self, status_filter: Option<TrainingStatus>) -> Vec<JobState> {
        let jobs = self.jobs.read().await;
        jobs.values()
            .filter(|j| status_filter.is_none() || Some(j.status) == status_filter)
            .cloned()
            .collect()
    }

    /// Count of currently running jobs.
    pub async fn running_count(&self) -> usize {
        self.jobs
            .read()
            .await
            .values()
            .filter(|j| j.status == TrainingStatus::Running)
            .count()
    }

    /// Update progress for a job (called by executor callbacks).
    pub async fn update_progress(&self, id: TrainingJobId, step: u64, epoch: f32, loss: f64) {
        let mut jobs = self.jobs.write().await;
        if let Some(state) = jobs.get_mut(&id) {
            state.update_progress(step, epoch, loss);
        }
    }

    /// Maximum concurrent job limit.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

fn estimate_total_steps(config: &TrainingJobConfig) -> u64 {
    // Rough estimate: epochs * (assumed dataset size / batch_size)
    let batch = config.hyperparams.batch_size.max(1) as u64;
    let epochs = config.hyperparams.epochs.max(1) as u64;
    let assumed_samples = config.dataset.max_samples.unwrap_or(10000) as u64;
    (assumed_samples / batch) * epochs
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_types::training::*;

    fn test_config() -> TrainingJobConfig {
        TrainingJobConfig {
            base_model: "test-model".into(),
            dataset: DatasetConfig {
                path: "/tmp/data.jsonl".into(),
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
            output_name: None,
            lora: None,
        }
    }

    #[tokio::test]
    async fn create_and_get_job() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let id = manager.create_job(test_config()).await.unwrap();
        let job = manager.get_job(id).await.unwrap();
        assert_eq!(job.status, TrainingStatus::Queued);
        assert_eq!(job.id, id);
    }

    #[tokio::test]
    async fn list_jobs_all() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        manager.create_job(test_config()).await.unwrap();
        manager.create_job(test_config()).await.unwrap();
        let all = manager.list_jobs(None).await;
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn list_jobs_filtered() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        manager.create_job(test_config()).await.unwrap();
        let queued = manager.list_jobs(Some(TrainingStatus::Queued)).await;
        assert_eq!(queued.len(), 1);
        let running = manager.list_jobs(Some(TrainingStatus::Running)).await;
        assert!(running.is_empty());
    }

    #[tokio::test]
    async fn get_job_not_found() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let result = manager.get_job(uuid::Uuid::new_v4()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn running_count_starts_at_zero() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        assert_eq!(manager.running_count().await, 0);
    }

    #[tokio::test]
    async fn cancel_not_found() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let result = manager.cancel_job(uuid::Uuid::new_v4()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn update_progress() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let id = manager.create_job(test_config()).await.unwrap();
        manager.update_progress(id, 50, 0.5, 0.42).await;
        let job = manager.get_job(id).await.unwrap();
        assert_eq!(job.current_step, 50);
        assert_eq!(job.current_loss, Some(0.42));
    }

    #[test]
    fn estimate_steps_calculation() {
        let config = test_config();
        let steps = estimate_total_steps(&config);
        // 100 samples / 4 batch * 1 epoch = 25
        assert_eq!(steps, 25);
    }

    #[test]
    fn estimate_steps_no_max_samples() {
        let mut config = test_config();
        config.dataset.max_samples = None;
        let steps = estimate_total_steps(&config);
        // 10000 default / 4 batch * 1 epoch = 2500
        assert_eq!(steps, 2500);
    }

    #[tokio::test]
    async fn max_concurrent_getter() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 3);
        assert_eq!(manager.max_concurrent(), 3);
    }
}
