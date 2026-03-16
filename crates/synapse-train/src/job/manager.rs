//! Training job manager — creates, tracks, and manages training job lifecycles.

use crate::executor::{ExecutorKind, TrainingExecutor};
use crate::job::status::JobState;
use crate::job::store::JobStore;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use synapse_types::SynapseError;
use synapse_types::TenantId;
use synapse_types::error::Result;
use synapse_types::training::{TrainingJobConfig, TrainingJobId, TrainingStatus};
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Manages all training jobs.
pub struct JobManager {
    jobs: Arc<RwLock<HashMap<TrainingJobId, JobState>>>,
    executor: Arc<dyn TrainingExecutor>,
    max_concurrent: usize,
    store: Option<Arc<Mutex<JobStore>>>,
}

impl JobManager {
    pub fn new(
        executor_kind: ExecutorKind,
        trainer_image: Option<String>,
        max_concurrent: usize,
    ) -> Self {
        let executor = Self::make_executor(executor_kind, trainer_image);

        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            executor,
            max_concurrent,
            store: None,
        }
    }

    /// Create a JobManager backed by a persistent SQLite store.
    pub fn new_with_store(
        executor_kind: ExecutorKind,
        trainer_image: Option<String>,
        max_concurrent: usize,
        store: JobStore,
    ) -> Self {
        let executor = Self::make_executor(executor_kind, trainer_image);

        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            executor,
            max_concurrent,
            store: Some(Arc::new(Mutex::new(store))),
        }
    }

    fn make_executor(
        executor_kind: ExecutorKind,
        trainer_image: Option<String>,
    ) -> Arc<dyn TrainingExecutor> {
        match executor_kind {
            ExecutorKind::Docker => Arc::new(crate::executor::docker::DockerExecutor::new(
                trainer_image.unwrap_or_else(|| "ghcr.io/maccracken/synapse-trainer:latest".into()),
            )),
            ExecutorKind::Subprocess => {
                Arc::new(crate::executor::subprocess::SubprocessExecutor::new())
            }
        }
    }

    /// Persist a job to the store, if one is configured. Logs warnings on failure.
    fn persist(&self, job: &JobState) {
        if let Some(store) = &self.store {
            if let Ok(store) = store.lock() {
                if let Err(e) = store.save_job(job) {
                    warn!(job_id = %job.id, error = %e, "Failed to persist job state");
                }
            }
        }
    }

    /// Recover non-terminal jobs from the store after a process restart.
    /// Jobs that were Running or Preparing are marked as Failed since the
    /// process crashed while they were in-flight.
    pub async fn recover(&self) -> Result<usize> {
        let store = match &self.store {
            Some(s) => s,
            None => return Ok(0),
        };

        let mut recovered = {
            let store = store.lock().map_err(|e| {
                SynapseError::StorageError(format!("Failed to lock job store: {e}"))
            })?;
            store.recover_jobs()?
        };

        let mut count = 0;
        let mut jobs = self.jobs.write().await;

        for job in &mut recovered {
            if job.status == TrainingStatus::Running || job.status == TrainingStatus::Preparing {
                job.fail("Process crashed \u{2014} job interrupted".into());
                self.persist(job);
            }
            info!(job_id = %job.id, status = ?job.status, "Recovered job from store");
            jobs.insert(job.id, job.clone());
            count += 1;
        }

        if count > 0 {
            info!(count, "Recovered jobs from persistent store");
        }

        Ok(count)
    }

    /// Create and enqueue a new training job.
    pub async fn create_job(
        &self,
        config: TrainingJobConfig,
        tenant_id: TenantId,
    ) -> Result<TrainingJobId> {
        config.hyperparams.validate()?;
        let id = uuid::Uuid::new_v4();
        let total_steps = estimate_total_steps(&config);
        let state = JobState::new(id, tenant_id, config, total_steps);

        self.persist(&state);
        info!(job_id = %id, "Created training job");
        self.jobs.write().await.insert(id, state);
        Ok(id)
    }

    /// Start a queued job.
    pub async fn start_job(&self, id: TrainingJobId) -> Result<()> {
        let mut jobs = self.jobs.write().await;

        // Check running count inside the write lock to prevent race condition
        // where multiple concurrent start_job calls all pass the check
        let running_count = jobs
            .values()
            .filter(|j| j.status == TrainingStatus::Running)
            .count();
        if running_count >= self.max_concurrent {
            return Err(SynapseError::TrainingError(format!(
                "Max concurrent jobs ({}) reached",
                self.max_concurrent
            )));
        }

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
        self.persist(state);

        let config = state.config.clone();
        let jobs_ref = self.jobs.clone();
        let executor = self.executor.clone();
        let store = self.store.clone();

        // Spawn the training in background
        tokio::spawn(async move {
            let result = executor.run(&config, id).await;
            let mut jobs = jobs_ref.write().await;
            if let Some(state) = jobs.get_mut(&id) {
                match result {
                    Ok(()) => state.complete(),
                    Err(e) => state.fail(e.to_string()),
                }
                // Persist terminal state
                if let Some(store) = &store {
                    if let Ok(store) = store.lock() {
                        if let Err(e) = store.save_job(state) {
                            warn!(job_id = %id, error = %e, "Failed to persist job completion");
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Cancel a running or queued job.
    pub async fn cancel_job(&self, id: TrainingJobId, tenant_id: &TenantId) -> Result<()> {
        // Validate the job exists, belongs to tenant, and is cancellable,
        // then release the lock before calling executor.cancel() to avoid
        // holding the write lock during a potentially long async operation
        // (which could deadlock if the executor also needs lock access).
        {
            let jobs = self.jobs.read().await;
            let state = jobs
                .get(&id)
                .ok_or_else(|| SynapseError::TrainingError(format!("Job {id} not found")))?;

            if &state.tenant_id != tenant_id {
                return Err(SynapseError::TrainingError(format!("Job {id} not found")));
            }

            if state.is_terminal() {
                return Err(SynapseError::TrainingError(format!(
                    "Job {id} already in terminal state {:?}",
                    state.status
                )));
            }
        }

        self.executor.cancel(id).await?;

        let mut jobs = self.jobs.write().await;
        if let Some(state) = jobs.get_mut(&id) {
            // Re-check: job may have completed between lock release and reacquire
            if !state.is_terminal() {
                state.cancel();
                self.persist(state);
            }
        }
        info!(job_id = %id, "Cancelled training job");
        Ok(())
    }

    /// Get a job's current state.
    pub async fn get_job(&self, id: TrainingJobId, tenant_id: &TenantId) -> Result<JobState> {
        let job = self
            .jobs
            .read()
            .await
            .get(&id)
            .cloned()
            .ok_or_else(|| SynapseError::TrainingError(format!("Job {id} not found")))?;

        if &job.tenant_id != tenant_id {
            return Err(SynapseError::TrainingError(format!("Job {id} not found")));
        }

        Ok(job)
    }

    /// List all jobs, optionally filtered by status, scoped to a tenant.
    pub async fn list_jobs(
        &self,
        status_filter: Option<TrainingStatus>,
        tenant_id: &TenantId,
    ) -> Vec<JobState> {
        let jobs = self.jobs.read().await;
        jobs.values()
            .filter(|j| &j.tenant_id == tenant_id)
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
            self.persist(state);
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
    use synapse_types::TenantId;
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
            max_steps: None,
            time_budget_secs: None,
        }
    }

    #[tokio::test]
    async fn create_and_get_job() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let tenant = TenantId::default_tenant();
        let id = manager
            .create_job(test_config(), tenant.clone())
            .await
            .unwrap();
        let job = manager.get_job(id, &tenant).await.unwrap();
        assert_eq!(job.status, TrainingStatus::Queued);
        assert_eq!(job.id, id);
    }

    #[tokio::test]
    async fn list_jobs_all() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let tenant = TenantId::default_tenant();
        manager
            .create_job(test_config(), tenant.clone())
            .await
            .unwrap();
        manager
            .create_job(test_config(), tenant.clone())
            .await
            .unwrap();
        let all = manager.list_jobs(None, &tenant).await;
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn list_jobs_filtered() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let tenant = TenantId::default_tenant();
        manager
            .create_job(test_config(), tenant.clone())
            .await
            .unwrap();
        let queued = manager
            .list_jobs(Some(TrainingStatus::Queued), &tenant)
            .await;
        assert_eq!(queued.len(), 1);
        let running = manager
            .list_jobs(Some(TrainingStatus::Running), &tenant)
            .await;
        assert!(running.is_empty());
    }

    #[tokio::test]
    async fn get_job_not_found() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let result = manager
            .get_job(uuid::Uuid::new_v4(), &TenantId::default_tenant())
            .await;
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
        let result = manager
            .cancel_job(uuid::Uuid::new_v4(), &TenantId::default_tenant())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn update_progress() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let tenant = TenantId::default_tenant();
        let id = manager
            .create_job(test_config(), tenant.clone())
            .await
            .unwrap();
        manager.update_progress(id, 50, 0.5, 0.42).await;
        let job = manager.get_job(id, &tenant).await.unwrap();
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

    #[tokio::test]
    async fn create_job_invalid_hyperparams() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let mut config = test_config();
        config.hyperparams.learning_rate = 0.0; // invalid
        let result = manager.create_job(config, TenantId::default_tenant()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn create_job_zero_batch_size() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let mut config = test_config();
        config.hyperparams.batch_size = 0;
        let result = manager.create_job(config, TenantId::default_tenant()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn update_progress_nonexistent_job() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        // Should not panic, just no-op
        manager
            .update_progress(uuid::Uuid::new_v4(), 10, 0.5, 0.3)
            .await;
    }

    #[tokio::test]
    async fn list_jobs_empty() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 2);
        let tenant = TenantId::default_tenant();
        assert!(manager.list_jobs(None, &tenant).await.is_empty());
        assert!(
            manager
                .list_jobs(Some(TrainingStatus::Running), &tenant)
                .await
                .is_empty()
        );
    }

    // -- Concurrent access tests --

    #[tokio::test]
    async fn concurrent_create_and_list() {
        let manager = std::sync::Arc::new(JobManager::new(ExecutorKind::Subprocess, None, 100));
        let tenant = TenantId::default_tenant();
        let mut handles = vec![];

        // Spawn 20 concurrent job creations
        for _ in 0..20 {
            let manager = manager.clone();
            let tenant = tenant.clone();
            handles.push(tokio::spawn(async move {
                manager.create_job(test_config(), tenant).await.unwrap();
            }));
        }

        // Concurrently list and count
        for _ in 0..20 {
            let manager = manager.clone();
            let tenant = tenant.clone();
            handles.push(tokio::spawn(async move {
                let _ = manager.list_jobs(None, &tenant).await;
                let _ = manager.running_count().await;
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(manager.list_jobs(None, &tenant).await.len(), 20);
    }

    #[tokio::test]
    async fn concurrent_create_and_update_progress() {
        let manager = std::sync::Arc::new(JobManager::new(ExecutorKind::Subprocess, None, 100));
        let tenant = TenantId::default_tenant();
        let mut ids = vec![];

        for _ in 0..10 {
            ids.push(
                manager
                    .create_job(test_config(), tenant.clone())
                    .await
                    .unwrap(),
            );
        }

        let mut handles = vec![];

        // Update progress concurrently
        for (i, &id) in ids.iter().enumerate() {
            let manager = manager.clone();
            handles.push(tokio::spawn(async move {
                for step in 0..10u64 {
                    manager
                        .update_progress(id, step, (i as f32) * 0.1, 0.5 - step as f64 * 0.01)
                        .await;
                }
            }));
        }

        // Concurrently read state
        for &id in &ids {
            let manager = manager.clone();
            let tenant = tenant.clone();
            handles.push(tokio::spawn(async move {
                let _ = manager.get_job(id, &tenant).await;
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // All jobs should still exist with progress
        for &id in &ids {
            let job = manager.get_job(id, &tenant).await.unwrap();
            assert_eq!(job.current_step, 9);
        }
    }
}
