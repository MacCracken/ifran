//! Training job manager — creates, tracks, and manages training job lifecycles.

use crate::budget::checker::BudgetChecker;
use crate::storage::traits::JobStore;
use crate::train::approval::gate::{ApprovalGate, ApprovalId};
use crate::train::executor::{ExecutorKind, TrainingExecutor};
use crate::train::job::status::JobState;
use crate::types::IfranError;
use crate::types::TenantId;
use crate::types::error::Result;
use crate::types::training::{TrainingJobConfig, TrainingJobId, TrainingMethod, TrainingStatus};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Returns true if the training method requires human approval before starting.
#[must_use]
#[inline]
fn requires_approval(method: TrainingMethod) -> bool {
    matches!(
        method,
        TrainingMethod::Rlhf | TrainingMethod::Dpo | TrainingMethod::FullFineTune
    )
}

/// Estimate GPU hours from a training config for budget enforcement.
#[must_use]
fn estimate_gpu_hours(config: &TrainingJobConfig) -> f64 {
    let samples = config.dataset.max_samples.unwrap_or(10_000) as f64;
    let epochs = config.hyperparams.epochs.max(1) as f64;
    let batch_size = config.hyperparams.batch_size.max(1) as f64;
    let steps = (samples / batch_size) * epochs;
    // Rough estimate: ~0.001 GPU-hours per step for a 7B model
    // This is intentionally conservative; real usage is reported after completion
    steps * 0.001
}

/// Manages all training jobs.
pub struct JobManager {
    jobs: Arc<RwLock<HashMap<TrainingJobId, JobState>>>,
    executor: Arc<dyn TrainingExecutor>,
    max_concurrent: usize,
    store: Option<Arc<Mutex<dyn JobStore>>>,
    approval_gate: Arc<Mutex<ApprovalGate>>,
    /// Maps job IDs to their approval request IDs.
    approval_map: Arc<RwLock<HashMap<TrainingJobId, ApprovalId>>>,
    budget_checker: Option<Arc<BudgetChecker>>,
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
            approval_gate: Arc::new(Mutex::new(ApprovalGate::new())),
            approval_map: Arc::new(RwLock::new(HashMap::new())),
            budget_checker: None,
        }
    }

    /// Create a JobManager backed by a persistent store.
    pub fn new_with_store(
        executor_kind: ExecutorKind,
        trainer_image: Option<String>,
        max_concurrent: usize,
        store: impl JobStore + 'static,
    ) -> Self {
        let executor = Self::make_executor(executor_kind, trainer_image);

        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            executor,
            max_concurrent,
            store: Some(Arc::new(Mutex::new(store))),
            approval_gate: Arc::new(Mutex::new(ApprovalGate::new())),
            approval_map: Arc::new(RwLock::new(HashMap::new())),
            budget_checker: None,
        }
    }

    /// Get a reference to the approval gate.
    #[must_use]
    pub fn approval_gate(&self) -> &Arc<Mutex<ApprovalGate>> {
        &self.approval_gate
    }

    fn make_executor(
        executor_kind: ExecutorKind,
        trainer_image: Option<String>,
    ) -> Arc<dyn TrainingExecutor> {
        match executor_kind {
            ExecutorKind::Docker => Arc::new(crate::train::executor::docker::DockerExecutor::new(
                trainer_image.unwrap_or_else(|| "ghcr.io/maccracken/ifran-trainer:latest".into()),
            )),
            ExecutorKind::Subprocess => {
                Arc::new(crate::train::executor::subprocess::SubprocessExecutor::new())
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

    /// Start a queued job. The caller must provide the tenant_id to verify ownership.
    ///
    /// High-risk methods (RLHF, DPO, FullFineTune) enter `PendingApproval` status
    /// instead of starting immediately. Use `approve_job()` to proceed.
    pub async fn start_job(&self, id: TrainingJobId, tenant_id: &TenantId) -> Result<()> {
        // Budget check before acquiring write lock (async call)
        {
            let jobs = self.jobs.read().await;
            let state = jobs
                .get(&id)
                .ok_or_else(|| IfranError::TrainingError(format!("Job {id} not found")))?;
            if &state.tenant_id != tenant_id {
                return Err(IfranError::TrainingError(format!("Job {id} not found")));
            }
            if let Some(checker) = &self.budget_checker {
                let gpu_hours = estimate_gpu_hours(&state.config);
                let status = checker.check_budget(tenant_id, gpu_hours).await?;
                if !status.allowed {
                    return Err(IfranError::BudgetExceeded(
                        status
                            .reason
                            .unwrap_or_else(|| "GPU budget exceeded".into()),
                    ));
                }
            }
        }

        let mut jobs = self.jobs.write().await;

        // Check running count inside the write lock to prevent race condition
        // where multiple concurrent start_job calls all pass the check
        let running_count = jobs
            .values()
            .filter(|j| j.status == TrainingStatus::Running)
            .count();
        if running_count >= self.max_concurrent {
            return Err(IfranError::TrainingError(format!(
                "Max concurrent jobs ({}) reached",
                self.max_concurrent
            )));
        }

        let state = jobs
            .get_mut(&id)
            .ok_or_else(|| IfranError::TrainingError(format!("Job {id} not found")))?;

        // Verify tenant ownership
        if &state.tenant_id != tenant_id {
            return Err(IfranError::TrainingError(format!("Job {id} not found")));
        }

        if state.status != TrainingStatus::Queued {
            return Err(IfranError::TrainingError(format!(
                "Job {id} is {:?}, not Queued",
                state.status
            )));
        }

        // High-risk methods require human approval before execution
        if requires_approval(state.config.method) {
            state.set_pending_approval();
            self.persist(state);
            let approval_req = self
                .approval_gate
                .lock()
                .map_err(|e| {
                    IfranError::TrainingError(format!("Failed to lock approval gate: {e}"))
                })?
                .request_approval(
                    &format!("training-{id}"),
                    "job_start",
                    &state.config.base_model,
                );
            self.approval_map.write().await.insert(id, approval_req.id);
            info!(job_id = %id, method = ?state.config.method, "Job requires approval before starting");
            return Ok(());
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

    /// Approve a pending-approval job and start it.
    pub async fn approve_job(
        &self,
        id: TrainingJobId,
        tenant_id: &TenantId,
        reviewer: &str,
        comment: Option<&str>,
    ) -> Result<()> {
        // Resolve the approval request
        let approval_id = {
            let map = self.approval_map.read().await;
            *map.get(&id).ok_or_else(|| {
                IfranError::TrainingError(format!("No pending approval for job {id}"))
            })?
        };

        self.approval_gate
            .lock()
            .map_err(|e| IfranError::TrainingError(format!("Failed to lock approval gate: {e}")))?
            .approve(approval_id, reviewer, comment);

        // Transition the job from PendingApproval → Running
        let mut jobs = self.jobs.write().await;

        let running_count = jobs
            .values()
            .filter(|j| j.status == TrainingStatus::Running)
            .count();
        if running_count >= self.max_concurrent {
            return Err(IfranError::TrainingError(format!(
                "Max concurrent jobs ({}) reached",
                self.max_concurrent
            )));
        }

        let state = jobs
            .get_mut(&id)
            .ok_or_else(|| IfranError::TrainingError(format!("Job {id} not found")))?;

        if &state.tenant_id != tenant_id {
            return Err(IfranError::TrainingError(format!("Job {id} not found")));
        }

        if state.status != TrainingStatus::PendingApproval {
            return Err(IfranError::TrainingError(format!(
                "Job {id} is {:?}, not PendingApproval",
                state.status
            )));
        }

        state.start();
        self.persist(state);
        info!(job_id = %id, reviewer, "Approved and started training job");

        let config = state.config.clone();
        let jobs_ref = self.jobs.clone();
        let executor = self.executor.clone();
        let store = self.store.clone();

        tokio::spawn(async move {
            let result = executor.run(&config, id).await;
            let mut jobs = jobs_ref.write().await;
            if let Some(state) = jobs.get_mut(&id) {
                match result {
                    Ok(()) => state.complete(),
                    Err(e) => state.fail(e.to_string()),
                }
                if let Some(store) = &store {
                    if let Ok(store) = store.lock() {
                        let _ = store.save_job(state);
                    }
                }
            }
        });

        // Clean up approval map
        self.approval_map.write().await.remove(&id);
        Ok(())
    }

    /// Reject a pending-approval job.
    pub async fn reject_job(
        &self,
        id: TrainingJobId,
        tenant_id: &TenantId,
        reviewer: &str,
        comment: Option<&str>,
    ) -> Result<()> {
        let approval_id = {
            let map = self.approval_map.read().await;
            *map.get(&id).ok_or_else(|| {
                IfranError::TrainingError(format!("No pending approval for job {id}"))
            })?
        };

        self.approval_gate
            .lock()
            .map_err(|e| IfranError::TrainingError(format!("Failed to lock approval gate: {e}")))?
            .reject(approval_id, reviewer, comment);

        let mut jobs = self.jobs.write().await;
        let state = jobs
            .get_mut(&id)
            .ok_or_else(|| IfranError::TrainingError(format!("Job {id} not found")))?;

        if &state.tenant_id != tenant_id {
            return Err(IfranError::TrainingError(format!("Job {id} not found")));
        }

        let reason = comment.unwrap_or("Approval rejected");
        state.fail(format!("Approval rejected by {reviewer}: {reason}"));
        self.persist(state);
        info!(job_id = %id, reviewer, "Rejected training job");

        self.approval_map.write().await.remove(&id);
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
                .ok_or_else(|| IfranError::TrainingError(format!("Job {id} not found")))?;

            if &state.tenant_id != tenant_id {
                return Err(IfranError::TrainingError(format!("Job {id} not found")));
            }

            if state.is_terminal() {
                return Err(IfranError::TrainingError(format!(
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
            .ok_or_else(|| IfranError::TrainingError(format!("Job {id} not found")))?;

        if &job.tenant_id != tenant_id {
            return Err(IfranError::TrainingError(format!("Job {id} not found")));
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
    #[must_use]
    #[inline]
    pub async fn running_count(&self) -> usize {
        self.jobs
            .read()
            .await
            .values()
            .filter(|j| j.status == TrainingStatus::Running)
            .count()
    }

    /// Count of currently queued jobs.
    #[must_use]
    #[inline]
    pub async fn queued_count(&self) -> usize {
        self.jobs
            .read()
            .await
            .values()
            .filter(|j| j.status == TrainingStatus::Queued)
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
    #[must_use]
    #[inline]
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Evict terminal jobs (completed/failed/cancelled) older than `ttl`.
    /// Removes from both in-memory map and persistent store.
    pub async fn evict_completed(&self, ttl: Duration) -> usize {
        let cutoff =
            Utc::now() - chrono::Duration::from_std(ttl).unwrap_or(chrono::Duration::zero());
        let mut jobs = self.jobs.write().await;
        let to_evict: Vec<TrainingJobId> = jobs
            .values()
            .filter(|j| j.is_terminal())
            .filter(|j| j.completed_at.is_some_and(|t| t < cutoff))
            .map(|j| j.id)
            .collect();

        let count = to_evict.len();
        for id in &to_evict {
            jobs.remove(id);
            if let Some(store) = &self.store {
                if let Ok(store) = store.lock() {
                    let _ = store.delete_job(*id);
                }
            }
        }

        if count > 0 {
            info!(count, "Evicted terminal jobs past TTL");
        }
        count
    }

    /// Start a background loop that periodically evicts terminal jobs.
    /// `ttl_secs == 0` means eviction is disabled.
    pub fn start_eviction_loop(self: &Arc<Self>, ttl_secs: u64) {
        if ttl_secs == 0 {
            return;
        }
        let manager = Arc::clone(self);
        let ttl = Duration::from_secs(ttl_secs);
        // Run eviction every hour or every TTL, whichever is shorter.
        let interval = Duration::from_secs(ttl_secs.min(3600));
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await; // first tick is immediate, skip it
            loop {
                ticker.tick().await;
                manager.evict_completed(ttl).await;
            }
        });
    }

    /// Cancel all non-terminal jobs belonging to a tenant.
    /// Used when a tenant is disabled.
    pub async fn cancel_tenant_jobs(&self, tenant_id: &TenantId) -> Result<usize> {
        // Collect cancellable job IDs first (read lock).
        let ids: Vec<TrainingJobId> = {
            let jobs = self.jobs.read().await;
            jobs.values()
                .filter(|j| &j.tenant_id == tenant_id && !j.is_terminal())
                .map(|j| j.id)
                .collect()
        };

        let mut cancelled = 0;
        for id in &ids {
            // Best-effort: executor cancel may fail if the job finished in between.
            let _ = self.executor.cancel(*id).await;
            let mut jobs = self.jobs.write().await;
            if let Some(state) = jobs.get_mut(id) {
                if !state.is_terminal() {
                    state.cancel();
                    self.persist(state);
                    cancelled += 1;
                }
            }
        }

        if cancelled > 0 {
            info!(tenant_id = %tenant_id.0, cancelled, "Cancelled in-flight jobs for disabled tenant");
        }
        Ok(cancelled)
    }
}

fn estimate_total_steps(config: &TrainingJobConfig) -> u64 {
    // Rough estimate: epochs * (assumed dataset size / batch_size)
    let batch = config.hyperparams.batch_size.max(1) as u64;
    let epochs = config.hyperparams.epochs.max(1) as u64;
    let assumed_samples = config.dataset.max_samples.unwrap_or(10000) as u64;
    (assumed_samples / batch).saturating_mul(epochs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TenantId;
    use crate::types::training::*;

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

    #[tokio::test]
    async fn list_jobs_tenant_filtered() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant_a = TenantId("tenant-a".into());
        let tenant_b = TenantId("tenant-b".into());

        manager
            .create_job(test_config(), tenant_a.clone())
            .await
            .unwrap();
        manager
            .create_job(test_config(), tenant_a.clone())
            .await
            .unwrap();
        manager
            .create_job(test_config(), tenant_b.clone())
            .await
            .unwrap();

        assert_eq!(manager.list_jobs(None, &tenant_a).await.len(), 2);
        assert_eq!(manager.list_jobs(None, &tenant_b).await.len(), 1);
    }

    #[tokio::test]
    async fn get_job_wrong_tenant() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant_a = TenantId("tenant-a".into());
        let tenant_b = TenantId("tenant-b".into());

        let id = manager
            .create_job(test_config(), tenant_a.clone())
            .await
            .unwrap();

        // Should fail when accessed with wrong tenant
        let result = manager.get_job(id, &tenant_b).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn cancel_job_wrong_tenant() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant_a = TenantId("tenant-a".into());
        let tenant_b = TenantId("tenant-b".into());

        let id = manager
            .create_job(test_config(), tenant_a.clone())
            .await
            .unwrap();

        // Should fail when cancelled with wrong tenant
        let result = manager.cancel_job(id, &tenant_b).await;
        assert!(result.is_err());
    }

    // -- Approval gate tests --

    fn rlhf_config() -> TrainingJobConfig {
        let mut cfg = test_config();
        cfg.method = TrainingMethod::Rlhf;
        cfg
    }

    fn dpo_config() -> TrainingJobConfig {
        let mut cfg = test_config();
        cfg.method = TrainingMethod::Dpo;
        cfg
    }

    fn full_finetune_config() -> TrainingJobConfig {
        let mut cfg = test_config();
        cfg.method = TrainingMethod::FullFineTune;
        cfg
    }

    #[test]
    fn requires_approval_for_high_risk_methods() {
        assert!(super::requires_approval(TrainingMethod::Rlhf));
        assert!(super::requires_approval(TrainingMethod::Dpo));
        assert!(super::requires_approval(TrainingMethod::FullFineTune));
        assert!(!super::requires_approval(TrainingMethod::Lora));
        assert!(!super::requires_approval(TrainingMethod::Qlora));
        assert!(!super::requires_approval(TrainingMethod::Distillation));
    }

    #[test]
    fn estimate_gpu_hours_calculation() {
        let config = test_config();
        let hours = super::estimate_gpu_hours(&config);
        // 100 samples / 4 batch * 1 epoch = 25 steps * 0.001 = 0.025 hours
        assert!((hours - 0.025).abs() < 0.001);
    }

    #[test]
    fn estimate_gpu_hours_no_max_samples() {
        let mut config = test_config();
        config.dataset.max_samples = None;
        let hours = super::estimate_gpu_hours(&config);
        // 10000 default / 4 batch * 1 epoch = 2500 steps * 0.001 = 2.5 hours
        assert!((hours - 2.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn rlhf_job_enters_pending_approval() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant = TenantId::default_tenant();

        let id = manager
            .create_job(rlhf_config(), tenant.clone())
            .await
            .unwrap();

        // start_job should put it into PendingApproval, not Running
        manager.start_job(id, &tenant).await.unwrap();
        let job = manager.get_job(id, &tenant).await.unwrap();
        assert_eq!(job.status, TrainingStatus::PendingApproval);
    }

    #[tokio::test]
    async fn dpo_job_enters_pending_approval() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant = TenantId::default_tenant();

        let id = manager
            .create_job(dpo_config(), tenant.clone())
            .await
            .unwrap();

        manager.start_job(id, &tenant).await.unwrap();
        let job = manager.get_job(id, &tenant).await.unwrap();
        assert_eq!(job.status, TrainingStatus::PendingApproval);
    }

    #[tokio::test]
    async fn full_finetune_enters_pending_approval() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant = TenantId::default_tenant();

        let id = manager
            .create_job(full_finetune_config(), tenant.clone())
            .await
            .unwrap();

        manager.start_job(id, &tenant).await.unwrap();
        let job = manager.get_job(id, &tenant).await.unwrap();
        assert_eq!(job.status, TrainingStatus::PendingApproval);
    }

    #[tokio::test]
    async fn lora_job_starts_directly() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant = TenantId::default_tenant();

        let id = manager
            .create_job(test_config(), tenant.clone())
            .await
            .unwrap();

        manager.start_job(id, &tenant).await.unwrap();
        let job = manager.get_job(id, &tenant).await.unwrap();
        // LoRA doesn't require approval — should go straight to Running
        assert_eq!(job.status, TrainingStatus::Running);
    }

    #[tokio::test]
    async fn approve_job_transitions_to_running() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant = TenantId::default_tenant();

        let id = manager
            .create_job(rlhf_config(), tenant.clone())
            .await
            .unwrap();

        manager.start_job(id, &tenant).await.unwrap();
        assert_eq!(
            manager.get_job(id, &tenant).await.unwrap().status,
            TrainingStatus::PendingApproval
        );

        // Approve should transition to Running
        manager
            .approve_job(id, &tenant, "admin", Some("Looks good"))
            .await
            .unwrap();

        let job = manager.get_job(id, &tenant).await.unwrap();
        assert_eq!(job.status, TrainingStatus::Running);
    }

    #[tokio::test]
    async fn reject_job_transitions_to_failed() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant = TenantId::default_tenant();

        let id = manager
            .create_job(dpo_config(), tenant.clone())
            .await
            .unwrap();

        manager.start_job(id, &tenant).await.unwrap();

        manager
            .reject_job(id, &tenant, "reviewer", Some("Too risky"))
            .await
            .unwrap();

        let job = manager.get_job(id, &tenant).await.unwrap();
        assert_eq!(job.status, TrainingStatus::Failed);
        assert!(job.error.unwrap().contains("Too risky"));
    }

    #[tokio::test]
    async fn approve_nonexistent_job_fails() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant = TenantId::default_tenant();
        let result = manager
            .approve_job(uuid::Uuid::new_v4(), &tenant, "admin", None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn reject_nonexistent_job_fails() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant = TenantId::default_tenant();
        let result = manager
            .reject_job(uuid::Uuid::new_v4(), &tenant, "admin", None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn approve_wrong_tenant_fails() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant_a = TenantId("a".into());
        let tenant_b = TenantId("b".into());

        let id = manager
            .create_job(rlhf_config(), tenant_a.clone())
            .await
            .unwrap();
        manager.start_job(id, &tenant_a).await.unwrap();

        let result = manager.approve_job(id, &tenant_b, "admin", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn approval_gate_accessor() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let gate = manager.approval_gate();
        let locked = gate.lock().unwrap();
        // Fresh gate has no pending requests
        assert!(locked.pending().is_empty());
    }

    #[tokio::test]
    async fn pending_approval_listed_in_gate() {
        let manager = JobManager::new(ExecutorKind::Subprocess, None, 10);
        let tenant = TenantId::default_tenant();

        let id = manager
            .create_job(rlhf_config(), tenant.clone())
            .await
            .unwrap();
        manager.start_job(id, &tenant).await.unwrap();

        let gate = manager.approval_gate();
        let locked = gate.lock().unwrap();
        assert_eq!(locked.pending().len(), 1);
    }
}
