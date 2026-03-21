//! Subprocess-based training executor for running Python training scripts
//! as child processes.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use ifran_types::IfranError;
use ifran_types::error::Result;
use ifran_types::training::{TrainingJobConfig, TrainingJobId};
use tokio::process::{Child, Command};
use tokio::sync::RwLock;
use tracing::info;

use super::TrainingExecutor;

#[derive(Default)]
pub struct SubprocessExecutor {
    processes: Arc<RwLock<HashMap<TrainingJobId, Child>>>,
}

impl SubprocessExecutor {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl TrainingExecutor for SubprocessExecutor {
    async fn run(&self, config: &TrainingJobConfig, job_id: TrainingJobId) -> Result<()> {
        let script = super::script_for_method(config.method);

        let config_json = serde_json::to_string(config)
            .map_err(|e| IfranError::TrainingError(e.to_string()))?;

        info!(job_id = %job_id, script = %script, "Starting training subprocess");

        let child = Command::new("python3")
            .args([script, "--config-json", &config_json])
            .env("JOB_ID", job_id.to_string())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| IfranError::TrainingError(format!("Failed to spawn: {e}")))?;

        self.processes.write().await.insert(job_id, child);

        // Wait for completion — take the child out of the map first to avoid
        // holding the write lock during the potentially long wait. This prevents
        // deadlock if cancel() is called concurrently.
        let mut child = {
            let mut procs = self.processes.write().await;
            match procs.remove(&job_id) {
                Some(child) => child,
                None => {
                    return Err(IfranError::TrainingError("Process disappeared".into()));
                }
            }
        };

        // Apply time budget if configured (budget + 30s grace period)
        let result = if let Some(budget) = config.time_budget_secs {
            let timeout_dur = std::time::Duration::from_secs(budget + 30);
            match tokio::time::timeout(timeout_dur, child.wait()).await {
                Ok(wait_result) => wait_result,
                Err(_) => {
                    let _ = child.kill().await;
                    info!(job_id = %job_id, budget_secs = budget, "Training subprocess timed out (expected)");
                    return Ok(());
                }
            }
        } else {
            child.wait().await
        };

        let status = result.map_err(|e| IfranError::TrainingError(e.to_string()))?;

        if !status.success() {
            return Err(IfranError::TrainingError(format!(
                "Training script exited with code: {:?}",
                status.code()
            )));
        }

        info!(job_id = %job_id, "Training subprocess completed");
        Ok(())
    }

    async fn cancel(&self, job_id: TrainingJobId) -> Result<()> {
        let mut procs = self.processes.write().await;
        if let Some(child) = procs.get_mut(&job_id) {
            let _ = child.kill().await;
            info!(job_id = %job_id, "Killed training subprocess");
        }
        procs.remove(&job_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_default() {
        let executor = SubprocessExecutor::new();
        // Should compile and not panic
        assert!(std::mem::size_of_val(&executor) > 0);
    }

    #[test]
    fn default_creates_instance() {
        let executor = SubprocessExecutor::default();
        assert!(std::mem::size_of_val(&executor) > 0);
    }

    #[tokio::test]
    async fn cancel_nonexistent_job_succeeds() {
        let executor = SubprocessExecutor::new();
        let job_id = uuid::Uuid::new_v4();
        let result = executor.cancel(job_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn processes_starts_empty() {
        let executor = SubprocessExecutor::new();
        assert!(executor.processes.read().await.is_empty());
    }

    #[tokio::test]
    async fn cancel_removes_tracked_process() {
        let executor = SubprocessExecutor::new();
        let job_id = uuid::Uuid::new_v4();

        // Insert a dummy process (sleep)
        let child = tokio::process::Command::new("sleep")
            .arg("60")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();

        executor.processes.write().await.insert(job_id, child);
        assert_eq!(executor.processes.read().await.len(), 1);

        executor.cancel(job_id).await.unwrap();
        assert!(executor.processes.read().await.is_empty());
    }

    #[test]
    fn config_serializes_for_args() {
        use ifran_types::training::*;
        let config = TrainingJobConfig {
            base_model: "llama-7b".into(),
            dataset: DatasetConfig {
                path: "/data/train.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: None,
            },
            method: TrainingMethod::Dpo,
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
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("llama-7b"));
        assert!(json.contains("dpo"));
    }
}
