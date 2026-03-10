//! Subprocess-based training executor for running Python training scripts
//! as child processes.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use synapse_types::SynapseError;
use synapse_types::error::Result;
use synapse_types::training::{TrainingJobConfig, TrainingJobId, TrainingMethod};
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
        let script = match config.method {
            TrainingMethod::Lora | TrainingMethod::Qlora => "scripts/train_sft.py",
            TrainingMethod::FullFineTune => "scripts/train_full.py",
            TrainingMethod::Dpo => "scripts/train_dpo.py",
            TrainingMethod::Rlhf => "scripts/train_rlhf.py",
            TrainingMethod::Distillation => "scripts/train_distill.py",
        };

        let config_json = serde_json::to_string(config)
            .map_err(|e| SynapseError::TrainingError(e.to_string()))?;

        info!(job_id = %job_id, script = %script, "Starting training subprocess");

        let child = Command::new("python3")
            .args([script, "--config-json", &config_json])
            .env("JOB_ID", job_id.to_string())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| SynapseError::TrainingError(format!("Failed to spawn: {e}")))?;

        self.processes.write().await.insert(job_id, child);

        // Wait for completion
        let status = {
            let mut procs = self.processes.write().await;
            if let Some(child) = procs.get_mut(&job_id) {
                child
                    .wait()
                    .await
                    .map_err(|e| SynapseError::TrainingError(e.to_string()))?
            } else {
                return Err(SynapseError::TrainingError("Process disappeared".into()));
            }
        };

        self.processes.write().await.remove(&job_id);

        if !status.success() {
            return Err(SynapseError::TrainingError(format!(
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
