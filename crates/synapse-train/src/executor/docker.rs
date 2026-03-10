//! Docker-based training executor for containerized training workloads.
//!
//! Launches a Docker container with the training image, mounts the dataset
//! and checkpoint directories, and passes the training config as JSON.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use synapse_types::SynapseError;
use synapse_types::error::Result;
use synapse_types::training::{TrainingJobConfig, TrainingJobId, TrainingMethod};
use tokio::process::Command;
use tokio::sync::RwLock;
use tracing::info;

use super::TrainingExecutor;

pub struct DockerExecutor {
    image: String,
    /// Track container IDs for cancellation.
    containers: Arc<RwLock<HashMap<TrainingJobId, String>>>,
}

impl DockerExecutor {
    pub fn new(image: String) -> Self {
        Self {
            image,
            containers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl TrainingExecutor for DockerExecutor {
    async fn run(&self, config: &TrainingJobConfig, job_id: TrainingJobId) -> Result<()> {
        let config_json = serde_json::to_string(config)
            .map_err(|e| SynapseError::TrainingError(e.to_string()))?;

        let container_name = format!("synapse-train-{}", job_id);
        let script = match config.method {
            TrainingMethod::Lora | TrainingMethod::Qlora => "scripts/train_sft.py",
            TrainingMethod::FullFineTune => "scripts/train_full.py",
            TrainingMethod::Dpo => "scripts/train_dpo.py",
            TrainingMethod::Rlhf => "scripts/train_rlhf.py",
            TrainingMethod::Distillation => "scripts/train_distill.py",
        };

        info!(job_id = %job_id, image = %self.image, script = %script, "Starting training container");

        let output = Command::new("docker")
            .args([
                "run",
                "--name",
                &container_name,
                "--gpus",
                "all",
                "--rm",
                "-e",
                &format!("TRAINING_CONFIG={config_json}"),
                "-e",
                &format!("JOB_ID={job_id}"),
                "-v",
                &format!("{}:/workspace/datasets/data", config.dataset.path),
                &self.image,
                script,
            ])
            .output()
            .await
            .map_err(|e| SynapseError::TrainingError(format!("Failed to start Docker: {e}")))?;

        // Track container for cancellation
        self.containers
            .write()
            .await
            .insert(job_id, container_name.clone());

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            self.containers.write().await.remove(&job_id);
            return Err(SynapseError::TrainingError(format!(
                "Training container failed: {stderr}"
            )));
        }

        self.containers.write().await.remove(&job_id);
        info!(job_id = %job_id, "Training container completed");
        Ok(())
    }

    async fn cancel(&self, job_id: TrainingJobId) -> Result<()> {
        let containers = self.containers.read().await;
        if let Some(name) = containers.get(&job_id) {
            let _ = Command::new("docker").args(["stop", name]).output().await;
            info!(job_id = %job_id, "Stopped training container");
        }
        Ok(())
    }
}
