//! Docker-based training executor for containerized training workloads.
//!
//! Launches a Docker container with the training image, mounts the dataset
//! and checkpoint directories, and passes the training config as JSON.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use synapse_types::SynapseError;
use synapse_types::error::Result;
use synapse_types::training::{TrainingJobConfig, TrainingJobId};
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
        let script = super::script_for_method(config.method);

        info!(job_id = %job_id, image = %self.image, script = %script, "Starting training container");

        // Track container BEFORE running so cancel() can find it during execution
        self.containers
            .write()
            .await
            .insert(job_id, container_name.clone());

        let docker_fut = Command::new("docker")
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
            .output();

        // Apply time budget if configured (budget + 30s grace period)
        let output = if let Some(budget) = config.time_budget_secs {
            let timeout_dur = std::time::Duration::from_secs(budget + 30);
            match tokio::time::timeout(timeout_dur, docker_fut).await {
                Ok(result) => match result {
                    Ok(output) => output,
                    Err(e) => {
                        self.containers.write().await.remove(&job_id);
                        return Err(SynapseError::TrainingError(format!(
                            "Failed to start Docker: {e}"
                        )));
                    }
                },
                Err(_) => {
                    // Timeout — stop the container gracefully
                    let _ = Command::new("docker")
                        .args(["stop", &container_name])
                        .output()
                        .await;
                    self.containers.write().await.remove(&job_id);
                    info!(job_id = %job_id, budget_secs = budget, "Training container timed out (expected)");
                    return Ok(());
                }
            }
        } else {
            match docker_fut.await {
                Ok(output) => output,
                Err(e) => {
                    self.containers.write().await.remove(&job_id);
                    return Err(SynapseError::TrainingError(format!(
                        "Failed to start Docker: {e}"
                    )));
                }
            }
        };

        self.containers.write().await.remove(&job_id);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SynapseError::TrainingError(format!(
                "Training container failed: {stderr}"
            )));
        }

        info!(job_id = %job_id, "Training container completed");
        Ok(())
    }

    async fn cancel(&self, job_id: TrainingJobId) -> Result<()> {
        let mut containers = self.containers.write().await;
        if let Some(name) = containers.remove(&job_id) {
            match Command::new("docker").args(["stop", &name]).output().await {
                Ok(output) if !output.status.success() => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    tracing::warn!(job_id = %job_id, error = %stderr, "Docker stop returned error");
                }
                Err(e) => {
                    tracing::warn!(job_id = %job_id, error = %e, "Failed to execute docker stop");
                }
                _ => {}
            }
            info!(job_id = %job_id, "Stopped training container");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sets_image() {
        let executor = DockerExecutor::new("my-training:latest".into());
        assert_eq!(executor.image, "my-training:latest");
    }

    #[test]
    fn container_name_format() {
        let job_id = uuid::Uuid::new_v4();
        let name = format!("synapse-train-{}", job_id);
        assert!(name.starts_with("synapse-train-"));
        assert!(name.len() > "synapse-train-".len());
    }

    #[tokio::test]
    async fn cancel_nonexistent_job_succeeds() {
        let executor = DockerExecutor::new("test:latest".into());
        let job_id = uuid::Uuid::new_v4();
        // Cancel with no tracked container should succeed silently
        let result = executor.cancel(job_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn containers_starts_empty() {
        let executor = DockerExecutor::new("test:latest".into());
        assert!(executor.containers.read().await.is_empty());
    }

    #[test]
    fn config_serializes_for_env() {
        use synapse_types::training::*;
        let config = TrainingJobConfig {
            base_model: "llama-7b".into(),
            dataset: DatasetConfig {
                path: "/data/train.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: Some(1000),
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
            output_name: Some("my-finetuned".into()),
            lora: None,
            max_steps: None,
            time_budget_secs: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("llama-7b"));
        assert!(json.contains("/data/train.jsonl"));
        assert!(json.contains("lora"));
    }
}
