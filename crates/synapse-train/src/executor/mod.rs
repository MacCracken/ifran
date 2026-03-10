pub mod docker;
pub mod native;
pub mod subprocess;

use async_trait::async_trait;
use synapse_types::error::Result;
use synapse_types::training::{TrainingJobConfig, TrainingJobId};

/// Which executor to use for training.
#[derive(Debug, Clone, Copy)]
pub enum ExecutorKind {
    Docker,
    Subprocess,
}

/// Trait for training executors that launch and manage training workloads.
#[async_trait]
pub trait TrainingExecutor: Send + Sync {
    /// Run a training job to completion.
    async fn run(&self, config: &TrainingJobConfig, job_id: TrainingJobId) -> Result<()>;

    /// Cancel a running job.
    async fn cancel(&self, job_id: TrainingJobId) -> Result<()>;
}
