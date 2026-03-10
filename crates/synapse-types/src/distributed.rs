//! Distributed training types.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::training::{TrainingJobConfig, TrainingStatus};

pub type DistributedJobId = Uuid;

/// Configuration for a distributed training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTrainingConfig {
    /// Base training config (method, hyperparams, dataset, etc.)
    pub base_config: TrainingJobConfig,
    /// Total number of workers (including coordinator).
    pub world_size: u32,
    /// Parallelism strategy.
    pub strategy: DistributedStrategy,
}

/// Parallelism strategy for distributed training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributedStrategy {
    /// Each worker trains on a data shard with full model copy.
    DataParallel,
    /// Model is split across workers.
    ModelParallel,
    /// Pipeline stages across workers.
    PipelineParallel,
}

/// Assignment of a worker in a distributed job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerAssignment {
    /// Worker rank (0 = coordinator).
    pub rank: u32,
    /// Synapse instance ID.
    pub instance_id: String,
    /// Worker's gRPC endpoint.
    pub endpoint: String,
    /// GPU device IDs assigned to this worker.
    pub device_ids: Vec<u32>,
}

/// State of a distributed training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedJobState {
    pub job_id: DistributedJobId,
    pub config: DistributedTrainingConfig,
    /// Instance ID of the coordinator node.
    pub coordinator: String,
    pub workers: Vec<WorkerAssignment>,
    pub status: TrainingStatus,
    /// Aggregated loss across all workers.
    pub aggregate_loss: Option<f64>,
    /// Number of workers that have completed.
    pub completed_workers: u32,
}
