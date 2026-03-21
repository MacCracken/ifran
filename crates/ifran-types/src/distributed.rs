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
    /// Optional placement policy for worker distribution.
    pub placement_policy: Option<PlacementPolicyKind>,
}

/// Placement policy for distributing workers across nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementPolicyKind {
    GpuAffinity,
    Balanced,
    CostAware,
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
    /// Ifran instance ID.
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
    /// Tenant that owns this job.
    pub tenant_id: String,
    pub workers: Vec<WorkerAssignment>,
    pub status: TrainingStatus,
    /// Aggregated loss across all workers.
    pub aggregate_loss: Option<f64>,
    /// Number of workers that have completed.
    pub completed_workers: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{
        DatasetConfig, DatasetFormat, HyperParams, TrainingJobConfig, TrainingMethod,
    };

    #[test]
    fn distributed_strategy_serde_roundtrip() {
        let strategies = [
            DistributedStrategy::DataParallel,
            DistributedStrategy::ModelParallel,
            DistributedStrategy::PipelineParallel,
        ];
        for s in &strategies {
            let json = serde_json::to_string(s).unwrap();
            let back: DistributedStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(*s, back);
        }
    }

    #[test]
    fn distributed_strategy_json_values() {
        assert_eq!(
            serde_json::to_string(&DistributedStrategy::DataParallel).unwrap(),
            "\"data_parallel\""
        );
        assert_eq!(
            serde_json::to_string(&DistributedStrategy::ModelParallel).unwrap(),
            "\"model_parallel\""
        );
    }

    #[test]
    fn worker_assignment_serde() {
        let w = WorkerAssignment {
            rank: 0,
            instance_id: "node-1".into(),
            endpoint: "http://node-1:50051".into(),
            device_ids: vec![0, 1],
        };
        let json = serde_json::to_string(&w).unwrap();
        let back: WorkerAssignment = serde_json::from_str(&json).unwrap();
        assert_eq!(back.rank, 0);
        assert_eq!(back.device_ids, vec![0, 1]);
    }

    fn make_test_config() -> DistributedTrainingConfig {
        DistributedTrainingConfig {
            base_config: TrainingJobConfig {
                base_model: "llama-7b".into(),
                dataset: DatasetConfig {
                    path: "/data/train.jsonl".into(),
                    format: DatasetFormat::Jsonl,
                    split: None,
                    max_samples: None,
                },
                method: TrainingMethod::Lora,
                hyperparams: HyperParams {
                    learning_rate: 2e-4,
                    epochs: 3,
                    batch_size: 8,
                    gradient_accumulation_steps: 4,
                    warmup_steps: 100,
                    weight_decay: 0.01,
                    max_seq_length: 2048,
                },
                output_name: None,
                lora: None,
                max_steps: None,
                time_budget_secs: None,
            },
            world_size: 4,
            strategy: DistributedStrategy::DataParallel,
            placement_policy: None,
        }
    }

    #[test]
    fn distributed_training_config_serde() {
        let config = make_test_config();
        let json = serde_json::to_string(&config).unwrap();
        let back: DistributedTrainingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.world_size, 4);
        assert_eq!(back.strategy, DistributedStrategy::DataParallel);
    }

    #[test]
    fn placement_policy_kind_serde_roundtrip() {
        let values = [
            PlacementPolicyKind::GpuAffinity,
            PlacementPolicyKind::Balanced,
            PlacementPolicyKind::CostAware,
        ];
        for v in &values {
            let json = serde_json::to_string(v).unwrap();
            let back: PlacementPolicyKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*v, back);
        }
    }

    #[test]
    fn placement_policy_kind_json_values() {
        assert_eq!(
            serde_json::to_string(&PlacementPolicyKind::GpuAffinity).unwrap(),
            "\"gpu_affinity\""
        );
        assert_eq!(
            serde_json::to_string(&PlacementPolicyKind::Balanced).unwrap(),
            "\"balanced\""
        );
        assert_eq!(
            serde_json::to_string(&PlacementPolicyKind::CostAware).unwrap(),
            "\"cost_aware\""
        );
    }

    #[test]
    fn distributed_config_with_placement_policy() {
        let mut config = make_test_config();
        config.placement_policy = Some(PlacementPolicyKind::CostAware);
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"cost_aware\""));
        let back: DistributedTrainingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.placement_policy, Some(PlacementPolicyKind::CostAware));
    }

    #[test]
    fn distributed_job_state_serde() {
        let state = DistributedJobState {
            job_id: Uuid::new_v4(),
            config: make_test_config(),
            coordinator: "node-1".into(),
            tenant_id: "default".into(),
            workers: vec![
                WorkerAssignment {
                    rank: 0,
                    instance_id: "node-1".into(),
                    endpoint: "http://node-1:50051".into(),
                    device_ids: vec![0],
                },
                WorkerAssignment {
                    rank: 1,
                    instance_id: "node-2".into(),
                    endpoint: "http://node-2:50051".into(),
                    device_ids: vec![0],
                },
            ],
            status: TrainingStatus::Running,
            aggregate_loss: Some(0.5),
            completed_workers: 0,
        };
        let json = serde_json::to_string(&state).unwrap();
        let back: DistributedJobState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.workers.len(), 2);
        assert_eq!(back.status, TrainingStatus::Running);
        assert_eq!(back.aggregate_loss, Some(0.5));
    }
}
