//! Distributed training worker — runs a training shard on this node.

use synapse_types::SynapseError;
use synapse_types::distributed::*;
use synapse_types::error::Result;
use synapse_types::training::TrainingStatus;

/// Represents a worker's local state in a distributed training job.
///
/// MVP: Tracks assignment and status. The actual training is delegated
/// to the existing executor infrastructure with shard-specific args.
pub struct DistributedWorker {
    pub assignment: WorkerAssignment,
    pub job_id: DistributedJobId,
    pub strategy: DistributedStrategy,
    pub status: TrainingStatus,
}

impl DistributedWorker {
    /// Create a new worker for a distributed job.
    pub fn new(
        job_id: DistributedJobId,
        assignment: WorkerAssignment,
        strategy: DistributedStrategy,
    ) -> Self {
        Self {
            assignment,
            job_id,
            strategy,
            status: TrainingStatus::Queued,
        }
    }

    /// Build extra CLI args for the training script based on distributed config.
    ///
    /// These get appended to the base training command so the script knows
    /// its rank, world size, and parallelism strategy.
    pub fn extra_args(&self, world_size: u32) -> Vec<String> {
        vec![
            "--rank".into(),
            self.assignment.rank.to_string(),
            "--world-size".into(),
            world_size.to_string(),
            "--strategy".into(),
            strategy_flag(self.strategy),
        ]
    }

    /// Mark this worker as running.
    pub fn start(&mut self) -> Result<()> {
        if self.status != TrainingStatus::Queued {
            return Err(SynapseError::DistributedError(format!(
                "Worker rank {} not in Queued state",
                self.assignment.rank
            )));
        }
        self.status = TrainingStatus::Running;
        Ok(())
    }

    /// Mark this worker as completed.
    pub fn complete(&mut self) -> Result<()> {
        if self.status != TrainingStatus::Running {
            return Err(SynapseError::DistributedError(format!(
                "Worker rank {} not in Running state",
                self.assignment.rank
            )));
        }
        self.status = TrainingStatus::Completed;
        Ok(())
    }

    /// Mark this worker as failed.
    pub fn fail(&mut self) {
        self.status = TrainingStatus::Failed;
    }

    /// Whether this worker is the coordinator (rank 0).
    pub fn is_coordinator(&self) -> bool {
        self.assignment.rank == 0
    }
}

fn strategy_flag(strategy: DistributedStrategy) -> String {
    match strategy {
        DistributedStrategy::DataParallel => "data_parallel".into(),
        DistributedStrategy::ModelParallel => "model_parallel".into(),
        DistributedStrategy::PipelineParallel => "pipeline_parallel".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_worker(rank: u32) -> DistributedWorker {
        DistributedWorker::new(
            uuid::Uuid::new_v4(),
            WorkerAssignment {
                rank,
                instance_id: format!("node-{rank}"),
                endpoint: format!("http://node-{rank}:9000"),
                device_ids: vec![0],
            },
            DistributedStrategy::DataParallel,
        )
    }

    #[test]
    fn extra_args_format() {
        let worker = test_worker(1);
        let args = worker.extra_args(4);
        assert_eq!(
            args,
            vec![
                "--rank",
                "1",
                "--world-size",
                "4",
                "--strategy",
                "data_parallel"
            ]
        );
    }

    #[test]
    fn lifecycle() {
        let mut worker = test_worker(0);
        assert_eq!(worker.status, TrainingStatus::Queued);
        worker.start().unwrap();
        assert_eq!(worker.status, TrainingStatus::Running);
        worker.complete().unwrap();
        assert_eq!(worker.status, TrainingStatus::Completed);
    }

    #[test]
    fn cannot_start_twice() {
        let mut worker = test_worker(0);
        worker.start().unwrap();
        assert!(worker.start().is_err());
    }

    #[test]
    fn cannot_complete_if_not_running() {
        let mut worker = test_worker(0);
        assert!(worker.complete().is_err());
    }

    #[test]
    fn is_coordinator() {
        assert!(test_worker(0).is_coordinator());
        assert!(!test_worker(1).is_coordinator());
    }

    #[test]
    fn fail_sets_status() {
        let mut worker = test_worker(0);
        worker.fail();
        assert_eq!(worker.status, TrainingStatus::Failed);
    }
}
