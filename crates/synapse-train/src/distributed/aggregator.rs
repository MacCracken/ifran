//! Checkpoint aggregator — merges worker checkpoints after distributed training.
//!
//! MVP: Simple averaging of model weights from per-worker checkpoint directories.
//! Future: Federated averaging with weighted contributions, gradient compression.

use std::path::{Path, PathBuf};

use synapse_types::SynapseError;
use synapse_types::error::Result;

/// Aggregation strategy for merging distributed checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    /// Simple average of all worker checkpoints (default for data-parallel).
    Average,
    /// Weighted average based on samples-per-worker.
    WeightedAverage,
}

/// Plan for aggregating checkpoints from distributed workers.
#[derive(Debug, Clone)]
pub struct AggregationPlan {
    pub method: AggregationMethod,
    /// Paths to per-worker checkpoint directories.
    pub worker_checkpoint_dirs: Vec<PathBuf>,
    /// Output directory for the merged checkpoint.
    pub output_dir: PathBuf,
}

impl AggregationPlan {
    /// Create a new aggregation plan.
    pub fn new(
        method: AggregationMethod,
        worker_checkpoint_dirs: Vec<PathBuf>,
        output_dir: PathBuf,
    ) -> Result<Self> {
        if worker_checkpoint_dirs.is_empty() {
            return Err(SynapseError::DistributedError(
                "No worker checkpoints to aggregate".into(),
            ));
        }
        Ok(Self {
            method,
            worker_checkpoint_dirs,
            output_dir,
        })
    }

    /// Build a shell command to run the aggregation script.
    ///
    /// MVP: Generates args for a Python aggregation script.
    /// The actual weight-averaging logic lives in Python (using torch).
    pub fn build_command(&self) -> Vec<String> {
        let mut cmd = vec![
            "python3".into(),
            "-m".into(),
            "synapse_scripts.aggregate_checkpoints".into(),
            "--method".into(),
            match self.method {
                AggregationMethod::Average => "average".into(),
                AggregationMethod::WeightedAverage => "weighted_average".into(),
            },
            "--output-dir".into(),
            self.output_dir.to_string_lossy().into_owned(),
        ];

        for dir in &self.worker_checkpoint_dirs {
            cmd.push("--checkpoint-dir".into());
            cmd.push(dir.to_string_lossy().into_owned());
        }

        cmd
    }
}

/// Build the expected checkpoint path for a given worker rank.
pub fn worker_checkpoint_dir(base_output_dir: &Path, rank: u32) -> PathBuf {
    base_output_dir.join(format!("worker-{rank}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_requires_checkpoints() {
        let result = AggregationPlan::new(
            AggregationMethod::Average,
            vec![],
            PathBuf::from("/tmp/merged"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_command_format() {
        let plan = AggregationPlan::new(
            AggregationMethod::Average,
            vec![PathBuf::from("/tmp/w0"), PathBuf::from("/tmp/w1")],
            PathBuf::from("/tmp/merged"),
        )
        .unwrap();

        let cmd = plan.build_command();
        assert_eq!(cmd[0], "python3");
        assert!(cmd.contains(&"--method".to_string()));
        assert!(cmd.contains(&"average".to_string()));
        assert_eq!(cmd.iter().filter(|a| *a == "--checkpoint-dir").count(), 2);
    }

    #[test]
    fn worker_checkpoint_dir_format() {
        let dir = worker_checkpoint_dir(Path::new("/output"), 3);
        assert_eq!(dir, PathBuf::from("/output/worker-3"));
    }

    #[test]
    fn weighted_average_command() {
        let plan = AggregationPlan::new(
            AggregationMethod::WeightedAverage,
            vec![PathBuf::from("/tmp/w0")],
            PathBuf::from("/tmp/out"),
        )
        .unwrap();

        let cmd = plan.build_command();
        assert!(cmd.contains(&"weighted_average".to_string()));
    }
}
