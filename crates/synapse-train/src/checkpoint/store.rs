//! Checkpoint store for persisting and retrieving model training checkpoints.

use std::path::{Path, PathBuf};
use synapse_types::error::Result;
use synapse_types::training::{CheckpointInfo, TrainingJobId};

/// Manages checkpoint storage on disk.
pub struct CheckpointStore {
    root: PathBuf,
}

impl CheckpointStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Directory for a specific job's checkpoints.
    pub fn job_dir(&self, job_id: TrainingJobId) -> PathBuf {
        self.root.join(job_id.to_string())
    }

    /// Ensure the job's checkpoint directory exists.
    pub fn ensure_dir(&self, job_id: TrainingJobId) -> Result<PathBuf> {
        let dir = self.job_dir(job_id);
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    /// List checkpoints for a job, sorted by step.
    pub fn list(&self, job_id: TrainingJobId) -> Result<Vec<CheckpointInfo>> {
        let dir = self.job_dir(job_id);
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut checkpoints = Vec::new();
        for entry in std::fs::read_dir(&dir)?.flatten() {
            let path = entry.path();
            let meta_path = path.join("checkpoint_meta.json");
            if meta_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&meta_path) {
                    if let Ok(info) = serde_json::from_str::<CheckpointInfo>(&content) {
                        checkpoints.push(info);
                    }
                }
            }
        }

        checkpoints.sort_by_key(|c| c.step);
        Ok(checkpoints)
    }

    /// Get the latest checkpoint for a job.
    pub fn latest(&self, job_id: TrainingJobId) -> Result<Option<CheckpointInfo>> {
        let checkpoints = self.list(job_id)?;
        Ok(checkpoints.into_iter().last())
    }

    /// Prune old checkpoints, keeping only the N most recent.
    pub fn prune(&self, job_id: TrainingJobId, keep: usize) -> Result<usize> {
        let checkpoints = self.list(job_id)?;
        if checkpoints.len() <= keep {
            return Ok(0);
        }

        let to_remove = &checkpoints[..checkpoints.len() - keep];
        let mut removed = 0;
        for cp in to_remove {
            let cp_path = Path::new(&cp.path);
            if cp_path.exists() {
                std::fs::remove_dir_all(cp_path)?;
                removed += 1;
            }
        }

        Ok(removed)
    }
}
