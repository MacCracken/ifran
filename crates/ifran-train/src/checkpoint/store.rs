//! Checkpoint store for persisting and retrieving model training checkpoints.

use ifran_types::error::Result;
use ifran_types::training::{CheckpointInfo, TrainingJobId};
use std::path::{Path, PathBuf};

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
            match std::fs::remove_dir_all(cp_path) {
                Ok(()) => removed += 1,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // Already deleted — skip silently
                }
                Err(e) => return Err(e.into()),
            }
        }

        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ifran_types::training::CheckpointInfo;

    #[test]
    fn job_dir_uses_id() {
        let store = CheckpointStore::new("/tmp/checkpoints");
        let id = uuid::Uuid::new_v4();
        let dir = store.job_dir(id);
        assert_eq!(dir, PathBuf::from(format!("/tmp/checkpoints/{id}")));
    }

    #[test]
    fn ensure_dir_creates_path() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CheckpointStore::new(tmp.path());
        let id = uuid::Uuid::new_v4();
        let dir = store.ensure_dir(id).unwrap();
        assert!(dir.exists());
    }

    #[test]
    fn list_empty_for_nonexistent_job() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CheckpointStore::new(tmp.path());
        let id = uuid::Uuid::new_v4();
        let cps = store.list(id).unwrap();
        assert!(cps.is_empty());
    }

    #[test]
    fn list_reads_checkpoint_meta() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CheckpointStore::new(tmp.path());
        let job_id = uuid::Uuid::new_v4();
        let job_dir = store.ensure_dir(job_id).unwrap();

        // Create a checkpoint directory with metadata
        let cp_dir = job_dir.join("step-100");
        std::fs::create_dir(&cp_dir).unwrap();
        let info = CheckpointInfo {
            step: 100,
            epoch: 1.0,
            loss: 0.5,
            path: cp_dir.to_string_lossy().to_string(),
            timestamp: chrono::Utc::now(),
        };
        let meta = serde_json::to_string(&info).unwrap();
        std::fs::write(cp_dir.join("checkpoint_meta.json"), meta).unwrap();

        let cps = store.list(job_id).unwrap();
        assert_eq!(cps.len(), 1);
        assert_eq!(cps[0].step, 100);
    }

    #[test]
    fn latest_returns_highest_step() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CheckpointStore::new(tmp.path());
        let job_id = uuid::Uuid::new_v4();
        let job_dir = store.ensure_dir(job_id).unwrap();

        for step in [50, 200, 100] {
            let cp_dir = job_dir.join(format!("step-{step}"));
            std::fs::create_dir(&cp_dir).unwrap();
            let info = CheckpointInfo {
                step,
                epoch: 1.0,
                loss: 0.5,
                path: cp_dir.to_string_lossy().to_string(),
                timestamp: chrono::Utc::now(),
            };
            std::fs::write(
                cp_dir.join("checkpoint_meta.json"),
                serde_json::to_string(&info).unwrap(),
            )
            .unwrap();
        }

        let latest = store.latest(job_id).unwrap().unwrap();
        assert_eq!(latest.step, 200);
    }

    #[test]
    fn prune_keeps_n_most_recent() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CheckpointStore::new(tmp.path());
        let job_id = uuid::Uuid::new_v4();
        let job_dir = store.ensure_dir(job_id).unwrap();

        for step in [100, 200, 300] {
            let cp_dir = job_dir.join(format!("step-{step}"));
            std::fs::create_dir(&cp_dir).unwrap();
            let info = CheckpointInfo {
                step,
                epoch: 1.0,
                loss: 0.5,
                path: cp_dir.to_string_lossy().to_string(),
                timestamp: chrono::Utc::now(),
            };
            std::fs::write(
                cp_dir.join("checkpoint_meta.json"),
                serde_json::to_string(&info).unwrap(),
            )
            .unwrap();
        }

        let removed = store.prune(job_id, 1).unwrap();
        assert_eq!(removed, 2);
        let remaining = store.list(job_id).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].step, 300);
    }

    #[test]
    fn prune_noop_when_under_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CheckpointStore::new(tmp.path());
        let job_id = uuid::Uuid::new_v4();
        let removed = store.prune(job_id, 5).unwrap();
        assert_eq!(removed, 0);
    }
}
