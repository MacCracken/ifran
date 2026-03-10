//! Training job status tracking and state transitions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use synapse_types::training::{CheckpointInfo, TrainingJobConfig, TrainingJobId, TrainingStatus};

/// Full state of a training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobState {
    pub id: TrainingJobId,
    pub config: TrainingJobConfig,
    pub status: TrainingStatus,
    pub current_step: u64,
    pub total_steps: u64,
    pub current_epoch: f32,
    pub current_loss: Option<f64>,
    pub checkpoints: Vec<CheckpointInfo>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

impl JobState {
    pub fn new(id: TrainingJobId, config: TrainingJobConfig, total_steps: u64) -> Self {
        Self {
            id,
            config,
            status: TrainingStatus::Queued,
            current_step: 0,
            total_steps,
            current_epoch: 0.0,
            current_loss: None,
            checkpoints: Vec::new(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    pub fn start(&mut self) {
        self.status = TrainingStatus::Running;
        self.started_at = Some(Utc::now());
    }

    pub fn update_progress(&mut self, step: u64, epoch: f32, loss: f64) {
        self.current_step = step;
        self.current_epoch = epoch;
        self.current_loss = Some(loss);
    }

    pub fn add_checkpoint(&mut self, checkpoint: CheckpointInfo) {
        self.checkpoints.push(checkpoint);
    }

    pub fn complete(&mut self) {
        self.status = TrainingStatus::Completed;
        self.completed_at = Some(Utc::now());
    }

    pub fn fail(&mut self, error: String) {
        self.status = TrainingStatus::Failed;
        self.completed_at = Some(Utc::now());
        self.error = Some(error);
    }

    pub fn cancel(&mut self) {
        self.status = TrainingStatus::Cancelled;
        self.completed_at = Some(Utc::now());
    }

    pub fn progress_percent(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            (self.current_step as f64 / self.total_steps as f64) * 100.0
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            TrainingStatus::Completed | TrainingStatus::Failed | TrainingStatus::Cancelled
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_types::training::*;

    fn test_config() -> TrainingJobConfig {
        TrainingJobConfig {
            base_model: "model".into(),
            dataset: DatasetConfig {
                path: "/data.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: None,
            },
            method: TrainingMethod::Lora,
            hyperparams: HyperParams {
                learning_rate: 2e-4,
                epochs: 3,
                batch_size: 4,
                gradient_accumulation_steps: 4,
                warmup_steps: 100,
                weight_decay: 0.01,
                max_seq_length: 2048,
            },
            output_name: None,
            lora: None,
        }
    }

    #[test]
    fn new_job_is_queued() {
        let job = JobState::new(uuid::Uuid::new_v4(), test_config(), 1000);
        assert_eq!(job.status, TrainingStatus::Queued);
        assert!(!job.is_terminal());
        assert!(job.started_at.is_none());
    }

    #[test]
    fn start_sets_running() {
        let mut job = JobState::new(uuid::Uuid::new_v4(), test_config(), 1000);
        job.start();
        assert_eq!(job.status, TrainingStatus::Running);
        assert!(job.started_at.is_some());
        assert!(!job.is_terminal());
    }

    #[test]
    fn update_progress_tracks_step() {
        let mut job = JobState::new(uuid::Uuid::new_v4(), test_config(), 1000);
        job.start();
        job.update_progress(500, 1.5, 0.42);
        assert_eq!(job.current_step, 500);
        assert_eq!(job.current_epoch, 1.5);
        assert_eq!(job.current_loss, Some(0.42));
    }

    #[test]
    fn complete_is_terminal() {
        let mut job = JobState::new(uuid::Uuid::new_v4(), test_config(), 1000);
        job.start();
        job.complete();
        assert_eq!(job.status, TrainingStatus::Completed);
        assert!(job.is_terminal());
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn fail_captures_error() {
        let mut job = JobState::new(uuid::Uuid::new_v4(), test_config(), 100);
        job.start();
        job.fail("OOM".into());
        assert_eq!(job.status, TrainingStatus::Failed);
        assert!(job.is_terminal());
        assert_eq!(job.error, Some("OOM".into()));
    }

    #[test]
    fn cancel_is_terminal() {
        let mut job = JobState::new(uuid::Uuid::new_v4(), test_config(), 100);
        job.cancel();
        assert_eq!(job.status, TrainingStatus::Cancelled);
        assert!(job.is_terminal());
    }

    #[test]
    fn progress_percent_calculation() {
        let mut job = JobState::new(uuid::Uuid::new_v4(), test_config(), 200);
        assert_eq!(job.progress_percent(), 0.0);
        job.update_progress(100, 1.0, 0.5);
        assert_eq!(job.progress_percent(), 50.0);
    }

    #[test]
    fn progress_percent_zero_total() {
        let job = JobState::new(uuid::Uuid::new_v4(), test_config(), 0);
        assert_eq!(job.progress_percent(), 0.0);
    }

    #[test]
    fn add_checkpoint() {
        let mut job = JobState::new(uuid::Uuid::new_v4(), test_config(), 1000);
        let cp = CheckpointInfo {
            step: 500,
            epoch: 1.5,
            loss: 0.3,
            path: "/tmp/cp-500".into(),
            timestamp: Utc::now(),
        };
        job.add_checkpoint(cp);
        assert_eq!(job.checkpoints.len(), 1);
        assert_eq!(job.checkpoints[0].step, 500);
    }
}
