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
        if self.total_steps == 0 { 0.0 }
        else { (self.current_step as f64 / self.total_steps as f64) * 100.0 }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.status, TrainingStatus::Completed | TrainingStatus::Failed | TrainingStatus::Cancelled)
    }
}
