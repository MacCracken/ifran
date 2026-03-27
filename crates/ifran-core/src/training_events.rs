//! Local training event bus.
//!
//! Provides observability for training job lifecycle without requiring
//! SecureYeoman. Events are broadcast via tokio channels to any local
//! subscriber (dashboards, monitoring, fleet peers, etc.).

use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::sync::broadcast;

/// A training lifecycle event.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TrainingEvent {
    /// Job has started running.
    JobStarted {
        job_id: String,
        model: String,
        timestamp: DateTime<Utc>,
    },
    /// Progress update (step completed).
    Progress {
        job_id: String,
        status: String,
        step: u64,
        loss: f64,
        timestamp: DateTime<Utc>,
    },
    /// Job was cancelled.
    JobCancelled {
        job_id: String,
        timestamp: DateTime<Utc>,
    },
    /// Job completed successfully.
    JobCompleted {
        job_id: String,
        timestamp: DateTime<Utc>,
    },
    /// Job failed.
    JobFailed {
        job_id: String,
        error: String,
        timestamp: DateTime<Utc>,
    },
    /// Worker assigned in a distributed job.
    WorkerAssigned {
        job_id: String,
        rank: u32,
        instance_id: String,
        endpoint: String,
        timestamp: DateTime<Utc>,
    },
    /// Checkpoint ready for synchronization.
    CheckpointReady {
        job_id: String,
        rank: u32,
        path: String,
        timestamp: DateTime<Utc>,
    },
}

/// Broadcast bus for training lifecycle events.
pub struct TrainingEventBus {
    sender: broadcast::Sender<TrainingEvent>,
}

impl TrainingEventBus {
    /// Create a new training event bus.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Subscribe to training events.
    pub fn subscribe(&self) -> broadcast::Receiver<TrainingEvent> {
        self.sender.subscribe()
    }

    /// Emit a training event.
    pub fn emit(&self, event: TrainingEvent) {
        let _ = self.sender.send(event);
    }

    /// Convenience: emit a progress event.
    pub fn report_progress(&self, job_id: &str, status: &str, step: u64, loss: f64) {
        self.emit(TrainingEvent::Progress {
            job_id: job_id.to_string(),
            status: status.to_string(),
            step,
            loss,
            timestamp: Utc::now(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_without_subscribers() {
        let bus = TrainingEventBus::new(16);
        // Should not panic
        bus.report_progress("job-1", "running", 10, 0.5);
    }

    #[test]
    fn subscribe_and_receive() {
        let bus = TrainingEventBus::new(16);
        let mut rx = bus.subscribe();

        bus.emit(TrainingEvent::JobStarted {
            job_id: "job-1".into(),
            model: "llama-7b".into(),
            timestamp: Utc::now(),
        });

        let event = rx.try_recv().unwrap();
        match event {
            TrainingEvent::JobStarted { job_id, model, .. } => {
                assert_eq!(job_id, "job-1");
                assert_eq!(model, "llama-7b");
            }
            _ => panic!("Expected JobStarted"),
        }
    }

    #[test]
    fn multiple_subscribers() {
        let bus = TrainingEventBus::new(16);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        bus.report_progress("job-1", "running", 5, 0.3);

        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());
    }

    #[test]
    fn event_serializes() {
        let event = TrainingEvent::JobCompleted {
            job_id: "abc".into(),
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("job_completed"));
        assert!(json.contains("abc"));
    }

    #[test]
    fn all_event_variants_serialize() {
        let events = vec![
            TrainingEvent::JobStarted {
                job_id: "1".into(),
                model: "m".into(),
                timestamp: Utc::now(),
            },
            TrainingEvent::Progress {
                job_id: "1".into(),
                status: "running".into(),
                step: 0,
                loss: 0.0,
                timestamp: Utc::now(),
            },
            TrainingEvent::JobCancelled {
                job_id: "1".into(),
                timestamp: Utc::now(),
            },
            TrainingEvent::JobCompleted {
                job_id: "1".into(),
                timestamp: Utc::now(),
            },
            TrainingEvent::JobFailed {
                job_id: "1".into(),
                error: "oom".into(),
                timestamp: Utc::now(),
            },
            TrainingEvent::WorkerAssigned {
                job_id: "1".into(),
                rank: 0,
                instance_id: "n1".into(),
                endpoint: "http://n1:8420".into(),
                timestamp: Utc::now(),
            },
            TrainingEvent::CheckpointReady {
                job_id: "1".into(),
                rank: 0,
                path: "/tmp/ckpt".into(),
                timestamp: Utc::now(),
            },
        ];
        for e in events {
            let json = serde_json::to_string(&e).unwrap();
            assert!(json.contains("type"));
        }
    }
}
