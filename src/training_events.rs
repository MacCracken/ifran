//! Local training event bus.
//!
//! Provides observability for training job lifecycle without requiring
//! SecureYeoman. Events are broadcast via tokio channels to any local
//! subscriber (dashboards, monitoring, fleet peers, etc.).

use std::sync::Arc;

use chrono::{DateTime, Utc};
use majra::namespace::Namespace;
use majra::pubsub::{PubSub, TypedMessage, TypedPubSub, TypedPubSubConfig};
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
    pubsub: TypedPubSub<TrainingEvent>,
    event_hub: Option<Arc<PubSub>>,
}

impl TrainingEventBus {
    /// Create a new training event bus.
    pub fn new(capacity: usize) -> Self {
        let pubsub = TypedPubSub::with_config(TypedPubSubConfig {
            channel_capacity: capacity,
            ..Default::default()
        });
        Self {
            pubsub,
            event_hub: None,
        }
    }

    pub fn with_hub(capacity: usize, hub: Arc<PubSub>) -> Self {
        let pubsub = TypedPubSub::with_config(TypedPubSubConfig {
            channel_capacity: capacity,
            ..Default::default()
        });
        Self {
            pubsub,
            event_hub: Some(hub),
        }
    }

    /// Subscribe to training events.
    pub fn subscribe(&self) -> broadcast::Receiver<TypedMessage<TrainingEvent>> {
        self.pubsub.subscribe("training/#")
    }

    /// Emit a training event.
    pub fn emit(&self, event: TrainingEvent) {
        #[allow(unreachable_patterns)]
        let topic = match &event {
            TrainingEvent::JobStarted { .. } => "training/started",
            TrainingEvent::Progress { .. } => "training/progress",
            TrainingEvent::JobCompleted { .. } => "training/completed",
            TrainingEvent::JobFailed { .. } => "training/failed",
            TrainingEvent::JobCancelled { .. } => "training/cancelled",
            TrainingEvent::WorkerAssigned { .. } => "training/worker_assigned",
            TrainingEvent::CheckpointReady { .. } => "training/checkpoint",
            _ => "training/other",
        };
        if let Some(hub) = &self.event_hub {
            if let Ok(json) = serde_json::to_value(&event) {
                hub.publish(topic, json);
            }
        }
        self.pubsub.publish(topic, event);
    }

    pub fn subscribe_namespaced(
        &self,
        namespace: &Namespace,
    ) -> broadcast::Receiver<TypedMessage<TrainingEvent>> {
        self.pubsub.subscribe(&namespace.pattern("training/#"))
    }

    #[allow(unreachable_patterns)]
    pub fn emit_namespaced(&self, event: TrainingEvent, namespace: &Namespace) {
        let topic = match &event {
            TrainingEvent::JobStarted { .. } => namespace.topic("training/started"),
            TrainingEvent::Progress { .. } => namespace.topic("training/progress"),
            TrainingEvent::JobCompleted { .. } => namespace.topic("training/completed"),
            TrainingEvent::JobFailed { .. } => namespace.topic("training/failed"),
            TrainingEvent::JobCancelled { .. } => namespace.topic("training/cancelled"),
            TrainingEvent::WorkerAssigned { .. } => namespace.topic("training/worker_assigned"),
            TrainingEvent::CheckpointReady { .. } => namespace.topic("training/checkpoint"),
            _ => namespace.topic("training/other"),
        };
        if let Some(hub) = &self.event_hub {
            if let Ok(json) = serde_json::to_value(&event) {
                hub.publish(&topic, json);
            }
        }
        self.pubsub.publish(&topic, event);
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

        let msg = rx.try_recv().unwrap();
        match msg.payload {
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
    fn namespaced_subscribe_isolates_tenants() {
        let bus = TrainingEventBus::new(16);
        let ns_a = Namespace::new("tenant-a");
        let ns_b = Namespace::new("tenant-b");

        let mut rx_a = bus.subscribe_namespaced(&ns_a);
        let mut rx_b = bus.subscribe_namespaced(&ns_b);

        bus.emit_namespaced(
            TrainingEvent::JobStarted {
                job_id: "job-1".into(),
                model: "llama-7b".into(),
                timestamp: Utc::now(),
            },
            &ns_a,
        );

        assert!(rx_a.try_recv().is_ok());
        assert!(rx_b.try_recv().is_err());
    }

    #[test]
    fn namespaced_emit_all_variants() {
        let bus = TrainingEventBus::new(16);
        let ns = Namespace::new("org-1");
        let mut rx = bus.subscribe_namespaced(&ns);

        bus.emit_namespaced(
            TrainingEvent::Progress {
                job_id: "j".into(),
                status: "running".into(),
                step: 1,
                loss: 0.5,
                timestamp: Utc::now(),
            },
            &ns,
        );

        let msg = rx.try_recv().unwrap();
        assert!(msg.topic.starts_with("org-1/"));
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
