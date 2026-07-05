//! GPU allocation event streaming.
//!
//! Broadcasts GPU allocation and release events via a tokio broadcast channel,
//! enabling observability and integration with external orchestrators.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use majra::namespace::Namespace;
use majra::pubsub::{PubSub, TypedMessage, TypedPubSub, TypedPubSubConfig};
use serde::Serialize;
use tokio::sync::broadcast;

use super::allocator::AllocationId;

/// A GPU lifecycle event.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum GpuEvent {
    /// GPU devices were allocated to a job.
    Allocated {
        allocation_id: String,
        device_ids: Vec<u32>,
        memory_mb: u64,
        job_label: String,
        timestamp: DateTime<Utc>,
    },
    /// GPU devices were released from a job.
    Released {
        allocation_id: String,
        device_ids: Vec<u32>,
        timestamp: DateTime<Utc>,
    },
}

/// Broadcast bus for GPU lifecycle events.
pub struct GpuEventBus {
    pubsub: TypedPubSub<GpuEvent>,
    event_hub: Option<Arc<PubSub>>,
}

impl GpuEventBus {
    /// Create a new event bus with the given channel capacity.
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

    /// Subscribe to GPU events.
    pub fn subscribe(&self) -> broadcast::Receiver<TypedMessage<GpuEvent>> {
        self.pubsub.subscribe("gpu/#")
    }

    /// Emit a GPU event. Returns the number of receivers that got it.
    pub fn emit(&self, event: GpuEvent) -> usize {
        let topic = match &event {
            GpuEvent::Allocated { .. } => "gpu/allocated",
            GpuEvent::Released { .. } => "gpu/released",
        };
        if let Some(hub) = &self.event_hub {
            if let Ok(json) = serde_json::to_value(&event) {
                hub.publish(topic, json);
            }
        }
        self.pubsub.publish(topic, event)
    }

    pub fn subscribe_namespaced(
        &self,
        namespace: &Namespace,
    ) -> broadcast::Receiver<TypedMessage<GpuEvent>> {
        self.pubsub.subscribe(&namespace.pattern("gpu/#"))
    }

    pub fn emit_namespaced(&self, event: GpuEvent, namespace: &Namespace) -> usize {
        let topic = match &event {
            GpuEvent::Allocated { .. } => namespace.topic("gpu/allocated"),
            GpuEvent::Released { .. } => namespace.topic("gpu/released"),
        };
        if let Some(hub) = &self.event_hub {
            if let Ok(json) = serde_json::to_value(&event) {
                hub.publish(&topic, json);
            }
        }
        self.pubsub.publish(&topic, event)
    }

    /// Emit an allocation event.
    pub fn emit_allocated(
        &self,
        allocation_id: AllocationId,
        device_ids: Vec<u32>,
        memory_mb: u64,
        job_label: &str,
    ) {
        self.emit(GpuEvent::Allocated {
            allocation_id: allocation_id.to_string(),
            device_ids,
            memory_mb,
            job_label: job_label.to_string(),
            timestamp: Utc::now(),
        });
    }

    /// Emit a release event.
    pub fn emit_released(&self, allocation_id: AllocationId, device_ids: Vec<u32>) {
        self.emit(GpuEvent::Released {
            allocation_id: allocation_id.to_string(),
            device_ids,
            timestamp: Utc::now(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_bus_emit_no_receivers() {
        let bus = GpuEventBus::new(16);
        let count = bus.emit(GpuEvent::Released {
            allocation_id: "test".into(),
            device_ids: vec![0],
            timestamp: Utc::now(),
        });
        assert_eq!(count, 0);
    }

    #[test]
    fn event_bus_subscribe_and_receive() {
        let bus = GpuEventBus::new(16);
        let mut rx = bus.subscribe();

        bus.emit_allocated(uuid::Uuid::new_v4(), vec![0, 1], 8192, "test-job");

        let msg = rx.try_recv().unwrap();
        match msg.payload {
            GpuEvent::Allocated {
                device_ids,
                memory_mb,
                job_label,
                ..
            } => {
                assert_eq!(device_ids, vec![0, 1]);
                assert_eq!(memory_mb, 8192);
                assert_eq!(job_label, "test-job");
            }
            _ => panic!("Expected Allocated event"),
        }
    }

    #[test]
    fn event_bus_multiple_subscribers() {
        let bus = GpuEventBus::new(16);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        bus.emit_released(uuid::Uuid::new_v4(), vec![0]);

        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());
    }

    #[test]
    fn namespaced_subscribe_isolates_tenants() {
        let bus = GpuEventBus::new(16);
        let ns_a = Namespace::new("tenant-a");
        let ns_b = Namespace::new("tenant-b");

        let mut rx_a = bus.subscribe_namespaced(&ns_a);
        let mut rx_b = bus.subscribe_namespaced(&ns_b);

        bus.emit_namespaced(
            GpuEvent::Allocated {
                allocation_id: "alloc-1".into(),
                device_ids: vec![0],
                memory_mb: 8192,
                job_label: "train".into(),
                timestamp: Utc::now(),
            },
            &ns_a,
        );

        assert!(rx_a.try_recv().is_ok());
        assert!(rx_b.try_recv().is_err());
    }

    #[test]
    fn namespaced_emit_returns_receiver_count() {
        let bus = GpuEventBus::new(16);
        let ns = Namespace::new("org-1");
        let _rx = bus.subscribe_namespaced(&ns);

        let count = bus.emit_namespaced(
            GpuEvent::Released {
                allocation_id: "alloc-1".into(),
                device_ids: vec![0],
                timestamp: Utc::now(),
            },
            &ns,
        );
        assert_eq!(count, 1);
    }

    #[test]
    fn gpu_event_serializes() {
        let event = GpuEvent::Allocated {
            allocation_id: "abc-123".into(),
            device_ids: vec![0],
            memory_mb: 4096,
            job_label: "inference".into(),
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("allocated"));
        assert!(json.contains("inference"));
    }
}
