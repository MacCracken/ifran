//! GPU allocation event streaming.
//!
//! Broadcasts GPU allocation and release events via a tokio broadcast channel,
//! enabling observability and integration with external orchestrators.

use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::sync::broadcast;

use super::allocator::AllocationId;

/// A GPU lifecycle event.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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
    sender: broadcast::Sender<GpuEvent>,
}

impl GpuEventBus {
    /// Create a new event bus with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Subscribe to GPU events.
    pub fn subscribe(&self) -> broadcast::Receiver<GpuEvent> {
        self.sender.subscribe()
    }

    /// Emit a GPU event. Returns the number of receivers that got it.
    pub fn emit(&self, event: GpuEvent) -> usize {
        self.sender.send(event).unwrap_or(0)
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

        let event = rx.try_recv().unwrap();
        match event {
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
