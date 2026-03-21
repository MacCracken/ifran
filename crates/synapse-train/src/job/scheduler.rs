//! Training job scheduler — priority queue with GPU-aware scheduling.
//!
//! Uses majra's multi-tier priority queue under the hood. Jobs are dequeued
//! highest-priority-first, FIFO within each priority tier.

use std::collections::HashSet;
use synapse_types::training::TrainingJobId;

pub use majra::queue::Priority;
use majra::queue::{PriorityQueue, QueueItem};

/// Priority-based job scheduler backed by majra.
///
/// Jobs cancelled via `remove()` are tracked in a skip set and filtered
/// on dequeue, since the underlying queue does not support random removal.
pub struct JobScheduler {
    queue: PriorityQueue<TrainingJobId>,
    cancelled: HashSet<TrainingJobId>,
}

impl JobScheduler {
    pub fn new() -> Self {
        Self {
            queue: PriorityQueue::new(),
            cancelled: HashSet::new(),
        }
    }

    /// Enqueue a job at the given priority.
    pub fn enqueue(&mut self, id: TrainingJobId, priority: Priority) {
        self.cancelled.remove(&id);
        self.queue.enqueue(QueueItem::new(priority, id));
    }

    /// Dequeue the highest-priority job, skipping cancelled entries.
    pub fn dequeue(&mut self) -> Option<TrainingJobId> {
        while let Some(item) = self.queue.dequeue() {
            if !self.cancelled.remove(&item.payload) {
                return Some(item.payload);
            }
        }
        None
    }

    /// Mark a job as cancelled so it is skipped on dequeue.
    pub fn remove(&mut self, id: &TrainingJobId) {
        self.cancelled.insert(*id);
    }

    /// Number of jobs waiting (approximate — includes cancelled entries
    /// not yet drained).
    pub fn pending_count(&self) -> usize {
        self.queue.len().saturating_sub(self.cancelled.len())
    }
}

impl Default for JobScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enqueue_dequeue_fifo_within_priority() {
        let mut sched = JobScheduler::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        sched.enqueue(id1, Priority::Normal);
        sched.enqueue(id2, Priority::Normal);
        assert_eq!(sched.pending_count(), 2);
        assert_eq!(sched.dequeue(), Some(id1));
        assert_eq!(sched.dequeue(), Some(id2));
        assert_eq!(sched.dequeue(), None);
    }

    #[test]
    fn remove_from_queue() {
        let mut sched = JobScheduler::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        sched.enqueue(id1, Priority::Normal);
        sched.enqueue(id2, Priority::Normal);
        sched.remove(&id1);
        assert_eq!(sched.pending_count(), 1);
        assert_eq!(sched.dequeue(), Some(id2));
    }

    #[test]
    fn default_is_empty() {
        let sched = JobScheduler::default();
        assert_eq!(sched.pending_count(), 0);
    }

    #[test]
    fn priority_ordering() {
        let mut sched = JobScheduler::new();
        let low = uuid::Uuid::new_v4();
        let high = uuid::Uuid::new_v4();
        let critical = uuid::Uuid::new_v4();
        sched.enqueue(low, Priority::Low);
        sched.enqueue(high, Priority::High);
        sched.enqueue(critical, Priority::Critical);
        assert_eq!(sched.dequeue(), Some(critical));
        assert_eq!(sched.dequeue(), Some(high));
        assert_eq!(sched.dequeue(), Some(low));
        assert_eq!(sched.dequeue(), None);
    }

    #[test]
    fn mixed_priority_with_cancel() {
        let mut sched = JobScheduler::new();
        let bg = uuid::Uuid::new_v4();
        let normal = uuid::Uuid::new_v4();
        let high = uuid::Uuid::new_v4();
        sched.enqueue(bg, Priority::Background);
        sched.enqueue(normal, Priority::Normal);
        sched.enqueue(high, Priority::High);
        sched.remove(&high);
        assert_eq!(sched.dequeue(), Some(normal));
        assert_eq!(sched.dequeue(), Some(bg));
        assert_eq!(sched.dequeue(), None);
    }
}
