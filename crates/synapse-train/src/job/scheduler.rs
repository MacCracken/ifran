//! Training job scheduler — priority queue with GPU-aware scheduling.

use std::collections::VecDeque;
use synapse_types::training::TrainingJobId;

/// Simple FIFO scheduler. Jobs are started in the order they were created,
/// subject to the max-concurrent limit enforced by JobManager.
pub struct JobScheduler {
    queue: VecDeque<TrainingJobId>,
}

impl JobScheduler {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    /// Enqueue a job.
    pub fn enqueue(&mut self, id: TrainingJobId) {
        self.queue.push_back(id);
    }

    /// Dequeue the next job to run.
    pub fn dequeue(&mut self) -> Option<TrainingJobId> {
        self.queue.pop_front()
    }

    /// Remove a cancelled job from the queue.
    pub fn remove(&mut self, id: &TrainingJobId) {
        self.queue.retain(|j| j != id);
    }

    /// Number of jobs waiting.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Peek at the next job without removing it.
    pub fn peek(&self) -> Option<&TrainingJobId> {
        self.queue.front()
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
    fn enqueue_dequeue_fifo() {
        let mut sched = JobScheduler::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        sched.enqueue(id1);
        sched.enqueue(id2);
        assert_eq!(sched.pending_count(), 2);
        assert_eq!(sched.dequeue(), Some(id1));
        assert_eq!(sched.dequeue(), Some(id2));
        assert_eq!(sched.dequeue(), None);
    }

    #[test]
    fn peek_does_not_remove() {
        let mut sched = JobScheduler::new();
        let id = uuid::Uuid::new_v4();
        sched.enqueue(id);
        assert_eq!(sched.peek(), Some(&id));
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn remove_from_queue() {
        let mut sched = JobScheduler::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        sched.enqueue(id1);
        sched.enqueue(id2);
        sched.remove(&id1);
        assert_eq!(sched.pending_count(), 1);
        assert_eq!(sched.dequeue(), Some(id2));
    }

    #[test]
    fn default_is_empty() {
        let sched = JobScheduler::default();
        assert_eq!(sched.pending_count(), 0);
        assert_eq!(sched.peek(), None);
    }
}
