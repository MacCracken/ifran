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
