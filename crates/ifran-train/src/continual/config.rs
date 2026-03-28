//! Configuration for continual/online learning.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Configuration for a continual learning session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualConfig {
    pub base_model: String,
    pub data_source: String,
    /// Max samples to keep in replay buffer.
    #[serde(default = "default_replay_size")]
    pub replay_buffer_size: usize,
    /// Fraction of replay buffer to mix with new data per update.
    #[serde(default = "default_replay_fraction")]
    pub replay_fraction: f64,
    /// Number of gradient accumulation steps per update.
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: u32,
    /// Learning rate for updates.
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    /// Max new samples before triggering an update.
    #[serde(default = "default_update_threshold")]
    pub update_threshold: usize,
}

fn default_replay_size() -> usize {
    10000
}
fn default_replay_fraction() -> f64 {
    0.2
}
fn default_grad_accum() -> u32 {
    4
}
fn default_lr() -> f64 {
    1e-5
}
fn default_update_threshold() -> usize {
    100
}

/// A replay buffer that maintains a sliding window of recent training samples.
///
/// Uses `VecDeque` for O(1) eviction from the front.
pub struct ReplayBuffer {
    samples: VecDeque<String>,
    capacity: usize,
}

impl ReplayBuffer {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.min(1_000_000);
        Self {
            samples: VecDeque::with_capacity(cap),
            capacity,
        }
    }

    pub fn add(&mut self, sample: String) {
        if self.capacity == 0 {
            return;
        }
        if self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    #[must_use]
    pub fn sample(&self, count: usize) -> Vec<&str> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        if self.samples.is_empty() || count == 0 {
            return Vec::new();
        }

        let actual = count.min(self.samples.len());
        let mut result = Vec::with_capacity(actual);
        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now().hash(&mut hasher);

        for i in 0..actual {
            hasher.write_usize(i);
            let idx = (hasher.finish() as usize) % self.samples.len();
            result.push(self.samples[idx].as_str());
        }
        result
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let json = r#"{"base_model": "llama", "data_source": "/data"}"#;
        let config: ContinualConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.replay_buffer_size, 10000);
        assert_eq!(config.replay_fraction, 0.2);
        assert_eq!(config.update_threshold, 100);
    }

    #[test]
    fn replay_buffer_add_and_len() {
        let mut buf = ReplayBuffer::new(5);
        assert!(buf.is_empty());
        buf.add("s1".into());
        buf.add("s2".into());
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn replay_buffer_capacity() {
        let mut buf = ReplayBuffer::new(3);
        buf.add("s1".into());
        buf.add("s2".into());
        buf.add("s3".into());
        buf.add("s4".into()); // should evict s1
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn replay_buffer_sample() {
        let mut buf = ReplayBuffer::new(100);
        for i in 0..50 {
            buf.add(format!("sample-{i}"));
        }
        let samples = buf.sample(5);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn replay_buffer_sample_empty() {
        let buf = ReplayBuffer::new(10);
        assert!(buf.sample(5).is_empty());
    }

    #[test]
    fn replay_buffer_sample_more_than_available() {
        let mut buf = ReplayBuffer::new(10);
        buf.add("a".into());
        buf.add("b".into());
        let samples = buf.sample(10);
        assert_eq!(samples.len(), 2); // can't sample more than available
    }

    #[test]
    fn config_custom_values() {
        let json = r#"{
            "base_model": "mistral-7b",
            "data_source": "/datasets/custom",
            "replay_buffer_size": 5000,
            "replay_fraction": 0.5,
            "gradient_accumulation_steps": 8,
            "learning_rate": 3e-4,
            "update_threshold": 50
        }"#;
        let config: ContinualConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.base_model, "mistral-7b");
        assert_eq!(config.data_source, "/datasets/custom");
        assert_eq!(config.replay_buffer_size, 5000);
        assert_eq!(config.replay_fraction, 0.5);
        assert_eq!(config.gradient_accumulation_steps, 8);
        assert!((config.learning_rate - 3e-4).abs() < f64::EPSILON);
        assert_eq!(config.update_threshold, 50);
    }

    #[test]
    fn buffer_capacity_zero() {
        let mut buf = ReplayBuffer::new(0);
        assert_eq!(buf.capacity(), 0);
        assert!(buf.is_empty());
        // Adding to a zero-capacity buffer should be a no-op, not panic.
        buf.add("should be ignored".into());
        assert!(buf.is_empty());
        assert!(buf.sample(5).is_empty());
    }
}
