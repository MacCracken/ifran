//! Backend health tracking with ring buffer and automatic failover.
//!
//! Each backend gets a health ring that tracks the last N request outcomes.
//! When failure rate exceeds a threshold, the backend is marked unhealthy.

use std::collections::HashMap;
use std::fmt;

use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Health status of a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => f.write_str("Healthy"),
            Self::Degraded => f.write_str("Degraded"),
            Self::Unhealthy => f.write_str("Unhealthy"),
        }
    }
}

/// Configuration for health tracking.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Number of recent results to track per backend.
    pub ring_size: usize,
    /// Failure rate threshold to mark as degraded (0.0-1.0).
    pub degraded_threshold: f64,
    /// Failure rate threshold to mark as unhealthy (0.0-1.0).
    pub unhealthy_threshold: f64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            ring_size: 20,
            degraded_threshold: 0.3,
            unhealthy_threshold: 0.6,
        }
    }
}

/// Ring buffer tracking recent request outcomes.
struct HealthRing {
    /// `true` = success, `false` = failure.
    outcomes: Vec<bool>,
    cursor: usize,
    filled: bool,
    config: HealthConfig,
}

impl HealthRing {
    #[inline]
    fn new(config: HealthConfig) -> Self {
        let ring_size = config.ring_size.max(1);
        Self {
            outcomes: vec![true; ring_size],
            cursor: 0,
            filled: false,
            config,
        }
    }

    #[inline]
    fn record(&mut self, success: bool) {
        self.outcomes[self.cursor] = success;
        self.cursor += 1;
        if self.cursor >= self.outcomes.len() {
            self.cursor = 0;
            self.filled = true;
        }
    }

    #[inline]
    #[must_use]
    fn failure_rate(&self) -> f64 {
        let count = if self.filled {
            self.outcomes.len()
        } else if self.cursor == 0 {
            return 0.0;
        } else {
            self.cursor
        };

        let failures = self.outcomes[..count].iter().filter(|&&ok| !ok).count();
        failures as f64 / count as f64
    }

    #[inline]
    #[must_use]
    fn status(&self) -> HealthStatus {
        let rate = self.failure_rate();
        if rate >= self.config.unhealthy_threshold {
            HealthStatus::Unhealthy
        } else if rate >= self.config.degraded_threshold {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }
}

/// Tracks health across all backends.
pub struct BackendHealthTracker {
    rings: RwLock<HashMap<String, HealthRing>>,
    config: HealthConfig,
}

impl BackendHealthTracker {
    /// Create a new tracker with the given configuration.
    #[must_use]
    pub fn new(config: HealthConfig) -> Self {
        Self {
            rings: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Record a request outcome for a backend.
    pub async fn record(&self, backend_id: &str, success: bool) {
        let mut rings = self.rings.write().await;
        let ring = rings
            .entry(backend_id.to_owned())
            .or_insert_with(|| HealthRing::new(self.config.clone()));
        ring.record(success);

        let status = ring.status();
        match status {
            HealthStatus::Unhealthy => {
                warn!(
                    backend = backend_id,
                    failure_rate = ring.failure_rate(),
                    "backend marked unhealthy"
                );
            }
            HealthStatus::Degraded => {
                debug!(
                    backend = backend_id,
                    failure_rate = ring.failure_rate(),
                    "backend degraded"
                );
            }
            HealthStatus::Healthy => {}
        }
    }

    /// Get the health status of a backend.
    #[must_use]
    pub async fn status(&self, backend_id: &str) -> HealthStatus {
        let rings = self.rings.read().await;
        rings
            .get(backend_id)
            .map_or(HealthStatus::Healthy, |ring| ring.status())
    }

    /// Get the failure rate for a backend.
    #[must_use]
    pub async fn failure_rate(&self, backend_id: &str) -> f64 {
        let rings = self.rings.read().await;
        rings
            .get(backend_id)
            .map_or(0.0, |ring| ring.failure_rate())
    }

    /// Check if a backend is available for requests.
    #[must_use]
    pub async fn is_available(&self, backend_id: &str) -> bool {
        self.status(backend_id).await != HealthStatus::Unhealthy
    }

    /// List all backends with their health status.
    #[must_use]
    pub async fn all_statuses(&self) -> Vec<(String, HealthStatus, f64)> {
        let rings = self.rings.read().await;
        rings
            .iter()
            .map(|(id, ring)| (id.clone(), ring.status(), ring.failure_rate()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> HealthConfig {
        HealthConfig {
            ring_size: 10,
            degraded_threshold: 0.3,
            unhealthy_threshold: 0.6,
        }
    }

    #[test]
    fn new_ring_starts_healthy() {
        let ring = HealthRing::new(default_config());
        assert_eq!(ring.status(), HealthStatus::Healthy);
        assert!((ring.failure_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn failures_degrade_then_unhealthy() {
        let mut ring = HealthRing::new(default_config());
        // 3 failures out of 10-slot ring (cursor=3, count=3) -> 100% fail
        for _ in 0..3 {
            ring.record(false);
        }
        // 3/3 = 1.0 > 0.6 -> unhealthy
        assert_eq!(ring.status(), HealthStatus::Unhealthy);

        // Now add successes to bring it down
        let mut ring2 = HealthRing::new(default_config());
        // 3 fail, 7 success -> 3/10 = 0.3 -> degraded
        for _ in 0..3 {
            ring2.record(false);
        }
        for _ in 0..7 {
            ring2.record(true);
        }
        assert_eq!(ring2.status(), HealthStatus::Degraded);
    }

    #[test]
    fn successes_recover_from_unhealthy() {
        let mut ring = HealthRing::new(default_config());
        // Fill with failures
        for _ in 0..10 {
            ring.record(false);
        }
        assert_eq!(ring.status(), HealthStatus::Unhealthy);

        // Now overwrite with successes (ring wraps)
        for _ in 0..10 {
            ring.record(true);
        }
        assert_eq!(ring.status(), HealthStatus::Healthy);
        assert!((ring.failure_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ring_buffer_wraps_correctly() {
        let config = HealthConfig {
            ring_size: 4,
            degraded_threshold: 0.3,
            unhealthy_threshold: 0.6,
        };
        let mut ring = HealthRing::new(config);

        // Fill: [F, F, S, S] -> 2/4 = 0.5 -> degraded
        ring.record(false);
        ring.record(false);
        ring.record(true);
        ring.record(true);
        assert_eq!(ring.status(), HealthStatus::Degraded);
        assert!(ring.filled);

        // Wrap: overwrites index 0,1 -> [S, S, S, S] -> 0/4 = 0.0
        ring.record(true);
        ring.record(true);
        assert_eq!(ring.status(), HealthStatus::Healthy);
        assert!((ring.failure_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn failure_rate_calculated_correctly() {
        let mut ring = HealthRing::new(HealthConfig {
            ring_size: 5,
            degraded_threshold: 0.3,
            unhealthy_threshold: 0.6,
        });

        // 2 failures out of 4 recorded -> 0.5
        ring.record(true);
        ring.record(false);
        ring.record(true);
        ring.record(false);
        assert!((ring.failure_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn failure_rate_zero_when_no_records() {
        let ring = HealthRing::new(default_config());
        assert!((ring.failure_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn tracker_unknown_backend_is_healthy() {
        let tracker = BackendHealthTracker::new(default_config());
        assert_eq!(tracker.status("unknown").await, HealthStatus::Healthy);
        assert!(tracker.is_available("unknown").await);
        assert!((tracker.failure_rate("unknown").await - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn tracker_is_available_false_for_unhealthy() {
        let tracker = BackendHealthTracker::new(default_config());
        for _ in 0..10 {
            tracker.record("bad-backend", false).await;
        }
        assert!(!tracker.is_available("bad-backend").await);
        assert_eq!(tracker.status("bad-backend").await, HealthStatus::Unhealthy);
    }

    #[tokio::test]
    async fn tracker_multiple_backends_independent() {
        let tracker = BackendHealthTracker::new(default_config());
        for _ in 0..10 {
            tracker.record("healthy-one", true).await;
            tracker.record("sick-one", false).await;
        }
        assert_eq!(tracker.status("healthy-one").await, HealthStatus::Healthy);
        assert_eq!(tracker.status("sick-one").await, HealthStatus::Unhealthy);
    }

    #[tokio::test]
    async fn tracker_all_statuses_lists_backends() {
        let tracker = BackendHealthTracker::new(default_config());
        tracker.record("a", true).await;
        tracker.record("b", false).await;

        let statuses = tracker.all_statuses().await;
        assert_eq!(statuses.len(), 2);

        let ids: Vec<&str> = statuses.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn ring_size_zero_clamped_to_one() {
        let config = HealthConfig {
            ring_size: 0,
            degraded_threshold: 0.3,
            unhealthy_threshold: 0.6,
        };
        let mut ring = HealthRing::new(config);
        // ring_size is clamped to 1, so single failure -> 100% failure rate
        ring.record(false);
        assert!((ring.failure_rate() - 1.0).abs() < f64::EPSILON);
        assert_eq!(ring.status(), HealthStatus::Unhealthy);
    }

    #[test]
    fn ring_partial_fill_only_counts_recorded() {
        let config = HealthConfig {
            ring_size: 10,
            degraded_threshold: 0.3,
            unhealthy_threshold: 0.6,
        };
        let mut ring = HealthRing::new(config);
        // Record only 2 entries: 1 fail, 1 success -> 1/2 = 0.5
        ring.record(false);
        ring.record(true);
        assert!((ring.failure_rate() - 0.5).abs() < f64::EPSILON);
        assert_eq!(ring.status(), HealthStatus::Degraded);
    }

    #[test]
    fn ring_wrapping_overwrites_oldest() {
        let config = HealthConfig {
            ring_size: 3,
            degraded_threshold: 0.3,
            unhealthy_threshold: 0.6,
        };
        let mut ring = HealthRing::new(config);
        // Fill: [F, F, F] -> 3/3 = 1.0 -> unhealthy
        ring.record(false);
        ring.record(false);
        ring.record(false);
        assert_eq!(ring.status(), HealthStatus::Unhealthy);
        assert!(ring.filled);

        // Wrap: overwrite index 0 with S -> [S, F, F] -> 2/3 = 0.67 -> unhealthy
        ring.record(true);
        assert!((ring.failure_rate() - 2.0 / 3.0).abs() < f64::EPSILON);

        // Overwrite index 1 with S -> [S, S, F] -> 1/3 = 0.33 -> degraded
        ring.record(true);
        assert_eq!(ring.status(), HealthStatus::Degraded);

        // Overwrite index 2 with S -> [S, S, S] -> 0/3 = 0.0 -> healthy
        ring.record(true);
        assert_eq!(ring.status(), HealthStatus::Healthy);
    }

    #[test]
    fn health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "Healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "Degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "Unhealthy");
    }
}
