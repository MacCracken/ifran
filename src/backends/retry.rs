//! Retry with exponential backoff and jitter for transient failures.
//!
//! Provides [`RetryConfig`] which calculates per-attempt delays and
//! classifies errors as retryable or permanent.

use std::time::Duration;

/// Configuration for retry behaviour.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 means no retries).
    pub max_retries: u32,
    /// Base delay before the first retry.
    pub base_delay: Duration,
    /// Upper bound on any single delay.
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
        }
    }
}

impl RetryConfig {
    /// Calculate delay for attempt `attempt` (0-indexed) with jitter.
    ///
    /// The delay grows exponentially from [`base_delay`](Self::base_delay),
    /// capped at [`max_delay`](Self::max_delay), with +/-25 % jitter.
    #[must_use]
    #[inline]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base = self.base_delay.as_millis() as u64;
        let exponential = base.saturating_mul(1u64 << attempt.min(10));
        let max = self.max_delay.as_millis() as u64;
        let capped = exponential.min(max);

        // +/-25 % jitter (deterministic based on attempt number)
        let jitter_range = capped / 4;
        let jitter = if jitter_range > 0 {
            (attempt as u64 * 7919) % (jitter_range * 2)
        } else {
            0
        };
        Duration::from_millis(capped.saturating_sub(jitter_range).saturating_add(jitter))
    }

    /// Check if an error message indicates a transient (retryable) failure.
    #[must_use]
    pub fn is_retryable(error: &str) -> bool {
        error.contains("connection refused")
            || error.contains("Connection refused")
            || error.contains("timed out")
            || error.contains("timeout")
            || error.contains("503")
            || error.contains("502")
            || error.contains("429")
            || error.contains("connect error")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sensible() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_retries, 2);
        assert_eq!(cfg.base_delay, Duration::from_millis(100));
        assert_eq!(cfg.max_delay, Duration::from_secs(5));
    }

    #[test]
    fn delay_increases_exponentially() {
        let cfg = RetryConfig {
            max_retries: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
        };
        let d0 = cfg.delay_for_attempt(0).as_millis();
        let d1 = cfg.delay_for_attempt(1).as_millis();
        let d2 = cfg.delay_for_attempt(2).as_millis();

        // Each attempt should roughly double (within jitter bounds)
        assert!(d1 > d0, "d1={d1} should be > d0={d0}");
        assert!(d2 > d1, "d2={d2} should be > d1={d1}");
    }

    #[test]
    fn delay_capped_at_max() {
        let cfg = RetryConfig {
            max_retries: 10,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(1),
        };
        // Attempt 10 would be 100 * 1024 = 102400 ms without cap
        let d = cfg.delay_for_attempt(10);
        // With jitter the delay is at most max_delay + 25%
        assert!(
            d.as_millis() <= 1250,
            "delay {} should be <= 1250 ms",
            d.as_millis()
        );
    }

    #[test]
    fn retryable_errors_detected() {
        assert!(RetryConfig::is_retryable("connection refused by host"));
        assert!(RetryConfig::is_retryable("Connection refused"));
        assert!(RetryConfig::is_retryable("request timed out"));
        assert!(RetryConfig::is_retryable("timeout after 30s"));
        assert!(RetryConfig::is_retryable("HTTP 503 Service Unavailable"));
        assert!(RetryConfig::is_retryable("HTTP 502 Bad Gateway"));
        assert!(RetryConfig::is_retryable("rate limited 429"));
        assert!(RetryConfig::is_retryable("connect error: refused"));
    }

    #[test]
    fn non_retryable_errors_not_retried() {
        assert!(!RetryConfig::is_retryable("invalid model name"));
        assert!(!RetryConfig::is_retryable("authentication failed"));
        assert!(!RetryConfig::is_retryable("bad request"));
        assert!(!RetryConfig::is_retryable("404 not found"));
    }

    #[test]
    fn attempt_zero_close_to_base() {
        let cfg = RetryConfig::default();
        let d = cfg.delay_for_attempt(0);
        // Should be base_delay +/- 25%
        assert!(d.as_millis() >= 75, "got {}", d.as_millis());
        assert!(d.as_millis() <= 125, "got {}", d.as_millis());
    }

    #[test]
    fn very_high_attempt_does_not_overflow() {
        let cfg = RetryConfig::default();
        // Should not panic
        let d = cfg.delay_for_attempt(u32::MAX);
        assert!(d.as_millis() <= cfg.max_delay.as_millis() + cfg.max_delay.as_millis() / 4 + 1);
    }
}
