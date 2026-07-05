//! Circuit breaker for inference backends.
//!
//! States: Closed -> Open -> HalfOpen -> Closed
//! - **Closed**: requests pass through normally.
//! - **Open**: requests fail immediately (backend assumed down).
//! - **HalfOpen**: one probe request allowed to test recovery.

use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Possible states of the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CircuitState {
    /// Requests pass through; failures are counted.
    Closed,
    /// Backend is assumed down; requests are rejected immediately.
    Open,
    /// A single probe request is allowed to check if the backend recovered.
    HalfOpen,
}

/// Internal state protected by a single mutex to avoid inconsistent lock ordering.
struct CircuitBreakerInner {
    state: CircuitState,
    last_failure: Option<Instant>,
    consecutive_failures: u32,
}

/// A circuit breaker that tracks consecutive failures and controls access
/// to a backend.
pub struct CircuitBreaker {
    failure_threshold: u32,
    recovery_timeout: Duration,
    inner: Mutex<CircuitBreakerInner>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// * `failure_threshold` — number of consecutive failures before the
    ///   circuit opens.
    /// * `recovery_timeout` — how long to wait in the open state before
    ///   transitioning to half-open.
    #[must_use]
    pub fn new(failure_threshold: u32, recovery_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            recovery_timeout,
            inner: Mutex::new(CircuitBreakerInner {
                state: CircuitState::Closed,
                last_failure: None,
                consecutive_failures: 0,
            }),
        }
    }

    /// Check if a request should be allowed through the breaker.
    ///
    /// Returns `true` when the request may proceed, `false` when the
    /// circuit is open and the recovery timeout has not yet elapsed.
    pub async fn allow_request(&self) -> bool {
        let mut inner = self.inner.lock().await;
        match inner.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(ts) = inner.last_failure {
                    if ts.elapsed() >= self.recovery_timeout {
                        inner.state = CircuitState::HalfOpen;
                        tracing::info!("Circuit breaker transitioning to half-open");
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => {
                // Allow one probe request — the caller must report
                // success or failure which will transition the state.
                true
            }
        }
    }

    /// Record a successful request, resetting the breaker to closed.
    pub async fn record_success(&self) {
        let mut inner = self.inner.lock().await;
        inner.consecutive_failures = 0;
        if inner.state != CircuitState::Closed {
            tracing::info!("Circuit breaker closing after successful probe");
            inner.state = CircuitState::Closed;
        }
    }

    /// Record a failed request.  If the failure threshold is reached the
    /// circuit opens.
    pub async fn record_failure(&self) {
        let mut inner = self.inner.lock().await;
        inner.consecutive_failures += 1;
        inner.last_failure = Some(Instant::now());
        let new_count = inner.consecutive_failures;

        if inner.state == CircuitState::HalfOpen {
            tracing::warn!("Probe request failed — circuit remains open");
            inner.state = CircuitState::Open;
        } else if new_count >= self.failure_threshold && inner.state != CircuitState::Open {
            tracing::warn!(
                failures = new_count,
                threshold = self.failure_threshold,
                "Circuit breaker opening"
            );
            inner.state = CircuitState::Open;
        }
    }

    /// Get the current circuit state.
    pub async fn state(&self) -> CircuitState {
        self.inner.lock().await.state
    }

    /// Get the current consecutive failure count.
    pub async fn failure_count(&self) -> u32 {
        self.inner.lock().await.consecutive_failures
    }
}

// Implement Debug manually since Mutex is not Debug-friendly.
impl std::fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("failure_threshold", &self.failure_threshold)
            .field("recovery_timeout", &self.recovery_timeout)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn starts_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn allows_requests_when_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        assert!(cb.allow_request().await);
    }

    #[tokio::test]
    async fn opens_after_n_failures() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(30));
        cb.record_failure().await;
        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Closed);
        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn rejects_requests_when_open() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(60));
        cb.record_failure().await;
        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Open);
        assert!(!cb.allow_request().await);
    }

    #[tokio::test]
    async fn transitions_to_half_open_after_timeout() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10));
        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Open);

        // Wait for the recovery timeout
        tokio::time::sleep(Duration::from_millis(20)).await;

        assert!(cb.allow_request().await);
        assert_eq!(cb.state().await, CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn successful_probe_closes_circuit() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10));
        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Open);

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(cb.allow_request().await); // transitions to HalfOpen
        assert_eq!(cb.state().await, CircuitState::HalfOpen);

        cb.record_success().await;
        assert_eq!(cb.state().await, CircuitState::Closed);
        assert_eq!(cb.failure_count().await, 0);
    }

    #[tokio::test]
    async fn failed_probe_keeps_open() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10));
        cb.record_failure().await;

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(cb.allow_request().await); // transitions to HalfOpen

        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn record_success_while_closed_stays_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        cb.record_success().await;
        assert_eq!(cb.state().await, CircuitState::Closed);
        assert_eq!(cb.failure_count().await, 0);
    }

    #[tokio::test]
    async fn failure_below_threshold_stays_closed() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(5));
        for _ in 0..4 {
            cb.record_failure().await;
        }
        assert_eq!(cb.state().await, CircuitState::Closed);
        assert_eq!(cb.failure_count().await, 4);
    }

    #[tokio::test]
    async fn debug_format_does_not_panic() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        cb.record_failure().await;
        let debug = format!("{cb:?}");
        assert!(debug.contains("CircuitBreaker"));
        assert!(debug.contains("failure_threshold"));
    }

    #[tokio::test]
    async fn success_resets_failure_count() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        cb.record_failure().await;
        cb.record_failure().await;
        assert_eq!(cb.failure_count().await, 2);

        cb.record_success().await;
        assert_eq!(cb.failure_count().await, 0);
        assert_eq!(cb.state().await, CircuitState::Closed);
    }
}
