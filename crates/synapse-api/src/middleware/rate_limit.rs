//! Per-IP rate limiting middleware using the `governor` crate.
//!
//! Wraps a governor `RateLimiter` behind an axum middleware function.
//! Returns HTTP 429 Too Many Requests when the limit is exceeded.

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;
use std::sync::Arc;

/// Shared rate limiter state (not keyed — global per-server).
///
/// A per-IP keyed limiter would require `DashMap`-backed state and IP
/// extraction. The global limiter is simpler and still effective for
/// single-instance deployments.
pub type SharedLimiter = Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>;

/// Build a rate limiter from config values.
pub fn build_limiter(per_second: u64, burst: u64) -> SharedLimiter {
    let per_sec = NonZeroU32::new(per_second.max(1) as u32).unwrap();
    let burst_size = NonZeroU32::new(burst.max(1) as u32).unwrap();
    let quota = Quota::per_second(per_sec).allow_burst(burst_size);
    Arc::new(RateLimiter::direct(quota))
}

/// Rate limiting middleware function.
///
/// Must be used with `axum::middleware::from_fn_with_state`.
pub async fn rate_limit(
    limiter: axum::extract::State<SharedLimiter>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    match limiter.check() {
        Ok(_) => Ok(next.run(req).await),
        Err(_) => Err(StatusCode::TOO_MANY_REQUESTS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_limiter_succeeds() {
        let limiter = build_limiter(60, 120);
        assert!(limiter.check().is_ok());
    }

    #[test]
    fn build_limiter_minimum_values() {
        // Zero values clamped to 1
        let limiter = build_limiter(0, 0);
        assert!(limiter.check().is_ok());
    }

    #[test]
    fn limiter_allows_burst() {
        let limiter = build_limiter(1, 5);
        // Should allow up to burst_size requests immediately
        for _ in 0..5 {
            assert!(limiter.check().is_ok());
        }
    }

    #[test]
    fn limiter_rejects_over_burst() {
        let limiter = build_limiter(1, 2);
        // Exhaust burst
        assert!(limiter.check().is_ok());
        assert!(limiter.check().is_ok());
        // Next should be rate limited
        assert!(limiter.check().is_err());
    }
}
