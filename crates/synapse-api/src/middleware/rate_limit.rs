//! Per-IP rate limiting middleware using the `governor` crate.
//!
//! Each client IP gets its own token-bucket rate limiter, backed by a
//! `DashMap`. Returns HTTP 429 Too Many Requests when the limit is exceeded.

use axum::extract::{ConnectInfo, Request};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use dashmap::DashMap;
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use std::net::{IpAddr, SocketAddr};
use std::num::NonZeroU32;
use std::sync::Arc;

type Bucket = Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>;

/// Per-IP rate limiter: each IP address gets its own token bucket.
#[derive(Clone)]
pub struct SharedLimiter {
    quota: Quota,
    buckets: Arc<DashMap<IpAddr, Bucket>>,
}

impl SharedLimiter {
    fn check_ip(&self, ip: IpAddr) -> Result<(), ()> {
        let limiter = self
            .buckets
            .entry(ip)
            .or_insert_with(|| Arc::new(RateLimiter::direct(self.quota)))
            .clone();
        limiter.check().map_err(|_| ())
    }
}

/// Build a per-IP rate limiter from config values.
pub fn build_limiter(per_second: u64, burst: u64) -> SharedLimiter {
    let per_sec = NonZeroU32::new(per_second.max(1) as u32).unwrap();
    let burst_size = NonZeroU32::new(burst.max(1) as u32).unwrap();
    let quota = Quota::per_second(per_sec).allow_burst(burst_size);
    SharedLimiter {
        quota,
        buckets: Arc::new(DashMap::new()),
    }
}

/// Rate limiting middleware function.
///
/// Extracts the client IP from `ConnectInfo<SocketAddr>` in request
/// extensions (provided automatically by `axum::serve`) and checks
/// the per-IP bucket.
pub async fn rate_limit(
    limiter: axum::extract::State<SharedLimiter>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let ip = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip())
        .unwrap_or(IpAddr::from([127, 0, 0, 1]));

    match limiter.check_ip(ip) {
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
        let ip = IpAddr::from([1, 2, 3, 4]);
        assert!(limiter.check_ip(ip).is_ok());
    }

    #[test]
    fn build_limiter_minimum_values() {
        let limiter = build_limiter(0, 0);
        let ip = IpAddr::from([1, 2, 3, 4]);
        assert!(limiter.check_ip(ip).is_ok());
    }

    #[test]
    fn limiter_allows_burst() {
        let limiter = build_limiter(1, 5);
        let ip = IpAddr::from([10, 0, 0, 1]);
        for _ in 0..5 {
            assert!(limiter.check_ip(ip).is_ok());
        }
    }

    #[test]
    fn limiter_rejects_over_burst() {
        let limiter = build_limiter(1, 2);
        let ip = IpAddr::from([10, 0, 0, 1]);
        assert!(limiter.check_ip(ip).is_ok());
        assert!(limiter.check_ip(ip).is_ok());
        assert!(limiter.check_ip(ip).is_err());
    }

    #[test]
    fn different_ips_have_separate_buckets() {
        let limiter = build_limiter(1, 1);
        let ip_a = IpAddr::from([10, 0, 0, 1]);
        let ip_b = IpAddr::from([10, 0, 0, 2]);

        // Exhaust IP A's bucket
        assert!(limiter.check_ip(ip_a).is_ok());
        assert!(limiter.check_ip(ip_a).is_err());

        // IP B should still be allowed
        assert!(limiter.check_ip(ip_b).is_ok());
    }
}
