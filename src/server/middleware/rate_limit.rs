//! Per-IP rate limiting middleware using majra's token bucket rate limiter.
//!
//! Each client IP gets its own token bucket. Returns HTTP 429 Too Many
//! Requests when the limit is exceeded.

use axum::extract::{ConnectInfo, Request};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use majra::namespace::Namespace;
use majra::ratelimit::RateLimiter;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};

/// Per-IP rate limiter backed by majra.
#[derive(Clone)]
pub struct SharedLimiter {
    inner: std::sync::Arc<RateLimiter>,
    eviction_started: std::sync::Arc<AtomicBool>,
}

/// Build a per-IP rate limiter from config values.
pub fn build_limiter(per_second: u64, burst: u64) -> SharedLimiter {
    SharedLimiter {
        inner: std::sync::Arc::new(RateLimiter::new(
            per_second.max(1) as f64,
            burst.max(1) as usize,
        )),
        eviction_started: std::sync::Arc::new(AtomicBool::new(false)),
    }
}

impl SharedLimiter {
    fn check_ip(&self, ip: IpAddr) -> Result<(), ()> {
        if self.inner.check(&ip.to_string()) {
            Ok(())
        } else {
            Err(())
        }
    }

    /// Spawn a background task that periodically evicts stale IP buckets.
    ///
    /// `max_idle` — buckets idle longer than this are removed.
    /// `interval` — how often the eviction sweep runs.
    pub fn start_eviction_loop(
        &self,
        max_idle: std::time::Duration,
        interval: std::time::Duration,
    ) {
        // Prevent spawning multiple eviction tasks
        if self
            .eviction_started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let limiter = self.inner.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                let evicted = limiter.evict_stale(max_idle);
                if evicted > 0 {
                    tracing::debug!(evicted, "Rate limiter: evicted stale IP buckets");
                }
            }
        });
    }

    /// Return statistics from the underlying rate limiter.
    pub fn stats(&self) -> majra::ratelimit::RateLimitStats {
        self.inner.stats()
    }

    pub fn check_namespaced(&self, key: &str, namespace: Option<&Namespace>) -> bool {
        let namespaced_key = match namespace {
            Some(ns) => ns.key(key),
            None => key.to_string(),
        };
        self.inner.check(&namespaced_key)
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

    #[test]
    fn check_namespaced_without_namespace() {
        let limiter = build_limiter(1, 1);
        assert!(limiter.check_namespaced("10.0.0.1", None));
        assert!(!limiter.check_namespaced("10.0.0.1", None));
    }

    #[tokio::test]
    async fn eviction_loop_removes_stale_keys() {
        let limiter = build_limiter(100, 100);
        let ip = IpAddr::from([192, 168, 1, 1]);

        // Create a bucket entry
        assert!(limiter.check_ip(ip).is_ok());
        let before = limiter.stats();
        assert!(before.active_keys >= 1);

        // Start eviction with a very short max_idle so the entry is immediately stale
        limiter.start_eviction_loop(
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(50),
        );

        // Give the eviction loop time to run (generous margin for instrumented builds)
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let after = limiter.stats();
        assert_eq!(
            after.active_keys, 0,
            "stale buckets should have been evicted"
        );
    }

    #[test]
    fn stats_returns_data() {
        let limiter = build_limiter(10, 10);
        let ip = IpAddr::from([10, 0, 0, 1]);
        assert!(limiter.check_ip(ip).is_ok());
        let stats = limiter.stats();
        assert!(stats.active_keys >= 1);
    }

    #[test]
    fn check_namespaced_isolates_tenants() {
        let limiter = build_limiter(1, 1);
        let ns_a = Namespace::new("tenant-a");
        let ns_b = Namespace::new("tenant-b");

        // Exhaust tenant-a's bucket for this IP
        assert!(limiter.check_namespaced("10.0.0.1", Some(&ns_a)));
        assert!(!limiter.check_namespaced("10.0.0.1", Some(&ns_a)));

        // tenant-b with same IP should still pass
        assert!(limiter.check_namespaced("10.0.0.1", Some(&ns_b)));
    }
}
