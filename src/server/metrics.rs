//! Application-wide Prometheus metrics.
//!
//! All metric handles are registered once at startup and cloned cheaply
//! (they are `Arc`-backed internally by the prometheus crate).

use prometheus::{Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry};

/// Application-wide Prometheus metrics.
#[derive(Clone)]
pub struct Metrics {
    /// HTTP request duration in seconds.
    pub request_duration: Histogram,
    /// Total HTTP requests served.
    pub requests_total: IntCounter,
    /// Total HTTP 429 rate-limit rejections.
    pub rate_limit_rejections: IntCounter,
    /// Currently loaded models.
    pub loaded_models: IntGauge,
    /// Active (running) training jobs.
    pub active_training_jobs: IntGauge,
    /// Queued training jobs.
    pub queued_training_jobs: IntGauge,
    /// Total inference requests.
    pub inference_requests: IntCounter,
    /// Inference cache hit rate (0.0–1.0, updated on metrics scrape).
    pub cache_hit_rate: prometheus::Gauge,
    /// Registered fleet nodes.
    pub fleet_nodes: IntGauge,
}

impl Metrics {
    /// Register all metrics with the given Prometheus registry.
    ///
    /// # Panics
    ///
    /// Panics if registration fails (duplicate metric names). This is called
    /// once at startup, so a panic here is an immediate, obvious bug.
    #[must_use]
    pub fn register(registry: &Registry) -> Self {
        let request_duration = Histogram::with_opts(
            HistogramOpts::new(
                "ifran_request_duration_seconds",
                "HTTP request duration in seconds",
            )
            .buckets(vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
        )
        .expect("valid histogram opts");
        registry
            .register(Box::new(request_duration.clone()))
            .expect("request_duration registration");

        let requests_total = IntCounter::with_opts(Opts::new(
            "ifran_requests_total",
            "Total HTTP requests served",
        ))
        .expect("valid counter opts");
        registry
            .register(Box::new(requests_total.clone()))
            .expect("requests_total registration");

        let rate_limit_rejections = IntCounter::with_opts(Opts::new(
            "ifran_rate_limit_rejections_total",
            "Total HTTP 429 rate-limit rejections",
        ))
        .expect("valid counter opts");
        registry
            .register(Box::new(rate_limit_rejections.clone()))
            .expect("rate_limit_rejections registration");

        let loaded_models = IntGauge::with_opts(Opts::new(
            "ifran_loaded_models",
            "Number of currently loaded models",
        ))
        .expect("valid gauge opts");
        registry
            .register(Box::new(loaded_models.clone()))
            .expect("loaded_models registration");

        let active_training_jobs = IntGauge::with_opts(Opts::new(
            "ifran_active_training_jobs",
            "Number of currently running training jobs",
        ))
        .expect("valid gauge opts");
        registry
            .register(Box::new(active_training_jobs.clone()))
            .expect("active_training_jobs registration");

        let queued_training_jobs = IntGauge::with_opts(Opts::new(
            "ifran_queued_training_jobs",
            "Number of queued training jobs",
        ))
        .expect("valid gauge opts");
        registry
            .register(Box::new(queued_training_jobs.clone()))
            .expect("queued_training_jobs registration");

        let inference_requests = IntCounter::with_opts(Opts::new(
            "ifran_inference_requests_total",
            "Total inference requests",
        ))
        .expect("valid counter opts");
        registry
            .register(Box::new(inference_requests.clone()))
            .expect("inference_requests registration");

        let cache_hit_rate = prometheus::Gauge::with_opts(Opts::new(
            "ifran_cache_hit_rate",
            "Inference cache hit rate (0.0 to 1.0)",
        ))
        .expect("valid gauge opts");
        registry
            .register(Box::new(cache_hit_rate.clone()))
            .expect("cache_hit_rate registration");

        let fleet_nodes = IntGauge::with_opts(Opts::new(
            "ifran_fleet_nodes",
            "Number of registered fleet nodes",
        ))
        .expect("valid gauge opts");
        registry
            .register(Box::new(fleet_nodes.clone()))
            .expect("fleet_nodes registration");

        Self {
            request_duration,
            requests_total,
            rate_limit_rejections,
            loaded_models,
            active_training_jobs,
            queued_training_jobs,
            inference_requests,
            cache_hit_rate,
            fleet_nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_register_succeeds() {
        let registry = Registry::new();
        let metrics = Metrics::register(&registry);

        // Verify counters start at zero
        assert_eq!(metrics.requests_total.get(), 0);
        assert_eq!(metrics.rate_limit_rejections.get(), 0);
        assert_eq!(metrics.inference_requests.get(), 0);

        // Verify gauges start at zero
        assert_eq!(metrics.loaded_models.get(), 0);
        assert_eq!(metrics.active_training_jobs.get(), 0);
        assert_eq!(metrics.queued_training_jobs.get(), 0);
        assert_eq!(metrics.fleet_nodes.get(), 0);
        assert!((metrics.cache_hit_rate.get() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_counters_increment() {
        let registry = Registry::new();
        let metrics = Metrics::register(&registry);

        metrics.requests_total.inc();
        metrics.requests_total.inc();
        assert_eq!(metrics.requests_total.get(), 2);

        metrics.rate_limit_rejections.inc();
        assert_eq!(metrics.rate_limit_rejections.get(), 1);

        metrics.inference_requests.inc();
        assert_eq!(metrics.inference_requests.get(), 1);
    }

    #[test]
    fn metrics_gauges_update() {
        let registry = Registry::new();
        let metrics = Metrics::register(&registry);

        metrics.loaded_models.set(3);
        assert_eq!(metrics.loaded_models.get(), 3);

        metrics.active_training_jobs.set(2);
        assert_eq!(metrics.active_training_jobs.get(), 2);

        metrics.fleet_nodes.set(5);
        assert_eq!(metrics.fleet_nodes.get(), 5);

        metrics.cache_hit_rate.set(0.85);
        assert!((metrics.cache_hit_rate.get() - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_histogram_observes() {
        let registry = Registry::new();
        let metrics = Metrics::register(&registry);

        metrics.request_duration.observe(0.042);
        metrics.request_duration.observe(0.123);
        assert_eq!(metrics.request_duration.get_sample_count(), 2);
    }

    #[test]
    fn metrics_clone_shares_state() {
        let registry = Registry::new();
        let m1 = Metrics::register(&registry);
        let m2 = m1.clone();

        m1.requests_total.inc();
        assert_eq!(m2.requests_total.get(), 1);
    }

    #[test]
    fn metrics_appear_in_gather() {
        let registry = Registry::new();
        let metrics = Metrics::register(&registry);

        metrics.requests_total.inc_by(42);
        metrics.loaded_models.set(7);

        let families = registry.gather();
        let names: Vec<&str> = families.iter().map(|f| f.name()).collect();
        assert!(names.contains(&"ifran_requests_total"));
        assert!(names.contains(&"ifran_loaded_models"));
        assert!(names.contains(&"ifran_request_duration_seconds"));
    }
}
