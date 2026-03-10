//! Telemetry middleware for request tracing, metrics, and logging.
//!
//! Wraps Axum with `tower-http` tracing to log all incoming requests
//! with method, path, status, and duration.

use tower_http::trace::TraceLayer;
use tracing::Level;

/// Create the tracing middleware layer.
pub fn layer()
-> TraceLayer<tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>>
{
    TraceLayer::new_for_http()
        .make_span_with(tower_http::trace::DefaultMakeSpan::new().level(Level::INFO))
        .on_response(tower_http::trace::DefaultOnResponse::new().level(Level::INFO))
        .on_failure(tower_http::trace::DefaultOnFailure::new().level(Level::ERROR))
}
