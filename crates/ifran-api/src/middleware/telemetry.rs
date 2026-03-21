//! Request telemetry and observability.
//!
//! Provides HTTP request tracing via tower-http and optional OTLP export
//! when compiled with the `otlp` feature flag.

use tower_http::trace::TraceLayer;
use tracing::Level;

/// Build the tower-http trace layer for request/response logging.
pub fn layer()
-> TraceLayer<tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>>
{
    TraceLayer::new_for_http()
        .make_span_with(tower_http::trace::DefaultMakeSpan::new().level(Level::INFO))
        .on_response(tower_http::trace::DefaultOnResponse::new().level(Level::INFO))
        .on_failure(tower_http::trace::DefaultOnFailure::new().level(Level::ERROR))
}

/// Initialize the tracing subscriber.
///
/// When compiled with `otlp` feature and `OTEL_EXPORTER_OTLP_ENDPOINT` is set,
/// exports traces and spans to an OTLP collector (e.g., daimon's collector).
/// Otherwise uses a stderr-based subscriber with env-filter.
pub fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    #[cfg(feature = "otlp")]
    {
        if std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_ok() {
            if let Ok(exporter) = opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .build()
            {
                let tracer_provider = opentelemetry_sdk::trace::TracerProvider::builder()
                    .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
                    .with_resource(opentelemetry_sdk::Resource::new(vec![
                        opentelemetry::KeyValue::new("service.name", "ifran"),
                    ]))
                    .build();

                use opentelemetry::trace::TracerProvider as _;
                let tracer = tracer_provider.tracer("ifran");

                let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

                use tracing_subscriber::layer::SubscriberExt;
                use tracing_subscriber::util::SubscriberInitExt;
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(tracing_subscriber::fmt::layer())
                    .with(telemetry)
                    .init();

                tracing::info!("OTLP tracing enabled");
                return;
            }
        }
    }

    // Fallback: stderr-only subscriber
    tracing_subscriber::fmt().with_env_filter(env_filter).init();
}
