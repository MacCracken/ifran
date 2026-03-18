//! Top-level REST API router that mounts all route groups.

use crate::middleware;
use crate::rest::{
    bridge, distributed, eval, experiment, fleet, inference, lineage, marketplace, models,
    openai_compat, rag, rlhf, system, tenants, training, versioning,
};
use crate::state::AppState;
use axum::Router;
use axum::middleware as axum_mw;
use axum::routing::{delete, get, post};
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;

/// Build the complete API router with all routes and middleware.
pub fn build(state: AppState) -> Router {
    let limiter = middleware::rate_limit::build_limiter(
        state.config.security.rate_limit_per_second,
        state.config.security.rate_limit_burst,
    );

    // Admin routes (multi-tenant only) — separate auth via SYNAPSE_ADMIN_KEY
    let admin_routes = if state.config.security.multi_tenant {
        Router::new()
            .route(
                "/admin/tenants",
                get(tenants::list_tenants).post(tenants::create_tenant),
            )
            .route("/admin/tenants/{id}", delete(tenants::disable_tenant))
            .layer(axum_mw::from_fn(tenants::require_admin_auth))
            .with_state(state.clone())
    } else {
        Router::new()
    };

    Router::new()
        // System
        .route("/health", get(system::health))
        .route("/system/status", get(system::status))
        // Models
        .route("/models", get(models::list_models))
        .route(
            "/models/{id}",
            get(models::get_model).delete(models::delete_model),
        )
        // Inference
        .route("/inference", post(inference::inference))
        .route("/inference/stream", post(inference::inference_stream))
        // Training
        .route(
            "/training/jobs",
            post(training::create_job).get(training::list_jobs),
        )
        .route("/training/jobs/{id}", get(training::get_job))
        .route("/training/jobs/{id}/cancel", post(training::cancel_job))
        .route("/training/jobs/{id}/stream", get(training::stream_job))
        // Distributed Training
        .route(
            "/training/distributed/jobs",
            post(distributed::create_job).get(distributed::list_jobs),
        )
        .route("/training/distributed/jobs/{id}", get(distributed::get_job))
        .route(
            "/training/distributed/jobs/{id}/workers",
            post(distributed::assign_worker),
        )
        .route(
            "/training/distributed/jobs/{id}/start",
            post(distributed::start_job),
        )
        .route(
            "/training/distributed/jobs/{id}/workers/{rank}/complete",
            post(distributed::worker_completed),
        )
        .route(
            "/training/distributed/jobs/{id}/fail",
            post(distributed::fail_job),
        )
        .route(
            "/training/distributed/jobs/{id}/aggregate",
            post(distributed::aggregate),
        )
        // Experiments
        .route(
            "/experiments",
            post(experiment::create_experiment).get(experiment::list_experiments),
        )
        .route("/experiments/{id}", get(experiment::get_experiment))
        .route(
            "/experiments/{id}/leaderboard",
            get(experiment::get_leaderboard),
        )
        .route("/experiments/{id}/stop", post(experiment::stop_experiment))
        // Eval
        .route("/eval/runs", post(eval::create_run).get(eval::list_runs))
        .route("/eval/runs/{id}", get(eval::get_run))
        // Marketplace
        .route("/marketplace/search", get(marketplace::search))
        .route("/marketplace/entries", get(marketplace::list_entries))
        .route("/marketplace/publish", post(marketplace::publish))
        .route(
            "/marketplace/entries/{name}",
            delete(marketplace::unpublish),
        )
        .route("/marketplace/download/{name}", get(marketplace::download))
        .route("/marketplace/pull", post(marketplace::pull))
        // RAG
        .route(
            "/rag/pipelines",
            post(rag::create_pipeline).get(rag::list_pipelines),
        )
        .route(
            "/rag/pipelines/{id}",
            get(rag::get_pipeline).delete(rag::delete_pipeline),
        )
        .route("/rag/pipelines/{id}/ingest", post(rag::ingest_document))
        .route("/rag/query", post(rag::query))
        // RLHF
        .route(
            "/rlhf/sessions",
            post(rlhf::create_session).get(rlhf::list_sessions),
        )
        .route("/rlhf/sessions/{id}", get(rlhf::get_session))
        .route(
            "/rlhf/sessions/{id}/pairs",
            post(rlhf::add_pairs).get(rlhf::get_pairs),
        )
        .route("/rlhf/pairs/{id}/annotate", post(rlhf::annotate))
        .route("/rlhf/sessions/{id}/export", post(rlhf::export_session))
        .route("/rlhf/sessions/{id}/stats", get(rlhf::get_stats))
        // Versioning
        .route(
            "/versions",
            post(versioning::create_version).get(versioning::list_versions),
        )
        .route("/versions/{id}", get(versioning::get_version))
        .route("/versions/{id}/lineage", get(versioning::get_lineage))
        // Lineage
        .route(
            "/lineage",
            post(lineage::record_node).get(lineage::list_nodes),
        )
        .route("/lineage/{id}", get(lineage::get_node))
        .route("/lineage/{id}/ancestry", get(lineage::get_ancestry))
        // Bridge
        .route("/bridge/status", get(bridge::status))
        .route("/bridge/connect", post(bridge::connect))
        .route("/bridge/heartbeat", post(bridge::heartbeat))
        // OpenAI-compatible
        .route("/v1/models", get(openai_compat::list_models))
        .route(
            "/v1/chat/completions",
            post(openai_compat::chat_completions),
        )
        // Fleet management
        .route(
            "/fleet/nodes",
            post(fleet::register_node).get(fleet::list_nodes),
        )
        .route("/fleet/nodes/{id}/heartbeat", post(fleet::heartbeat))
        .route("/fleet/nodes/{id}", delete(fleet::remove_node))
        .route("/fleet/stats", get(fleet::fleet_stats))
        // GPU telemetry
        .route("/system/gpu/telemetry", get(system::gpu_telemetry))
        // Model discovery
        .route("/models/discover", get(system::discover_models))
        .merge(admin_routes)
        // Middleware stack (axum applies bottom-up: last .layer() runs first)
        // 1. Auth (innermost — runs last on request, first on response)
        .layer(axum_mw::from_fn_with_state(
            state.clone(),
            middleware::auth::require_auth,
        ))
        // 2. CORS
        .layer(build_cors_layer(
            &state.config.security.cors_allowed_origins,
        ))
        // 3. Body size limit
        .layer(RequestBodyLimitLayer::new(
            state.config.security.max_body_size_bytes,
        ))
        // 4. Telemetry
        .layer(middleware::telemetry::layer())
        // 5. Rate limiting (outermost — runs first, protects against flood)
        .layer(axum_mw::from_fn_with_state(
            limiter,
            middleware::rate_limit::rate_limit,
        ))
        // State
        .with_state(state)
}

/// Build a CORS layer from configured allowed origins.
///
/// - Empty list = permissive (backward compatible with dev setups)
/// - `["*"]` = permissive (explicit wildcard)
/// - Specific origins = restrictive
fn build_cors_layer(origins: &[String]) -> CorsLayer {
    if origins.is_empty() || (origins.len() == 1 && origins[0] == "*") {
        return CorsLayer::permissive();
    }

    let parsed: Vec<axum::http::HeaderValue> =
        origins.iter().filter_map(|o| o.parse().ok()).collect();

    CorsLayer::new()
        .allow_origin(parsed)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cors_empty_is_permissive() {
        // Should not panic
        let _layer = build_cors_layer(&[]);
    }

    #[test]
    fn cors_wildcard_is_permissive() {
        let _layer = build_cors_layer(&["*".into()]);
    }

    #[test]
    fn cors_specific_origins() {
        let _layer = build_cors_layer(&[
            "https://app.example.com".into(),
            "https://admin.example.com".into(),
        ]);
    }

    #[test]
    fn cors_invalid_origin_skipped() {
        // Invalid origins are silently filtered out
        let _layer = build_cors_layer(&["\x00invalid".into(), "https://valid.com".into()]);
    }
}
