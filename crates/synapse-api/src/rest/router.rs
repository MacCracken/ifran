//! Top-level REST API router that mounts all route groups.

use crate::middleware;
use crate::rest::{bridge, distributed, eval, inference, marketplace, models, openai_compat, system, training};
use crate::state::AppState;
use axum::Router;
use axum::middleware as axum_mw;
use axum::routing::{delete, get, post};
use tower_http::cors::CorsLayer;

/// Build the complete API router with all routes and middleware.
pub fn build(state: AppState) -> Router {
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
        // Distributed Training
        .route(
            "/training/distributed/jobs",
            post(distributed::create_job).get(distributed::list_jobs),
        )
        .route(
            "/training/distributed/jobs/{id}",
            get(distributed::get_job),
        )
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
        .route(
            "/marketplace/download/{name}",
            get(marketplace::download),
        )
        .route("/marketplace/pull", post(marketplace::pull))
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
        // Middleware (order: outermost first — auth runs before telemetry)
        .layer(axum_mw::from_fn(middleware::auth::require_auth))
        .layer(CorsLayer::permissive())
        .layer(middleware::telemetry::layer())
        // State
        .with_state(state)
}
