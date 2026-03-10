//! Top-level REST API router that mounts all route groups.

use axum::routing::{get, post};
use axum::Router;
use crate::middleware;
use crate::rest::{inference, models, openai_compat, system, training};
use crate::state::AppState;
use tower_http::cors::CorsLayer;

/// Build the complete API router with all routes and middleware.
pub fn build(state: AppState) -> Router {
    Router::new()
        // System
        .route("/health", get(system::health))
        .route("/system/status", get(system::status))
        // Models
        .route("/models", get(models::list_models))
        .route("/models/{id}", get(models::get_model).delete(models::delete_model))
        // Inference
        .route("/inference", post(inference::inference))
        .route("/inference/stream", post(inference::inference_stream))
        // Training
        .route("/training/jobs", post(training::create_job).get(training::list_jobs))
        .route("/training/jobs/{id}", get(training::get_job))
        .route("/training/jobs/{id}/cancel", post(training::cancel_job))
        // OpenAI-compatible
        .route("/v1/models", get(openai_compat::list_models))
        .route("/v1/chat/completions", post(openai_compat::chat_completions))
        // Middleware
        .layer(CorsLayer::permissive())
        .layer(middleware::telemetry::layer())
        // State
        .with_state(state)
}
