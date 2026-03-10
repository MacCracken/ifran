/// Start the synapse API server.

use synapse_core::config::SynapseConfig;
use synapse_types::error::Result;
use synapse_types::SynapseError;

pub async fn execute(bind: Option<&str>) -> Result<()> {
    let config = SynapseConfig::default();
    let addr = bind.unwrap_or(&config.server.bind);

    eprintln!("Starting Synapse API server on {addr}");
    eprintln!("  REST:  http://{addr}");
    eprintln!("  gRPC:  {}", config.server.grpc_bind);
    eprintln!("  OpenAI-compat: http://{addr}/v1/chat/completions");

    // TODO: Wire up actual Axum router from synapse-api once Phase 4 is complete.
    // For now, start a minimal health-check server so `synapse serve` is functional.

    let app = axum::Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route(
            "/v1/models",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({ "data": [] }))
            }),
        );

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| SynapseError::Other(format!("Failed to bind {addr}: {e}")))?;

    eprintln!("Server listening on {addr}");

    axum::serve(listener, app)
        .await
        .map_err(|e| SynapseError::Other(e.to_string()))?;

    Ok(())
}
