/// Start the synapse API server.

use synapse_api::state::AppState;
use synapse_core::config::SynapseConfig;
use synapse_types::error::Result;
use synapse_types::SynapseError;

pub async fn execute(bind: Option<&str>) -> Result<()> {
    let mut config = SynapseConfig::default();
    if let Some(addr) = bind {
        config.server.bind = addr.to_string();
    }

    let bind_addr = config.server.bind.clone();

    eprintln!("Starting Synapse API server on {bind_addr}");
    eprintln!("  REST:  http://{bind_addr}");
    eprintln!("  OpenAI-compat: http://{bind_addr}/v1/chat/completions");

    let state = AppState::new(config)
        .map_err(|e| SynapseError::Other(format!("Failed to initialize state: {e}")))?;
    let app = synapse_api::router::build(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| SynapseError::Other(format!("Failed to bind {bind_addr}: {e}")))?;

    eprintln!("Server ready");

    axum::serve(listener, app)
        .await
        .map_err(|e| SynapseError::Other(e.to_string()))?;

    Ok(())
}
