/// Start the ifran API server.
use ifran_api::state::AppState;
use ifran_core::config::IfranConfig;
use ifran_types::IfranError;
use ifran_types::error::Result;

pub async fn execute(bind: Option<&str>) -> Result<()> {
    let mut config = IfranConfig::discover();
    if let Some(addr) = bind {
        config.server.bind = addr.to_string();
    }

    let bind_addr = config.server.bind.clone();

    eprintln!("Starting Ifran API server on {bind_addr}");
    eprintln!("  REST:  http://{bind_addr}");
    eprintln!("  OpenAI-compat: http://{bind_addr}/v1/chat/completions");

    let state = AppState::new(config)
        .map_err(|e| IfranError::Other(format!("Failed to initialize state: {e}")))?;
    let app = ifran_api::router::build(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| IfranError::Other(format!("Failed to bind {bind_addr}: {e}")))?;

    eprintln!("Server ready");

    axum::serve(listener, app)
        .await
        .map_err(|e| IfranError::Other(e.to_string()))?;

    Ok(())
}
