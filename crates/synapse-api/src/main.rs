/// Binary entrypoint for the synapse-server.
use synapse_api::state::AppState;
use synapse_core::config::SynapseConfig;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config = SynapseConfig::discover();
    let bind_addr = config.server.bind.clone();

    let state = AppState::new(config).expect("Failed to initialize application state");
    let app = synapse_api::router::build(state);

    tracing::info!("Starting synapse-server on {bind_addr}");

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .expect("Failed to bind address");

    axum::serve(listener, app).await.expect("Server error");
}
