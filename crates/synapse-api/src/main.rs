/// Binary entrypoint for the synapse-server.

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    tracing::info!("Starting synapse-server...");

    // TODO: Initialize application state, routes, and start the server.
}
