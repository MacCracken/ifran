/// Synapse CLI - command-line interface for managing models and inference.

use clap::{Parser, Subcommand};

mod commands;
mod output;

#[derive(Parser)]
#[command(name = "synapse", about = "Local LLM inference and training platform")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pull a model from a registry
    Pull,
    /// List locally available models
    List,
    /// Run inference on a model
    Run,
    /// Start the API server
    Serve,
    /// Start or manage training jobs
    Train,
    /// Show status of running models and jobs
    Status,
    /// Remove a local model
    Remove,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Pull => commands::pull::execute().await,
        Commands::List => commands::list::execute().await,
        Commands::Run => commands::run::execute().await,
        Commands::Serve => commands::serve::execute().await,
        Commands::Train => commands::train::execute().await,
        Commands::Status => commands::status::execute().await,
        Commands::Remove => commands::remove::execute().await,
    }
}
