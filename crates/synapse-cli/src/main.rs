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
    Pull {
        /// Model repo ID (e.g. meta-llama/Llama-3.1-8B-Instruct)
        model: String,
        /// Quantization filter (e.g. q4_k_m, q5_k_m, f16)
        #[arg(short, long)]
        quant: Option<String>,
    },
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
    Remove {
        /// Model name or ID to remove
        model: String,
        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Pull { model, quant } => commands::pull::execute(&model, quant.as_deref()).await,
        Commands::List => commands::list::execute().await,
        Commands::Run => { commands::run::execute().await; Ok(()) }
        Commands::Serve => { commands::serve::execute().await; Ok(()) }
        Commands::Train => { commands::train::execute().await; Ok(()) }
        Commands::Status => { commands::status::execute().await; Ok(()) }
        Commands::Remove { model, yes } => commands::remove::execute(&model, yes).await,
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
