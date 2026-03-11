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
    /// Run interactive inference on a model
    Run {
        /// Model name or ID
        model: String,
    },
    /// Start the API server
    Serve {
        /// Bind address (e.g. 0.0.0.0:8420)
        #[arg(short, long)]
        bind: Option<String>,
    },
    /// Start a training job
    Train {
        /// Base model name or path
        #[arg(long)]
        base_model: String,
        /// Dataset path (JSONL file)
        #[arg(long)]
        dataset: String,
        /// Training method: lora, qlora, full, dpo, rlhf, distillation
        #[arg(long, default_value = "lora")]
        method: String,
        /// Enable distributed training
        #[arg(long)]
        distributed: bool,
        /// Number of workers for distributed training
        #[arg(long)]
        world_size: Option<u32>,
        /// Distributed strategy: data_parallel, model_parallel, pipeline_parallel
        #[arg(long)]
        strategy: Option<String>,
    },
    /// Show status of system, models, and jobs
    Status,
    /// Remove a local model
    Remove {
        /// Model name or ID to remove
        model: String,
        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Evaluate a model with benchmarks
    Eval {
        /// Model name
        model: String,
        /// Benchmark: perplexity, mmlu, hellaswag, humaneval, custom
        #[arg(long, default_value = "perplexity")]
        benchmark: String,
        /// Dataset path (required for custom benchmark)
        #[arg(long)]
        dataset: Option<String>,
        /// Max samples to evaluate
        #[arg(long)]
        sample_limit: Option<usize>,
    },
    /// Marketplace commands
    Marketplace {
        #[command(subcommand)]
        action: MarketplaceAction,
    },
}

#[derive(Subcommand)]
enum MarketplaceAction {
    /// Search the model marketplace
    Search {
        /// Search query
        query: Option<String>,
    },
    /// Publish a local model to the marketplace
    Publish {
        /// Model name to publish
        model: String,
    },
    /// Unpublish a model from the marketplace
    Unpublish {
        /// Model name to unpublish
        model: String,
    },
    /// Pull a model from a marketplace peer
    Pull {
        /// Model name to pull
        model: String,
        /// Peer URL to pull from (e.g. http://node-2:8420)
        #[arg(long)]
        peer: String,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Pull { model, quant } => commands::pull::execute(&model, quant.as_deref()).await,
        Commands::List => commands::list::execute().await,
        Commands::Run { model } => commands::run::execute(&model).await,
        Commands::Serve { bind } => commands::serve::execute(bind.as_deref()).await,
        Commands::Train {
            base_model,
            dataset,
            method,
            distributed,
            world_size,
            strategy,
        } => {
            if distributed {
                commands::train::execute_distributed(
                    &base_model,
                    &dataset,
                    &method,
                    world_size.unwrap_or(2),
                    strategy.as_deref().unwrap_or("data_parallel"),
                )
                .await
            } else {
                commands::train::execute(&base_model, &dataset, &method).await
            }
        }
        Commands::Status => commands::status::execute().await,
        Commands::Remove { model, yes } => commands::remove::execute(&model, yes).await,
        Commands::Eval {
            model,
            benchmark,
            dataset,
            sample_limit,
        } => commands::eval::execute(&model, &benchmark, dataset.as_deref(), sample_limit).await,
        Commands::Marketplace { action } => match action {
            MarketplaceAction::Search { query } => {
                commands::marketplace::search(query.as_deref()).await
            }
            MarketplaceAction::Publish { model } => commands::marketplace::publish(&model).await,
            MarketplaceAction::Unpublish { model } => {
                commands::marketplace::unpublish(&model).await
            }
            MarketplaceAction::Pull { model, peer } => {
                commands::marketplace::pull(&model, &peer).await
            }
        },
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
