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
    /// Run autonomous hyperparameter experiments
    Experiment {
        #[command(subcommand)]
        action: ExperimentAction,
    },
}

#[derive(Subcommand)]
enum ExperimentAction {
    /// Run an experiment from a TOML program file
    Run {
        /// Path to experiment program TOML file
        program: String,
    },
    /// List all experiments
    List,
    /// Show experiment status
    Status {
        /// Experiment ID (shows latest if omitted)
        id: Option<String>,
    },
    /// Show trial leaderboard for an experiment
    Leaderboard {
        /// Experiment ID
        id: String,
        /// Max results to show
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Stop a running experiment
    Stop {
        /// Experiment ID
        id: String,
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
        Commands::Experiment { action } => match action {
            ExperimentAction::Run { program } => commands::experiment::run(&program).await,
            ExperimentAction::List => commands::experiment::list().await,
            ExperimentAction::Status { id } => commands::experiment::status(id.as_deref()).await,
            ExperimentAction::Leaderboard { id, limit } => {
                commands::experiment::leaderboard(&id, limit).await
            }
            ExperimentAction::Stop { id } => commands::experiment::stop(&id).await,
        },
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_pull_basic() {
        let cli = Cli::try_parse_from(["synapse", "pull", "meta-llama/Llama-3.1-8B"]).unwrap();
        match cli.command {
            Commands::Pull { model, quant } => {
                assert_eq!(model, "meta-llama/Llama-3.1-8B");
                assert!(quant.is_none());
            }
            _ => panic!("expected Pull command"),
        }
    }

    #[test]
    fn cli_pull_with_quant() {
        let cli = Cli::try_parse_from([
            "synapse",
            "pull",
            "meta-llama/Llama-3.1-8B",
            "--quant",
            "q4_k_m",
        ])
        .unwrap();
        match cli.command {
            Commands::Pull { quant, .. } => {
                assert_eq!(quant.unwrap(), "q4_k_m");
            }
            _ => panic!("expected Pull command"),
        }
    }

    #[test]
    fn cli_list() {
        let cli = Cli::try_parse_from(["synapse", "list"]).unwrap();
        assert!(matches!(cli.command, Commands::List));
    }

    #[test]
    fn cli_run() {
        let cli = Cli::try_parse_from(["synapse", "run", "llama-7b"]).unwrap();
        match cli.command {
            Commands::Run { model } => assert_eq!(model, "llama-7b"),
            _ => panic!("expected Run command"),
        }
    }

    #[test]
    fn cli_serve_default() {
        let cli = Cli::try_parse_from(["synapse", "serve"]).unwrap();
        match cli.command {
            Commands::Serve { bind } => assert!(bind.is_none()),
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn cli_serve_with_bind() {
        let cli = Cli::try_parse_from(["synapse", "serve", "--bind", "0.0.0.0:9000"]).unwrap();
        match cli.command {
            Commands::Serve { bind } => assert_eq!(bind.unwrap(), "0.0.0.0:9000"),
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn cli_train_default_method() {
        let cli = Cli::try_parse_from([
            "synapse",
            "train",
            "--base-model",
            "llama-7b",
            "--dataset",
            "/data/train.jsonl",
        ])
        .unwrap();
        match cli.command {
            Commands::Train {
                base_model,
                dataset,
                method,
                distributed,
                ..
            } => {
                assert_eq!(base_model, "llama-7b");
                assert_eq!(dataset, "/data/train.jsonl");
                assert_eq!(method, "lora");
                assert!(!distributed);
            }
            _ => panic!("expected Train command"),
        }
    }

    #[test]
    fn cli_train_distributed() {
        let cli = Cli::try_parse_from([
            "synapse",
            "train",
            "--base-model",
            "llama-7b",
            "--dataset",
            "/data/train.jsonl",
            "--distributed",
            "--world-size",
            "4",
            "--strategy",
            "model_parallel",
        ])
        .unwrap();
        match cli.command {
            Commands::Train {
                distributed,
                world_size,
                strategy,
                ..
            } => {
                assert!(distributed);
                assert_eq!(world_size, Some(4));
                assert_eq!(strategy.unwrap(), "model_parallel");
            }
            _ => panic!("expected Train command"),
        }
    }

    #[test]
    fn cli_status() {
        let cli = Cli::try_parse_from(["synapse", "status"]).unwrap();
        assert!(matches!(cli.command, Commands::Status));
    }

    #[test]
    fn cli_remove() {
        let cli = Cli::try_parse_from(["synapse", "remove", "llama-7b"]).unwrap();
        match cli.command {
            Commands::Remove { model, yes } => {
                assert_eq!(model, "llama-7b");
                assert!(!yes);
            }
            _ => panic!("expected Remove command"),
        }
    }

    #[test]
    fn cli_remove_with_yes() {
        let cli = Cli::try_parse_from(["synapse", "remove", "llama-7b", "-y"]).unwrap();
        match cli.command {
            Commands::Remove { yes, .. } => assert!(yes),
            _ => panic!("expected Remove command"),
        }
    }

    #[test]
    fn cli_eval_defaults() {
        let cli = Cli::try_parse_from(["synapse", "eval", "llama-7b"]).unwrap();
        match cli.command {
            Commands::Eval {
                model,
                benchmark,
                dataset,
                sample_limit,
            } => {
                assert_eq!(model, "llama-7b");
                assert_eq!(benchmark, "perplexity");
                assert!(dataset.is_none());
                assert!(sample_limit.is_none());
            }
            _ => panic!("expected Eval command"),
        }
    }

    #[test]
    fn cli_eval_custom() {
        let cli = Cli::try_parse_from([
            "synapse",
            "eval",
            "llama-7b",
            "--benchmark",
            "custom",
            "--dataset",
            "/data/eval.jsonl",
            "--sample-limit",
            "100",
        ])
        .unwrap();
        match cli.command {
            Commands::Eval {
                benchmark,
                dataset,
                sample_limit,
                ..
            } => {
                assert_eq!(benchmark, "custom");
                assert_eq!(dataset.unwrap(), "/data/eval.jsonl");
                assert_eq!(sample_limit, Some(100));
            }
            _ => panic!("expected Eval command"),
        }
    }

    #[test]
    fn cli_marketplace_search() {
        let cli = Cli::try_parse_from(["synapse", "marketplace", "search", "llama"]).unwrap();
        match cli.command {
            Commands::Marketplace {
                action: MarketplaceAction::Search { query },
            } => assert_eq!(query.unwrap(), "llama"),
            _ => panic!("expected Marketplace Search"),
        }
    }

    #[test]
    fn cli_marketplace_publish() {
        let cli = Cli::try_parse_from(["synapse", "marketplace", "publish", "my-model"]).unwrap();
        match cli.command {
            Commands::Marketplace {
                action: MarketplaceAction::Publish { model },
            } => assert_eq!(model, "my-model"),
            _ => panic!("expected Marketplace Publish"),
        }
    }

    #[test]
    fn cli_marketplace_pull() {
        let cli = Cli::try_parse_from([
            "synapse",
            "marketplace",
            "pull",
            "model-x",
            "--peer",
            "http://node-2:8420",
        ])
        .unwrap();
        match cli.command {
            Commands::Marketplace {
                action: MarketplaceAction::Pull { model, peer },
            } => {
                assert_eq!(model, "model-x");
                assert_eq!(peer, "http://node-2:8420");
            }
            _ => panic!("expected Marketplace Pull"),
        }
    }

    #[test]
    fn cli_no_args_fails() {
        assert!(Cli::try_parse_from(["synapse"]).is_err());
    }

    #[test]
    fn cli_unknown_command_fails() {
        assert!(Cli::try_parse_from(["synapse", "nonexistent"]).is_err());
    }
}
