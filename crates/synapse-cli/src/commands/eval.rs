//! CLI command for model evaluation.

use synapse_core::eval::runner::EvalRunner;
use synapse_types::eval::*;

/// Run an evaluation benchmark on a model.
pub async fn execute(
    model: &str,
    benchmark: &str,
    dataset: Option<&str>,
    sample_limit: Option<usize>,
) -> synapse_types::error::Result<()> {
    let runner = EvalRunner::new();

    let kind = parse_benchmark(benchmark)?;

    let config = EvalConfig {
        model_name: model.to_string(),
        benchmarks: vec![kind],
        sample_limit,
        dataset_path: dataset.map(String::from),
    };

    let run_id = runner.create_run(config).await?;
    runner.start_run(run_id).await?;

    match kind {
        BenchmarkKind::Custom => {
            let dataset_path = dataset.ok_or_else(|| {
                synapse_types::SynapseError::EvalError("Custom benchmark requires --dataset".into())
            })?;

            // MVP: stub inference function — real implementation will connect to backend
            let infer_fn = |_prompt: String| async {
                Err(synapse_types::SynapseError::EvalError(
                    "Inference not yet wired to CLI (use API instead)".into(),
                ))
            };

            let result = runner
                .run_custom_benchmark(run_id, dataset_path, sample_limit, model, infer_fn)
                .await?;

            println!("Benchmark: custom");
            println!("Score: {:.4}", result.score);
            println!("Samples: {}", result.samples_evaluated);
            println!("Duration: {:.2}s", result.duration_secs);
        }
        other => {
            // TODO: Wire standard benchmarks (perplexity, MMLU, etc.)
            runner.complete_run(run_id).await?;
            println!("Benchmark {other:?} is not yet implemented.");
            println!("Run ID: {run_id}");
            println!("Use the API for full eval support.");
        }
    }

    Ok(())
}

fn parse_benchmark(s: &str) -> synapse_types::error::Result<BenchmarkKind> {
    match s.to_lowercase().as_str() {
        "perplexity" | "ppl" => Ok(BenchmarkKind::Perplexity),
        "mmlu" => Ok(BenchmarkKind::Mmlu),
        "hellaswag" => Ok(BenchmarkKind::HellaSwag),
        "humaneval" | "human_eval" => Ok(BenchmarkKind::HumanEval),
        "custom" => Ok(BenchmarkKind::Custom),
        _ => Err(synapse_types::SynapseError::EvalError(format!(
            "Unknown benchmark: {s}. Options: perplexity, mmlu, hellaswag, humaneval, custom"
        ))),
    }
}
