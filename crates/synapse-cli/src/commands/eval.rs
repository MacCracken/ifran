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

    let run_id = runner.create_run(config, "default").await?;
    runner.start_run(run_id).await?;

    let dataset_path = dataset.ok_or_else(|| {
        synapse_types::SynapseError::EvalError(
            "Dataset path required (--dataset <path>). Provide a JSONL file with prompt/expected fields.".into(),
        )
    })?;

    // Build inference function via HTTP to local API server
    let http_client = synapse_core::pull::downloader::build_client()?;
    let infer_fn = |prompt: String| {
        let client = http_client.clone();
        async move {
            let resp = client
                .post("http://127.0.0.1:8420/inference")
                .json(&serde_json::json!({
                    "model": "default",
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": 0.0,
                }))
                .send()
                .await
                .map_err(|e| {
                    synapse_types::SynapseError::EvalError(format!(
                        "Inference request failed (is the API server running?): {e}"
                    ))
                })?;

            if !resp.status().is_success() {
                return Err(synapse_types::SynapseError::EvalError(format!(
                    "Inference returned HTTP {}",
                    resp.status()
                )));
            }

            let body: serde_json::Value = resp.json().await.map_err(|e| {
                synapse_types::SynapseError::EvalError(format!("Invalid inference response: {e}"))
            })?;

            body["text"].as_str().map(String::from).ok_or_else(|| {
                synapse_types::SynapseError::EvalError(
                    "No 'text' field in inference response".into(),
                )
            })
        }
    };

    println!("Running {kind:?} benchmark on '{model}'...");
    println!("Dataset: {dataset_path}");
    if let Some(limit) = sample_limit {
        println!("Sample limit: {limit}");
    }
    println!();

    match runner
        .run_benchmark(run_id, kind, dataset_path, sample_limit, model, &infer_fn)
        .await
    {
        Ok(result) => {
            runner.complete_run(run_id).await?;
            println!("Benchmark: {kind:?}");
            if kind == BenchmarkKind::Perplexity {
                println!("Perplexity: {:.2} (lower is better)", result.score);
            } else {
                println!("Score: {:.4} ({:.1}%)", result.score, result.score * 100.0);
            }
            println!("Samples evaluated: {}", result.samples_evaluated);
            println!("Duration: {:.2}s", result.duration_secs);
        }
        Err(e) => {
            runner.fail_run(run_id, e.to_string()).await?;
            eprintln!("Benchmark failed: {e}");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_benchmark_perplexity() {
        assert_eq!(
            parse_benchmark("perplexity").unwrap(),
            BenchmarkKind::Perplexity
        );
        assert_eq!(parse_benchmark("ppl").unwrap(), BenchmarkKind::Perplexity);
    }

    #[test]
    fn parse_benchmark_mmlu() {
        assert_eq!(parse_benchmark("mmlu").unwrap(), BenchmarkKind::Mmlu);
    }

    #[test]
    fn parse_benchmark_hellaswag() {
        assert_eq!(
            parse_benchmark("hellaswag").unwrap(),
            BenchmarkKind::HellaSwag
        );
    }

    #[test]
    fn parse_benchmark_humaneval() {
        assert_eq!(
            parse_benchmark("humaneval").unwrap(),
            BenchmarkKind::HumanEval
        );
        assert_eq!(
            parse_benchmark("human_eval").unwrap(),
            BenchmarkKind::HumanEval
        );
    }

    #[test]
    fn parse_benchmark_custom() {
        assert_eq!(parse_benchmark("custom").unwrap(), BenchmarkKind::Custom);
    }

    #[test]
    fn parse_benchmark_case_insensitive() {
        assert_eq!(
            parse_benchmark("PERPLEXITY").unwrap(),
            BenchmarkKind::Perplexity
        );
        assert_eq!(parse_benchmark("Mmlu").unwrap(), BenchmarkKind::Mmlu);
        assert_eq!(
            parse_benchmark("HellaSwag").unwrap(),
            BenchmarkKind::HellaSwag
        );
        assert_eq!(
            parse_benchmark("HumanEval").unwrap(),
            BenchmarkKind::HumanEval
        );
    }

    #[test]
    fn parse_benchmark_invalid() {
        assert!(parse_benchmark("nonexistent").is_err());
        assert!(parse_benchmark("").is_err());
    }
}
