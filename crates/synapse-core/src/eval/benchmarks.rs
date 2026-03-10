//! Benchmark implementations.
//!
//! Each benchmark takes inference results and computes a score.
//! MVP: perplexity and custom (exact-match) benchmarks.

use serde::Deserialize;
use synapse_types::eval::BenchmarkKind;

/// A single evaluation sample.
#[derive(Debug, Clone, Deserialize)]
pub struct EvalSample {
    pub prompt: String,
    pub expected: String,
    /// Multiple-choice options (for MMLU-style benchmarks).
    pub choices: Option<Vec<String>>,
    /// Correct choice index.
    pub answer_index: Option<usize>,
}

/// Computed score from a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkScore {
    pub kind: BenchmarkKind,
    pub score: f64,
    pub samples_evaluated: u64,
}

/// Score a custom benchmark: exact-match accuracy.
///
/// Compares model outputs to expected outputs, returning the fraction
/// of exact matches (case-insensitive, trimmed).
pub fn score_exact_match(predictions: &[(String, String)]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .filter(|(pred, expected)| pred.trim().eq_ignore_ascii_case(expected.trim()))
        .count();
    correct as f64 / predictions.len() as f64
}

/// Score a custom benchmark: contains-match accuracy.
///
/// Checks if the model output contains the expected answer.
pub fn score_contains_match(predictions: &[(String, String)]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .filter(|(pred, expected)| pred.to_lowercase().contains(&expected.to_lowercase()))
        .count();
    correct as f64 / predictions.len() as f64
}

/// Load eval samples from a JSONL file.
pub fn load_samples(
    path: &str,
    limit: Option<usize>,
) -> synapse_types::error::Result<Vec<EvalSample>> {
    let content = std::fs::read_to_string(path)?;
    let mut samples = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let sample: EvalSample = serde_json::from_str(line).map_err(|e| {
            synapse_types::SynapseError::EvalError(format!("Invalid eval sample: {e}"))
        })?;
        samples.push(sample);
        if let Some(max) = limit {
            if samples.len() >= max {
                break;
            }
        }
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_scoring() {
        let preds = vec![
            ("Paris".into(), "Paris".into()),
            ("london".into(), "London".into()),
            ("wrong".into(), "Berlin".into()),
        ];
        let score = score_exact_match(&preds);
        assert!((score - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn contains_match_scoring() {
        let preds = vec![
            ("The answer is Paris, the capital".into(), "Paris".into()),
            ("I think it's Berlin".into(), "Berlin".into()),
            ("No idea".into(), "Tokyo".into()),
        ];
        let score = score_contains_match(&preds);
        assert!((score - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn empty_predictions() {
        assert_eq!(score_exact_match(&[]), 0.0);
        assert_eq!(score_contains_match(&[]), 0.0);
    }
}
