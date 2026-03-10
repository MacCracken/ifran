//! Model evaluation benchmark types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type EvalRunId = Uuid;

/// Available benchmark types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkKind {
    /// Perplexity on a text corpus.
    Perplexity,
    /// Multiple-choice accuracy (MMLU-style).
    Mmlu,
    /// Common-sense reasoning (HellaSwag-style).
    HellaSwag,
    /// Code generation (HumanEval-style pass@k).
    HumanEval,
    /// User-supplied eval dataset (prompt → expected output).
    Custom,
}

/// Configuration for an evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    pub model_name: String,
    pub benchmarks: Vec<BenchmarkKind>,
    /// Max samples per benchmark (None = use all).
    pub sample_limit: Option<usize>,
    /// Path to custom eval dataset (JSONL with prompt/expected fields).
    pub dataset_path: Option<String>,
}

/// Result of a single benchmark evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub run_id: EvalRunId,
    pub model_name: String,
    pub benchmark: BenchmarkKind,
    /// Primary metric: accuracy (0.0–1.0) for classification, perplexity value for perplexity.
    pub score: f64,
    /// Additional metrics (e.g. per-category scores).
    pub details: Option<serde_json::Value>,
    pub samples_evaluated: u64,
    pub duration_secs: f64,
    pub evaluated_at: DateTime<Utc>,
}

/// Status of an eval run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalStatus {
    Queued,
    Running,
    Completed,
    Failed,
}
