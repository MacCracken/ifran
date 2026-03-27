//! Model evaluation benchmark types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type EvalRunId = Uuid;

/// Available benchmark types.
#[non_exhaustive]
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
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_kind_serde_roundtrip() {
        let kinds = [
            BenchmarkKind::Perplexity,
            BenchmarkKind::Mmlu,
            BenchmarkKind::HellaSwag,
            BenchmarkKind::HumanEval,
            BenchmarkKind::Custom,
        ];
        for k in &kinds {
            let json = serde_json::to_string(k).unwrap();
            let back: BenchmarkKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*k, back);
        }
    }

    #[test]
    fn eval_status_serde_roundtrip() {
        let statuses = [
            EvalStatus::Queued,
            EvalStatus::Running,
            EvalStatus::Completed,
            EvalStatus::Failed,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: EvalStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*s, back);
        }
    }

    #[test]
    fn eval_config_serde() {
        let config = EvalConfig {
            model_name: "llama-7b".into(),
            benchmarks: vec![BenchmarkKind::Mmlu, BenchmarkKind::Perplexity],
            sample_limit: Some(100),
            dataset_path: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: EvalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.benchmarks.len(), 2);
        assert_eq!(back.sample_limit, Some(100));
    }

    #[test]
    fn eval_result_serde() {
        let result = EvalResult {
            run_id: Uuid::new_v4(),
            model_name: "test".into(),
            benchmark: BenchmarkKind::Mmlu,
            score: 0.85,
            details: Some(serde_json::json!({"category": "stem", "accuracy": 0.9})),
            samples_evaluated: 500,
            duration_secs: 120.5,
            evaluated_at: Utc::now(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: EvalResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_name, "test");
        assert!((back.score - 0.85).abs() < f64::EPSILON);
        assert!(back.details.is_some());
    }

    #[test]
    fn benchmark_kind_invalid_json() {
        let result = serde_json::from_str::<BenchmarkKind>("\"invalid\"");
        assert!(result.is_err());
    }

    #[test]
    fn eval_status_invalid_json() {
        let result = serde_json::from_str::<EvalStatus>("\"invalid\"");
        assert!(result.is_err());
    }

    #[test]
    fn eval_config_with_dataset_path() {
        let config = EvalConfig {
            model_name: "test".into(),
            benchmarks: vec![BenchmarkKind::Custom],
            sample_limit: None,
            dataset_path: Some("/data/eval.jsonl".into()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: EvalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dataset_path, Some("/data/eval.jsonl".into()));
    }

    #[test]
    fn eval_result_no_details() {
        let result = EvalResult {
            run_id: Uuid::new_v4(),
            model_name: "test".into(),
            benchmark: BenchmarkKind::Perplexity,
            score: 12.5,
            details: None,
            samples_evaluated: 100,
            duration_secs: 30.0,
            evaluated_at: Utc::now(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: EvalResult = serde_json::from_str(&json).unwrap();
        assert!(back.details.is_none());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_benchmark_kind() -> impl Strategy<Value = BenchmarkKind> {
        prop_oneof![
            Just(BenchmarkKind::Perplexity),
            Just(BenchmarkKind::Mmlu),
            Just(BenchmarkKind::HellaSwag),
            Just(BenchmarkKind::HumanEval),
            Just(BenchmarkKind::Custom),
        ]
    }

    fn arb_eval_status() -> impl Strategy<Value = EvalStatus> {
        prop_oneof![
            Just(EvalStatus::Queued),
            Just(EvalStatus::Running),
            Just(EvalStatus::Completed),
            Just(EvalStatus::Failed),
        ]
    }

    proptest! {
        #[test]
        fn benchmark_kind_roundtrips(k in arb_benchmark_kind()) {
            let json = serde_json::to_string(&k).unwrap();
            let back: BenchmarkKind = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(k, back);
        }

        #[test]
        fn eval_status_roundtrips(s in arb_eval_status()) {
            let json = serde_json::to_string(&s).unwrap();
            let back: EvalStatus = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(s, back);
        }

        #[test]
        fn eval_result_score_preserved(
            score in -1000.0f64..1000.0,
            samples in 0u64..1_000_000,
        ) {
            let result = EvalResult {
                run_id: Uuid::new_v4(),
                model_name: "test".into(),
                benchmark: BenchmarkKind::Custom,
                score,
                details: None,
                samples_evaluated: samples,
                duration_secs: 0.0,
                evaluated_at: Utc::now(),
            };
            let json = serde_json::to_string(&result).unwrap();
            let back: EvalResult = serde_json::from_str(&json).unwrap();
            // JSON roundtrip may introduce tiny floating-point drift
            prop_assert!((result.score - back.score).abs() < 1e-10);
            prop_assert_eq!(result.samples_evaluated, back.samples_evaluated);
        }
    }
}
