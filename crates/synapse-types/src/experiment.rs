//! Types for the autonomous experiment system (AutoResearch-inspired).

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::eval::BenchmarkKind;
use crate::training::{HyperParams, TrainingMethod};

pub type ExperimentId = Uuid;
pub type TrialId = Uuid;

/// Whether the objective should be minimized or maximized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Minimize,
    Maximize,
}

impl Direction {
    /// Returns true if `a` is better than `b` according to this direction.
    pub fn is_better(&self, a: f64, b: f64) -> bool {
        match self {
            Direction::Minimize => a < b,
            Direction::Maximize => a > b,
        }
    }
}

/// Search strategy for exploring the hyperparameter space.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "strategy")]
pub enum SearchStrategy {
    Grid,
    Random { n_trials: u32 },
}

/// A single parameter range in the search space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamRange {
    pub name: String,
    #[serde(flatten)]
    pub values: ParamValues,
}

/// Possible value specifications for a search parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValues {
    Discrete { values: Vec<f64> },
    Range { min: f64, max: f64, step: f64 },
}

impl ParamValues {
    /// Expand this parameter specification into concrete values.
    pub fn expand(&self) -> Vec<f64> {
        match self {
            ParamValues::Discrete { values } => values.clone(),
            ParamValues::Range { min, max, step } => {
                let mut vals = Vec::new();
                let mut v = *min;
                while v <= *max + f64::EPSILON {
                    vals.push(v);
                    v += step;
                }
                vals
            }
        }
    }
}

/// The objective metric and optimization direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentObjective {
    pub metric: BenchmarkKind,
    pub direction: Direction,
}

/// Full specification for an experiment run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentProgram {
    pub name: String,
    pub base_model: String,
    pub dataset_path: String,
    #[serde(default = "default_dataset_format")]
    pub dataset_format: String,
    pub method: TrainingMethod,
    /// Per-trial wall-clock budget in seconds.
    pub time_budget_secs: u64,
    pub objective: ExperimentObjective,
    pub search: SearchStrategy,
    pub search_space: Vec<ParamRange>,
    pub base_hyperparams: HyperParams,
    /// Maximum number of trials (caps grid/random).
    #[serde(default)]
    pub max_trials: Option<u32>,
    /// Sample limit for evaluation.
    #[serde(default)]
    pub eval_sample_limit: Option<usize>,
}

fn default_dataset_format() -> String {
    "jsonl".into()
}

/// Status of an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Running,
    Completed,
    Stopped,
    Failed,
}

/// Status of a single trial within an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    Training,
    Evaluating,
    Completed,
    Failed,
}

/// Result of a single trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub trial_id: TrialId,
    pub experiment_id: ExperimentId,
    pub trial_number: u32,
    pub hyperparams: HyperParams,
    pub train_loss: Option<f64>,
    pub eval_score: Option<f64>,
    pub status: TrialStatus,
    pub duration_secs: Option<f64>,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub checkpoint_path: Option<String>,
    pub is_best: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_serde_roundtrip() {
        for d in [Direction::Minimize, Direction::Maximize] {
            let json = serde_json::to_string(&d).unwrap();
            let back: Direction = serde_json::from_str(&json).unwrap();
            assert_eq!(d, back);
        }
    }

    #[test]
    fn direction_is_better() {
        assert!(Direction::Minimize.is_better(1.0, 2.0));
        assert!(!Direction::Minimize.is_better(2.0, 1.0));
        assert!(Direction::Maximize.is_better(2.0, 1.0));
        assert!(!Direction::Maximize.is_better(1.0, 2.0));
    }

    #[test]
    fn experiment_status_serde_roundtrip() {
        for s in [
            ExperimentStatus::Running,
            ExperimentStatus::Completed,
            ExperimentStatus::Stopped,
            ExperimentStatus::Failed,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: ExperimentStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn trial_status_serde_roundtrip() {
        for s in [
            TrialStatus::Training,
            TrialStatus::Evaluating,
            TrialStatus::Completed,
            TrialStatus::Failed,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: TrialStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn param_values_discrete_expand() {
        let pv = ParamValues::Discrete {
            values: vec![1e-5, 5e-5, 1e-4],
        };
        let expanded = pv.expand();
        assert_eq!(expanded.len(), 3);
        assert!((expanded[0] - 1e-5).abs() < f64::EPSILON);
    }

    #[test]
    fn param_values_range_expand() {
        let pv = ParamValues::Range {
            min: 0.0,
            max: 1.0,
            step: 0.5,
        };
        let expanded = pv.expand();
        assert_eq!(expanded.len(), 3);
        assert!((expanded[2] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn search_strategy_grid_serde() {
        let s = SearchStrategy::Grid;
        let json = serde_json::to_string(&s).unwrap();
        let back: SearchStrategy = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, SearchStrategy::Grid));
    }

    #[test]
    fn search_strategy_random_serde() {
        let s = SearchStrategy::Random { n_trials: 10 };
        let json = serde_json::to_string(&s).unwrap();
        let back: SearchStrategy = serde_json::from_str(&json).unwrap();
        match back {
            SearchStrategy::Random { n_trials } => assert_eq!(n_trials, 10),
            _ => panic!("expected Random"),
        }
    }

    #[test]
    fn experiment_program_serde() {
        let program = ExperimentProgram {
            name: "lr-sweep".into(),
            base_model: "llama-8b".into(),
            dataset_path: "/data/finetune.jsonl".into(),
            dataset_format: "jsonl".into(),
            method: TrainingMethod::Lora,
            time_budget_secs: 300,
            objective: ExperimentObjective {
                metric: BenchmarkKind::Perplexity,
                direction: Direction::Minimize,
            },
            search: SearchStrategy::Grid,
            search_space: vec![ParamRange {
                name: "learning_rate".into(),
                values: ParamValues::Discrete {
                    values: vec![1e-5, 5e-5, 1e-4],
                },
            }],
            base_hyperparams: crate::training::HyperParams {
                learning_rate: 2e-4,
                epochs: 3,
                batch_size: 4,
                gradient_accumulation_steps: 4,
                warmup_steps: 100,
                weight_decay: 0.01,
                max_seq_length: 2048,
            },
            max_trials: Some(10),
            eval_sample_limit: Some(100),
        };
        let json = serde_json::to_string(&program).unwrap();
        let back: ExperimentProgram = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "lr-sweep");
        assert_eq!(back.time_budget_secs, 300);
        assert_eq!(back.search_space.len(), 1);
    }

    #[test]
    fn trial_result_serde() {
        let trial = TrialResult {
            trial_id: Uuid::new_v4(),
            experiment_id: Uuid::new_v4(),
            trial_number: 1,
            hyperparams: crate::training::HyperParams {
                learning_rate: 1e-4,
                epochs: 3,
                batch_size: 4,
                gradient_accumulation_steps: 4,
                warmup_steps: 100,
                weight_decay: 0.01,
                max_seq_length: 2048,
            },
            train_loss: Some(0.42),
            eval_score: Some(5.23),
            status: TrialStatus::Completed,
            duration_secs: Some(295.0),
            started_at: Some(chrono::Utc::now()),
            completed_at: Some(chrono::Utc::now()),
            checkpoint_path: Some("/checkpoints/trial-1".into()),
            is_best: true,
        };
        let json = serde_json::to_string(&trial).unwrap();
        let back: TrialResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.trial_number, 1);
        assert!(back.is_best);
        assert_eq!(back.eval_score, Some(5.23));
    }
}
