//! Search space engine for experiment hyperparameter exploration.

use crate::types::experiment::{ParamRange, SearchStrategy};
use crate::types::training::HyperParams;
use rand::RngExt;
use rand_core::UnwrapErr;

/// Generates trial hyperparameter configurations from a search space.
pub struct SearchSpace {
    base: HyperParams,
    ranges: Vec<ParamRange>,
    strategy: SearchStrategy,
}

impl SearchSpace {
    pub fn new(base: HyperParams, ranges: Vec<ParamRange>, strategy: SearchStrategy) -> Self {
        Self {
            base,
            ranges,
            strategy,
        }
    }

    /// Generate all trial hyperparameter combinations.
    pub fn generate_trials(&self) -> Vec<HyperParams> {
        match &self.strategy {
            SearchStrategy::Grid => self.generate_grid(),
            SearchStrategy::Random { n_trials } => self.generate_random(*n_trials),
        }
    }

    fn generate_grid(&self) -> Vec<HyperParams> {
        if self.ranges.is_empty() {
            return vec![self.base.clone()];
        }

        // Expand each range into concrete values
        let expanded: Vec<Vec<(String, f64)>> = self
            .ranges
            .iter()
            .map(|r| {
                r.values
                    .expand()
                    .into_iter()
                    .map(|v| (r.name.clone(), v))
                    .collect()
            })
            .collect();

        // Cartesian product
        let mut combos: Vec<Vec<(String, f64)>> = vec![vec![]];
        for dimension in &expanded {
            let mut new_combos = Vec::new();
            for existing in &combos {
                for item in dimension {
                    let mut combo = existing.clone();
                    combo.push(item.clone());
                    new_combos.push(combo);
                }
            }
            combos = new_combos;
        }

        combos
            .into_iter()
            .map(|params| {
                let mut hp = self.base.clone();
                for (name, value) in params {
                    apply_param(&mut hp, &name, value);
                }
                hp
            })
            .collect()
    }

    fn generate_random(&self, n_trials: u32) -> Vec<HyperParams> {
        if self.ranges.is_empty() {
            return vec![self.base.clone()];
        }

        let mut rng = UnwrapErr(rand::rng());
        (0..n_trials)
            .map(|_| {
                let mut hp = self.base.clone();
                for range in &self.ranges {
                    let values = range.values.expand();
                    if !values.is_empty() {
                        let idx = rng.random_range(0..values.len());
                        apply_param(&mut hp, &range.name, values[idx]);
                    }
                }
                hp
            })
            .collect()
    }
}

/// Map a parameter name to the corresponding HyperParams field.
pub fn apply_param(hp: &mut HyperParams, name: &str, value: f64) {
    match name {
        "learning_rate" => hp.learning_rate = value,
        "epochs" => hp.epochs = value as u32,
        "batch_size" => hp.batch_size = value as u32,
        "gradient_accumulation_steps" => hp.gradient_accumulation_steps = value as u32,
        "warmup_steps" => hp.warmup_steps = value as u32,
        "weight_decay" => hp.weight_decay = value,
        "max_seq_length" => hp.max_seq_length = value as u32,
        _ => tracing::warn!(param = name, "Unknown hyperparameter — ignoring"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::experiment::{ParamValues, SearchStrategy};

    fn base_hp() -> HyperParams {
        HyperParams {
            learning_rate: 2e-4,
            epochs: 3,
            batch_size: 4,
            gradient_accumulation_steps: 4,
            warmup_steps: 100,
            weight_decay: 0.01,
            max_seq_length: 2048,
        }
    }

    #[test]
    fn grid_empty_ranges() {
        let ss = SearchSpace::new(base_hp(), vec![], SearchStrategy::Grid);
        let trials = ss.generate_trials();
        assert_eq!(trials.len(), 1);
        assert!((trials[0].learning_rate - 2e-4).abs() < f64::EPSILON);
    }

    #[test]
    fn grid_single_param() {
        let ranges = vec![ParamRange {
            name: "learning_rate".into(),
            values: ParamValues::Discrete {
                values: vec![1e-5, 5e-5, 1e-4],
            },
        }];
        let ss = SearchSpace::new(base_hp(), ranges, SearchStrategy::Grid);
        let trials = ss.generate_trials();
        assert_eq!(trials.len(), 3);
        assert!((trials[0].learning_rate - 1e-5).abs() < f64::EPSILON);
        assert!((trials[1].learning_rate - 5e-5).abs() < f64::EPSILON);
        assert!((trials[2].learning_rate - 1e-4).abs() < f64::EPSILON);
    }

    #[test]
    fn grid_two_params_cartesian() {
        let ranges = vec![
            ParamRange {
                name: "learning_rate".into(),
                values: ParamValues::Discrete {
                    values: vec![1e-5, 1e-4],
                },
            },
            ParamRange {
                name: "weight_decay".into(),
                values: ParamValues::Discrete {
                    values: vec![0.0, 0.01, 0.1],
                },
            },
        ];
        let ss = SearchSpace::new(base_hp(), ranges, SearchStrategy::Grid);
        let trials = ss.generate_trials();
        // 2 x 3 = 6 combinations
        assert_eq!(trials.len(), 6);
    }

    #[test]
    fn grid_range_values() {
        let ranges = vec![ParamRange {
            name: "weight_decay".into(),
            values: ParamValues::Range {
                min: 0.0,
                max: 0.1,
                step: 0.05,
            },
        }];
        let ss = SearchSpace::new(base_hp(), ranges, SearchStrategy::Grid);
        let trials = ss.generate_trials();
        assert_eq!(trials.len(), 3); // 0.0, 0.05, 0.1
    }

    #[test]
    fn random_generates_n_trials() {
        let ranges = vec![ParamRange {
            name: "learning_rate".into(),
            values: ParamValues::Discrete {
                values: vec![1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            },
        }];
        let ss = SearchSpace::new(base_hp(), ranges, SearchStrategy::Random { n_trials: 7 });
        let trials = ss.generate_trials();
        assert_eq!(trials.len(), 7);
    }

    #[test]
    fn apply_param_learning_rate() {
        let mut hp = base_hp();
        apply_param(&mut hp, "learning_rate", 1e-5);
        assert!((hp.learning_rate - 1e-5).abs() < f64::EPSILON);
    }

    #[test]
    fn apply_param_epochs() {
        let mut hp = base_hp();
        apply_param(&mut hp, "epochs", 5.0);
        assert_eq!(hp.epochs, 5);
    }

    #[test]
    fn apply_param_weight_decay() {
        let mut hp = base_hp();
        apply_param(&mut hp, "weight_decay", 0.1);
        assert!((hp.weight_decay - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn apply_param_batch_size() {
        let mut hp = base_hp();
        apply_param(&mut hp, "batch_size", 8.0);
        assert_eq!(hp.batch_size, 8);
    }

    #[test]
    fn apply_param_unknown_ignored() {
        let mut hp = base_hp();
        let original_lr = hp.learning_rate;
        apply_param(&mut hp, "nonexistent_param", 999.0);
        assert!((hp.learning_rate - original_lr).abs() < f64::EPSILON);
    }
}
