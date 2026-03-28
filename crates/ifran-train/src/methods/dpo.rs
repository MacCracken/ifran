//! Direct Preference Optimization (DPO) training method.

use ifran_types::training::{HyperParams, TrainingJobConfig};

#[must_use]
pub fn default_hyperparams() -> HyperParams {
    HyperParams {
        learning_rate: 5e-6,
        epochs: 1,
        batch_size: 2,
        gradient_accumulation_steps: 8,
        warmup_steps: 50,
        weight_decay: 0.0,
        max_seq_length: 1024,
    }
}

#[must_use]
pub fn build_args(config: &TrainingJobConfig) -> Vec<String> {
    vec![
        "--base-model".into(),
        config.base_model.clone(),
        "--dataset".into(),
        config.dataset.path.clone(),
        "--method".into(),
        "dpo".into(),
        "--lr".into(),
        config.hyperparams.learning_rate.to_string(),
        "--epochs".into(),
        config.hyperparams.epochs.to_string(),
        "--batch-size".into(),
        config.hyperparams.batch_size.to_string(),
        "--beta".into(),
        "0.1".into(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ifran_types::training::*;

    #[test]
    fn dpo_hyperparams() {
        let hp = default_hyperparams();
        assert_eq!(hp.epochs, 1);
        assert_eq!(hp.batch_size, 2);
        assert!(hp.learning_rate < 1e-4); // DPO uses lower LR
    }

    #[test]
    fn dpo_build_args() {
        let cfg = TrainingJobConfig {
            base_model: "model".into(),
            dataset: DatasetConfig {
                path: "/data.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: None,
            },
            method: TrainingMethod::Dpo,
            hyperparams: default_hyperparams(),
            output_name: None,
            lora: None,
            max_steps: None,
            time_budget_secs: None,
        };
        let args = build_args(&cfg);
        assert!(args.contains(&"dpo".to_string()));
        assert!(args.contains(&"--beta".to_string()));
    }
}
