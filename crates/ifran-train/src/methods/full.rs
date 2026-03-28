//! Full parameter fine-tuning method.

use ifran_types::training::{HyperParams, TrainingJobConfig};

#[must_use]
pub fn default_hyperparams() -> HyperParams {
    HyperParams {
        learning_rate: 5e-5,
        epochs: 3,
        batch_size: 2,
        gradient_accumulation_steps: 8,
        warmup_steps: 200,
        weight_decay: 0.01,
        max_seq_length: 2048,
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
        "full".into(),
        "--lr".into(),
        config.hyperparams.learning_rate.to_string(),
        "--epochs".into(),
        config.hyperparams.epochs.to_string(),
        "--batch-size".into(),
        config.hyperparams.batch_size.to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ifran_types::training::*;

    #[test]
    fn full_hyperparams() {
        let hp = default_hyperparams();
        assert_eq!(hp.epochs, 3);
        assert_eq!(hp.gradient_accumulation_steps, 8);
    }

    #[test]
    fn full_build_args() {
        let cfg = TrainingJobConfig {
            base_model: "model".into(),
            dataset: DatasetConfig {
                path: "/data.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: None,
            },
            method: TrainingMethod::FullFineTune,
            hyperparams: default_hyperparams(),
            output_name: None,
            lora: None,
            max_steps: None,
            time_budget_secs: None,
        };
        let args = build_args(&cfg);
        assert!(args.contains(&"full".to_string()));
        assert!(args.contains(&"--base-model".to_string()));
    }
}
