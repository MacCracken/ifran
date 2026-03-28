//! Knowledge distillation training method.
//! Transfers knowledge from a teacher model to a smaller student model.

use ifran_types::training::TrainingJobConfig;

#[must_use]
pub fn build_args(config: &TrainingJobConfig) -> Vec<String> {
    vec![
        "--base-model".into(),
        config.base_model.clone(),
        "--dataset".into(),
        config.dataset.path.clone(),
        "--method".into(),
        "distillation".into(),
        "--lr".into(),
        config.hyperparams.learning_rate.to_string(),
        "--epochs".into(),
        config.hyperparams.epochs.to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ifran_types::training::*;

    #[test]
    fn distillation_build_args() {
        let cfg = TrainingJobConfig {
            base_model: "student-model".into(),
            dataset: DatasetConfig {
                path: "/data.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: None,
            },
            method: TrainingMethod::Distillation,
            hyperparams: HyperParams {
                learning_rate: 1e-4,
                epochs: 5,
                batch_size: 4,
                gradient_accumulation_steps: 4,
                warmup_steps: 100,
                weight_decay: 0.01,
                max_seq_length: 2048,
            },
            output_name: None,
            lora: None,
            max_steps: None,
            time_budget_secs: None,
        };
        let args = build_args(&cfg);
        assert!(args.contains(&"distillation".to_string()));
        assert!(args.contains(&"student-model".to_string()));
        assert!(args.contains(&"5".to_string())); // epochs
    }
}
