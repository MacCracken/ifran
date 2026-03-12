//! RLHF training method — placeholder for PPO-based alignment.
//! Requires a reward model; orchestrated via the training executor.

use synapse_types::training::TrainingJobConfig;

pub fn build_args(config: &TrainingJobConfig) -> Vec<String> {
    vec![
        "--base-model".into(),
        config.base_model.clone(),
        "--dataset".into(),
        config.dataset.path.clone(),
        "--method".into(),
        "rlhf".into(),
        "--lr".into(),
        config.hyperparams.learning_rate.to_string(),
        "--epochs".into(),
        config.hyperparams.epochs.to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_types::training::*;

    #[test]
    fn rlhf_build_args() {
        let cfg = TrainingJobConfig {
            base_model: "model".into(),
            dataset: DatasetConfig {
                path: "/data.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: None,
            },
            method: TrainingMethod::Rlhf,
            hyperparams: HyperParams {
                learning_rate: 1e-5,
                epochs: 2,
                batch_size: 2,
                gradient_accumulation_steps: 8,
                warmup_steps: 50,
                weight_decay: 0.0,
                max_seq_length: 1024,
            },
            output_name: None,
            lora: None,
            max_steps: None,
            time_budget_secs: None,
        };
        let args = build_args(&cfg);
        assert!(args.contains(&"rlhf".to_string()));
        assert!(args.contains(&"--lr".to_string()));
    }
}
