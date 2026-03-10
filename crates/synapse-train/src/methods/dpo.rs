//! Direct Preference Optimization (DPO) training method.

use synapse_types::training::{HyperParams, TrainingJobConfig};

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

pub fn build_args(config: &TrainingJobConfig) -> Vec<String> {
    vec![
        "--base-model".into(), config.base_model.clone(),
        "--dataset".into(), config.dataset.path.clone(),
        "--method".into(), "dpo".into(),
        "--lr".into(), config.hyperparams.learning_rate.to_string(),
        "--epochs".into(), config.hyperparams.epochs.to_string(),
        "--batch-size".into(), config.hyperparams.batch_size.to_string(),
        "--beta".into(), "0.1".into(),
    ]
}
