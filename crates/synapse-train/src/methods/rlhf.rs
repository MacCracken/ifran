//! RLHF training method — placeholder for PPO-based alignment.
//! Requires a reward model; orchestrated via the training executor.

use synapse_types::training::TrainingJobConfig;

pub fn build_args(config: &TrainingJobConfig) -> Vec<String> {
    vec![
        "--base-model".into(), config.base_model.clone(),
        "--dataset".into(), config.dataset.path.clone(),
        "--method".into(), "rlhf".into(),
        "--lr".into(), config.hyperparams.learning_rate.to_string(),
        "--epochs".into(), config.hyperparams.epochs.to_string(),
    ]
}
