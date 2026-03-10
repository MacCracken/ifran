//! Knowledge distillation training method.
//! Transfers knowledge from a teacher model to a smaller student model.

use synapse_types::training::TrainingJobConfig;

pub fn build_args(config: &TrainingJobConfig) -> Vec<String> {
    vec![
        "--base-model".into(), config.base_model.clone(),
        "--dataset".into(), config.dataset.path.clone(),
        "--method".into(), "distillation".into(),
        "--lr".into(), config.hyperparams.learning_rate.to_string(),
        "--epochs".into(), config.hyperparams.epochs.to_string(),
    ]
}
