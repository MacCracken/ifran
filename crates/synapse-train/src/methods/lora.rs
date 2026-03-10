//! LoRA/QLoRA fine-tuning method — default configs and argument generation.

use synapse_types::training::{HyperParams, LoraConfig, TrainingJobConfig};

pub fn default_lora_config() -> LoraConfig {
    LoraConfig {
        rank: 16,
        alpha: 32.0,
        dropout: 0.05,
        target_modules: vec![
            "q_proj".into(), "k_proj".into(),
            "v_proj".into(), "o_proj".into(),
        ],
    }
}

pub fn default_hyperparams() -> HyperParams {
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

pub fn build_args(config: &TrainingJobConfig) -> Vec<String> {
    let lora = config.lora.as_ref().cloned().unwrap_or_else(default_lora_config);
    vec![
        "--base-model".into(), config.base_model.clone(),
        "--dataset".into(), config.dataset.path.clone(),
        "--method".into(), "lora".into(),
        "--lora-rank".into(), lora.rank.to_string(),
        "--lora-alpha".into(), lora.alpha.to_string(),
        "--lora-dropout".into(), lora.dropout.to_string(),
        "--lr".into(), config.hyperparams.learning_rate.to_string(),
        "--epochs".into(), config.hyperparams.epochs.to_string(),
        "--batch-size".into(), config.hyperparams.batch_size.to_string(),
    ]
}
