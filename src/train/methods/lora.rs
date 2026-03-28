//! LoRA/QLoRA fine-tuning method — default configs and argument generation.

use crate::types::training::{HyperParams, LoraConfig, TrainingJobConfig};

#[must_use]
pub fn default_lora_config() -> LoraConfig {
    LoraConfig {
        rank: 16,
        alpha: 32.0,
        dropout: 0.05,
        target_modules: vec![
            "q_proj".into(),
            "k_proj".into(),
            "v_proj".into(),
            "o_proj".into(),
        ],
    }
}

#[must_use]
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

#[must_use]
pub fn build_args(config: &TrainingJobConfig) -> Vec<String> {
    let lora = config
        .lora
        .as_ref()
        .cloned()
        .unwrap_or_else(default_lora_config);
    vec![
        "--base-model".into(),
        config.base_model.clone(),
        "--dataset".into(),
        config.dataset.path.clone(),
        "--method".into(),
        "lora".into(),
        "--lora-rank".into(),
        lora.rank.to_string(),
        "--lora-alpha".into(),
        lora.alpha.to_string(),
        "--lora-dropout".into(),
        lora.dropout.to_string(),
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
    use crate::types::training::*;

    #[test]
    fn lora_config_defaults() {
        let cfg = default_lora_config();
        assert_eq!(cfg.rank, 16);
        assert_eq!(cfg.alpha, 32.0);
        assert_eq!(cfg.target_modules.len(), 4);
    }

    #[test]
    fn hyperparams_defaults() {
        let hp = default_hyperparams();
        assert_eq!(hp.epochs, 3);
        assert_eq!(hp.batch_size, 4);
        assert!(hp.learning_rate > 0.0);
    }

    fn test_config() -> TrainingJobConfig {
        TrainingJobConfig {
            base_model: "meta-llama/Llama-2-7b".into(),
            dataset: DatasetConfig {
                path: "/data/train.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: None,
            },
            method: TrainingMethod::Lora,
            hyperparams: default_hyperparams(),
            output_name: None,
            lora: Some(default_lora_config()),
            max_steps: None,
            time_budget_secs: None,
        }
    }

    #[test]
    fn build_args_includes_model_and_dataset() {
        let args = build_args(&test_config());
        assert!(args.contains(&"--base-model".to_string()));
        assert!(args.contains(&"meta-llama/Llama-2-7b".to_string()));
        assert!(args.contains(&"--dataset".to_string()));
        assert!(args.contains(&"--lora-rank".to_string()));
        assert!(args.contains(&"16".to_string()));
    }

    #[test]
    fn build_args_uses_default_lora_when_none() {
        let mut cfg = test_config();
        cfg.lora = None;
        let args = build_args(&cfg);
        // Should still have lora-rank from the default
        assert!(args.contains(&"--lora-rank".to_string()));
    }
}
