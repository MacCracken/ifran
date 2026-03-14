use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type TrainingJobId = Uuid;

/// Training method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMethod {
    FullFineTune,
    Lora,
    Qlora,
    Dpo,
    Rlhf,
    Distillation,
}

/// Configuration for a training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJobConfig {
    pub base_model: String,
    pub dataset: DatasetConfig,
    pub method: TrainingMethod,
    pub hyperparams: HyperParams,
    pub output_name: Option<String>,
    pub lora: Option<LoraConfig>,
    /// Maximum training steps (overrides epochs when set).
    #[serde(default)]
    pub max_steps: Option<u64>,
    /// Wall-clock time budget in seconds. Training stops after this duration.
    #[serde(default)]
    pub time_budget_secs: Option<u64>,
}

/// Dataset configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub path: String,
    pub format: DatasetFormat,
    pub split: Option<String>,
    pub max_samples: Option<usize>,
}

/// Supported dataset formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetFormat {
    Jsonl,
    Parquet,
    Csv,
    HuggingFace,
}

/// Training hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParams {
    pub learning_rate: f64,
    pub epochs: u32,
    pub batch_size: u32,
    pub gradient_accumulation_steps: u32,
    pub warmup_steps: u32,
    pub weight_decay: f64,
    pub max_seq_length: u32,
}

impl HyperParams {
    /// Validate hyperparameters, returning an error if any are invalid.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(crate::SynapseError::TrainingError(
                "learning_rate must be positive".into(),
            ));
        }
        if self.epochs == 0 {
            return Err(crate::SynapseError::TrainingError(
                "epochs must be at least 1".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(crate::SynapseError::TrainingError(
                "batch_size must be at least 1".into(),
            ));
        }
        if self.max_seq_length == 0 {
            return Err(crate::SynapseError::TrainingError(
                "max_seq_length must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// LoRA-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: u32,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
}

/// Training job status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStatus {
    Queued,
    Preparing,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Checkpoint metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub step: u64,
    pub epoch: f32,
    pub loss: f64,
    pub path: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_method_serde_roundtrip() {
        let methods = [
            TrainingMethod::FullFineTune,
            TrainingMethod::Lora,
            TrainingMethod::Qlora,
            TrainingMethod::Dpo,
            TrainingMethod::Rlhf,
            TrainingMethod::Distillation,
        ];
        for m in &methods {
            let json = serde_json::to_string(m).unwrap();
            let back: TrainingMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(*m, back);
        }
    }

    #[test]
    fn training_method_json_values() {
        assert_eq!(
            serde_json::to_string(&TrainingMethod::FullFineTune).unwrap(),
            "\"full_fine_tune\""
        );
        assert_eq!(
            serde_json::to_string(&TrainingMethod::Lora).unwrap(),
            "\"lora\""
        );
        assert_eq!(
            serde_json::to_string(&TrainingMethod::Qlora).unwrap(),
            "\"qlora\""
        );
    }

    #[test]
    fn dataset_format_serde_roundtrip() {
        let formats = [
            DatasetFormat::Jsonl,
            DatasetFormat::Parquet,
            DatasetFormat::Csv,
            DatasetFormat::HuggingFace,
        ];
        for f in &formats {
            let json = serde_json::to_string(f).unwrap();
            let back: DatasetFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(*f, back);
        }
    }

    #[test]
    fn training_status_serde_roundtrip() {
        let statuses = [
            TrainingStatus::Queued,
            TrainingStatus::Preparing,
            TrainingStatus::Running,
            TrainingStatus::Paused,
            TrainingStatus::Completed,
            TrainingStatus::Failed,
            TrainingStatus::Cancelled,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: TrainingStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*s, back);
        }
    }

    #[test]
    fn training_job_config_serde() {
        let config = TrainingJobConfig {
            base_model: "llama-7b".into(),
            dataset: DatasetConfig {
                path: "/data/train.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: Some("train".into()),
                max_samples: Some(1000),
            },
            method: TrainingMethod::Lora,
            hyperparams: HyperParams {
                learning_rate: 2e-4,
                epochs: 3,
                batch_size: 8,
                gradient_accumulation_steps: 4,
                warmup_steps: 100,
                weight_decay: 0.01,
                max_seq_length: 2048,
            },
            output_name: Some("my-finetune".into()),
            lora: Some(LoraConfig {
                rank: 16,
                alpha: 32.0,
                dropout: 0.05,
                target_modules: vec!["q_proj".into(), "v_proj".into()],
            }),
            max_steps: None,
            time_budget_secs: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: TrainingJobConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.base_model, "llama-7b");
        assert_eq!(back.method, TrainingMethod::Lora);
        assert_eq!(back.hyperparams.epochs, 3);
        assert_eq!(back.lora.unwrap().rank, 16);
    }

    #[test]
    fn checkpoint_info_serde() {
        let cp = CheckpointInfo {
            step: 500,
            epoch: 1.5,
            loss: 0.42,
            path: "/checkpoints/step-500".into(),
            timestamp: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&cp).unwrap();
        let back: CheckpointInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.step, 500);
        assert!((back.loss - 0.42).abs() < f64::EPSILON);
    }

    #[test]
    fn hyperparams_serde() {
        let hp = HyperParams {
            learning_rate: 1e-5,
            epochs: 1,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            warmup_steps: 0,
            weight_decay: 0.0,
            max_seq_length: 512,
        };
        let json = serde_json::to_string(&hp).unwrap();
        let back: HyperParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_seq_length, 512);
    }

    #[test]
    fn hyperparams_validate_zero_learning_rate() {
        let hp = HyperParams {
            learning_rate: 0.0,
            epochs: 1,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            warmup_steps: 0,
            weight_decay: 0.0,
            max_seq_length: 512,
        };
        assert!(hp.validate().is_err());
    }

    #[test]
    fn hyperparams_validate_negative_learning_rate() {
        let hp = HyperParams {
            learning_rate: -1e-4,
            epochs: 1,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            warmup_steps: 0,
            weight_decay: 0.0,
            max_seq_length: 512,
        };
        assert!(hp.validate().is_err());
    }

    #[test]
    fn hyperparams_validate_zero_epochs() {
        let hp = HyperParams {
            learning_rate: 1e-4,
            epochs: 0,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            warmup_steps: 0,
            weight_decay: 0.0,
            max_seq_length: 512,
        };
        assert!(hp.validate().is_err());
    }

    #[test]
    fn hyperparams_validate_zero_batch_size() {
        let hp = HyperParams {
            learning_rate: 1e-4,
            epochs: 1,
            batch_size: 0,
            gradient_accumulation_steps: 1,
            warmup_steps: 0,
            weight_decay: 0.0,
            max_seq_length: 512,
        };
        assert!(hp.validate().is_err());
    }

    #[test]
    fn hyperparams_validate_zero_max_seq_length() {
        let hp = HyperParams {
            learning_rate: 1e-4,
            epochs: 1,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            warmup_steps: 0,
            weight_decay: 0.0,
            max_seq_length: 0,
        };
        assert!(hp.validate().is_err());
    }

    #[test]
    fn hyperparams_validate_success() {
        let hp = HyperParams {
            learning_rate: 2e-4,
            epochs: 3,
            batch_size: 8,
            gradient_accumulation_steps: 4,
            warmup_steps: 100,
            weight_decay: 0.01,
            max_seq_length: 2048,
        };
        assert!(hp.validate().is_ok());
    }

    #[test]
    fn training_method_invalid_json() {
        let result = serde_json::from_str::<TrainingMethod>("\"invalid\"");
        assert!(result.is_err());
    }

    #[test]
    fn dataset_format_invalid_json() {
        let result = serde_json::from_str::<DatasetFormat>("\"invalid\"");
        assert!(result.is_err());
    }

    #[test]
    fn training_status_invalid_json() {
        let result = serde_json::from_str::<TrainingStatus>("\"invalid\"");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_training_method() -> impl Strategy<Value = TrainingMethod> {
        prop_oneof![
            Just(TrainingMethod::FullFineTune),
            Just(TrainingMethod::Lora),
            Just(TrainingMethod::Qlora),
            Just(TrainingMethod::Dpo),
            Just(TrainingMethod::Rlhf),
            Just(TrainingMethod::Distillation),
        ]
    }

    fn arb_training_status() -> impl Strategy<Value = TrainingStatus> {
        prop_oneof![
            Just(TrainingStatus::Queued),
            Just(TrainingStatus::Preparing),
            Just(TrainingStatus::Running),
            Just(TrainingStatus::Paused),
            Just(TrainingStatus::Completed),
            Just(TrainingStatus::Failed),
            Just(TrainingStatus::Cancelled),
        ]
    }

    fn arb_dataset_format() -> impl Strategy<Value = DatasetFormat> {
        prop_oneof![
            Just(DatasetFormat::Jsonl),
            Just(DatasetFormat::Parquet),
            Just(DatasetFormat::Csv),
            Just(DatasetFormat::HuggingFace),
        ]
    }

    proptest! {
        #[test]
        fn training_method_roundtrips(m in arb_training_method()) {
            let json = serde_json::to_string(&m).unwrap();
            let back: TrainingMethod = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(m, back);
        }

        #[test]
        fn training_status_roundtrips(s in arb_training_status()) {
            let json = serde_json::to_string(&s).unwrap();
            let back: TrainingStatus = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(s, back);
        }

        #[test]
        fn dataset_format_roundtrips(f in arb_dataset_format()) {
            let json = serde_json::to_string(&f).unwrap();
            let back: DatasetFormat = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(f, back);
        }

        #[test]
        fn hyperparams_valid_always_pass_validation(
            lr in 1e-8f64..1.0,
            epochs in 1u32..100,
            batch in 1u32..256,
            seq_len in 1u32..65536,
        ) {
            let hp = HyperParams {
                learning_rate: lr,
                epochs,
                batch_size: batch,
                gradient_accumulation_steps: 1,
                warmup_steps: 0,
                weight_decay: 0.0,
                max_seq_length: seq_len,
            };
            prop_assert!(hp.validate().is_ok());
        }
    }
}
