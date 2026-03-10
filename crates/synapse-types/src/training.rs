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
