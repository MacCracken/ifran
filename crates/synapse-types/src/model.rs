use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a model in the local catalog.
pub type ModelId = Uuid;

/// Supported model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    Onnx,
    TensorRt,
    PyTorch,
    Bin,
}

/// Quantization level for GGUF models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantLevel {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q6K,
    Q5KM,
    Q5KS,
    Q4KM,
    Q4KS,
    Q4_0,
    Q3KM,
    Q3KS,
    Q2K,
    Iq4Xs,
    Iq3Xxs,
    None,
}

/// Model metadata stored in the local catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: ModelId,
    pub name: String,
    pub repo_id: Option<String>,
    pub format: ModelFormat,
    pub quant: QuantLevel,
    pub size_bytes: u64,
    pub parameter_count: Option<u64>,
    pub architecture: Option<String>,
    pub license: Option<String>,
    pub local_path: String,
    pub sha256: Option<String>,
    pub pulled_at: chrono::DateTime<chrono::Utc>,
}

/// Manifest describing a model to be loaded by a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub info: ModelInfo,
    pub context_length: Option<u32>,
    pub gpu_layers: Option<u32>,
    pub tensor_split: Option<Vec<f32>>,
}
