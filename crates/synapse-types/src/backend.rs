use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendId(pub String);

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Hardware accelerator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceleratorType {
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    Cpu,
}

/// Capabilities reported by a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub accelerators: Vec<AcceleratorType>,
    pub max_context_length: Option<u32>,
    pub supports_streaming: bool,
    pub supports_embeddings: bool,
    pub supports_vision: bool,
}

/// Device configuration for model loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub accelerator: AcceleratorType,
    pub device_ids: Vec<u32>,
    pub memory_limit_mb: Option<u64>,
}
