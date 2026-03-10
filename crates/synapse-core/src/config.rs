use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level Synapse configuration, loaded from synapse.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseConfig {
    pub server: ServerConfig,
    pub storage: StorageConfig,
    pub backends: BackendsConfig,
    pub training: TrainingConfig,
    pub bridge: BridgeConfig,
    pub hardware: HardwareConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub bind: String,
    pub grpc_bind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub models_dir: PathBuf,
    pub database: PathBuf,
    pub cache_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendsConfig {
    pub default: String,
    pub enabled: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub executor: String,
    pub trainer_image: Option<String>,
    pub max_concurrent_jobs: u32,
    pub checkpoints_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    pub sy_endpoint: Option<String>,
    pub enabled: bool,
    pub heartbeat_interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub gpu_memory_reserve_mb: u64,
}

impl Default for SynapseConfig {
    fn default() -> Self {
        let home = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        let synapse_dir = home.join(".synapse");

        Self {
            server: ServerConfig {
                bind: "0.0.0.0:8420".into(),
                grpc_bind: "0.0.0.0:8421".into(),
            },
            storage: StorageConfig {
                models_dir: synapse_dir.join("models"),
                database: synapse_dir.join("synapse.db"),
                cache_dir: synapse_dir.join("cache"),
            },
            backends: BackendsConfig {
                default: "llamacpp".into(),
                enabled: vec!["llamacpp".into(), "candle".into()],
            },
            training: TrainingConfig {
                executor: "docker".into(),
                trainer_image: Some("ghcr.io/maccracken/synapse-trainer:latest".into()),
                max_concurrent_jobs: 2,
                checkpoints_dir: synapse_dir.join("checkpoints"),
            },
            bridge: BridgeConfig {
                sy_endpoint: None,
                enabled: false,
                heartbeat_interval_secs: 10,
            },
            hardware: HardwareConfig {
                gpu_memory_reserve_mb: 512,
            },
        }
    }
}

impl SynapseConfig {
    pub fn load(path: &std::path::Path) -> synapse_types::error::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| synapse_types::SynapseError::ConfigError(e.to_string()))?;
        toml::from_str(&content)
            .map_err(|e| synapse_types::SynapseError::ConfigError(e.to_string()))
    }
}
