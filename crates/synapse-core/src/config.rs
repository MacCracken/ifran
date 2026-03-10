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
    /// Load config from a specific file path.
    pub fn load(path: &std::path::Path) -> synapse_types::error::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| synapse_types::SynapseError::ConfigError(e.to_string()))?;
        toml::from_str(&content)
            .map_err(|e| synapse_types::SynapseError::ConfigError(e.to_string()))
    }

    /// Discover and load config using the standard resolution chain:
    ///
    /// 1. `SYNAPSE_CONFIG` environment variable
    /// 2. `~/.synapse/synapse.toml` (user config)
    /// 3. `/etc/synapse/synapse.toml` (system config, Agnosticos install)
    /// 4. Built-in defaults
    pub fn discover() -> Self {
        // 1. Explicit env override
        if let Ok(path) = std::env::var("SYNAPSE_CONFIG") {
            let p = PathBuf::from(&path);
            if p.exists()
                && let Ok(cfg) = Self::load(&p)
            {
                return cfg;
            }
        }

        // 2. User config
        if let Ok(home) = std::env::var("HOME") {
            let user_config = PathBuf::from(home).join(".synapse/synapse.toml");
            if user_config.exists()
                && let Ok(cfg) = Self::load(&user_config)
            {
                return cfg;
            }
        }

        // 3. System config (Agnosticos / systemd)
        let system_config = PathBuf::from("/etc/synapse/synapse.toml");
        if system_config.exists()
            && let Ok(cfg) = Self::load(&system_config)
        {
            return cfg;
        }

        // 4. Built-in defaults
        Self::default()
    }

    /// Returns the config file path that would be used by `discover()`,
    /// or `None` if falling back to defaults.
    pub fn discover_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("SYNAPSE_CONFIG") {
            let p = PathBuf::from(&path);
            if p.exists() {
                return Some(p);
            }
        }

        if let Ok(home) = std::env::var("HOME") {
            let user_config = PathBuf::from(home).join(".synapse/synapse.toml");
            if user_config.exists() {
                return Some(user_config);
            }
        }

        let system_config = PathBuf::from("/etc/synapse/synapse.toml");
        if system_config.exists() {
            return Some(system_config);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let cfg = SynapseConfig::default();
        assert_eq!(cfg.server.bind, "0.0.0.0:8420");
        assert_eq!(cfg.server.grpc_bind, "0.0.0.0:8421");
        assert_eq!(cfg.backends.default, "llamacpp");
        assert_eq!(cfg.training.max_concurrent_jobs, 2);
        assert!(!cfg.bridge.enabled);
        assert_eq!(cfg.hardware.gpu_memory_reserve_mb, 512);
    }

    #[test]
    fn default_storage_paths_under_synapse_dir() {
        let cfg = SynapseConfig::default();
        let models = cfg.storage.models_dir.to_string_lossy();
        let db = cfg.storage.database.to_string_lossy();
        let cache = cfg.storage.cache_dir.to_string_lossy();
        assert!(models.contains(".synapse/models") || models.contains("synapse"));
        assert!(db.ends_with("synapse.db"));
        assert!(cache.contains("cache"));
    }

    #[test]
    fn load_valid_toml() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let toml_content = r#"
[server]
bind = "127.0.0.1:9000"
grpc_bind = "127.0.0.1:9001"

[storage]
models_dir = "/tmp/models"
database = "/tmp/test.db"
cache_dir = "/tmp/cache"

[backends]
default = "ollama"
enabled = ["ollama"]

[training]
executor = "subprocess"
max_concurrent_jobs = 4
checkpoints_dir = "/tmp/checkpoints"

[bridge]
enabled = true
heartbeat_interval_secs = 30

[hardware]
gpu_memory_reserve_mb = 1024
"#;
        std::fs::write(tmp.path(), toml_content).unwrap();
        let cfg = SynapseConfig::load(tmp.path()).unwrap();
        assert_eq!(cfg.server.bind, "127.0.0.1:9000");
        assert_eq!(cfg.backends.default, "ollama");
        assert_eq!(cfg.training.max_concurrent_jobs, 4);
        assert!(cfg.bridge.enabled);
        assert_eq!(cfg.hardware.gpu_memory_reserve_mb, 1024);
    }

    #[test]
    fn load_invalid_toml_returns_error() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "this is not valid toml {{{{").unwrap();
        assert!(SynapseConfig::load(tmp.path()).is_err());
    }

    #[test]
    fn load_missing_file_returns_error() {
        let result = SynapseConfig::load(std::path::Path::new("/nonexistent/config.toml"));
        assert!(result.is_err());
    }
}
