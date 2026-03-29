//! Shared test utilities for server handler tests.

#[cfg(test)]
pub mod helpers {
    use crate::config::*;
    use crate::server::state::AppState;

    /// Create a minimal test config pointing at a temporary directory.
    #[must_use]
    pub fn test_config(tmp: &tempfile::TempDir) -> IfranConfig {
        IfranConfig {
            server: ServerConfig {
                bind: "127.0.0.1:0".into(),
                grpc_bind: "127.0.0.1:0".into(),
                ws_bind: None,
            },
            storage: StorageConfig {
                models_dir: tmp.path().join("models"),
                database: tmp.path().join("test.db"),
                cache_dir: tmp.path().join("cache"),
                backend: Default::default(),
                postgres_url: None,
                postgres_pool_size: 8,
            },
            backends: BackendsConfig {
                default: "llamacpp".into(),
                enabled: vec!["llamacpp".into()],
            },
            training: TrainingConfig {
                executor: "subprocess".into(),
                trainer_image: None,
                max_concurrent_jobs: 2,
                checkpoints_dir: tmp.path().join("checkpoints"),
                job_eviction_ttl_secs: 86400,
            },
            bridge: BridgeConfig {
                sy_endpoint: None,
                enabled: false,
                heartbeat_interval_secs: 10,
            },
            hardware: HardwareConfig {
                gpu_memory_reserve_mb: 512,
                telemetry_interval_secs: 0,
            },
            security: SecurityConfig::default(),
            budget: BudgetConfig::default(),
            fleet: FleetConfig::default(),
        }
    }

    /// Create a test `AppState` from a temporary directory.
    #[must_use]
    pub fn test_state(tmp: &tempfile::TempDir) -> AppState {
        let config = test_config(tmp);
        AppState::new(config).unwrap()
    }
}
