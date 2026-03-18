/// Shared application state accessible across all API handlers.
use std::collections::HashMap;
use std::sync::Arc;
use synapse_backends::BackendRouter;
use synapse_bridge::client::BridgeClient;
use synapse_bridge::protocol::ProtocolConfig;
use synapse_bridge::server::BridgeServer;
use synapse_core::config::SynapseConfig;
use synapse_core::eval::runner::EvalRunner;
use synapse_core::experiment::store::ExperimentStore;
use synapse_core::fleet::manager::FleetManager;
use synapse_core::hardware::events::GpuEventBus;
use synapse_core::hardware::telemetry::{TelemetryConfig, TelemetryLoop};
use synapse_core::lifecycle::manager::ModelManager;
use synapse_core::lineage::store::LineageStore;
use synapse_core::marketplace::catalog::MarketplaceCatalog;
use synapse_core::rag::store::RagStore;
use synapse_core::rlhf::store::AnnotationStore;
use synapse_core::storage::db::ModelDatabase;
use synapse_core::tenant::store::TenantStore;
use synapse_core::versioning::store::VersionStore;
use synapse_train::distributed::coordinator::DistributedCoordinator;
use synapse_train::executor::ExecutorKind;
use synapse_train::experiment::runner::ExperimentHandle;
use synapse_train::job::manager::JobManager;
use synapse_train::job::store::JobStore;
use synapse_types::experiment::ExperimentId;
use tokio::sync::Mutex;

/// Application state shared across all handlers via Axum's State extractor.
///
/// `ModelDatabase` and `MarketplaceCatalog` (rusqlite) are not Send, so they're
/// wrapped in a tokio Mutex which is held briefly for each DB operation.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<SynapseConfig>,
    pub db: Arc<Mutex<ModelDatabase>>,
    pub backends: Arc<BackendRouter>,
    pub model_manager: Arc<ModelManager>,
    pub job_manager: Arc<JobManager>,
    pub eval_runner: Arc<EvalRunner>,
    pub marketplace_catalog: Arc<Mutex<MarketplaceCatalog>>,
    pub distributed_coordinator: Arc<DistributedCoordinator>,
    pub experiment_store: Option<Arc<Mutex<ExperimentStore>>>,
    pub experiment_runners: Arc<Mutex<HashMap<ExperimentId, ExperimentHandle>>>,
    pub rag_store: Option<Arc<Mutex<RagStore>>>,
    pub annotation_store: Option<Arc<Mutex<AnnotationStore>>>,
    pub tenant_store: Option<Arc<Mutex<TenantStore>>>,
    pub lineage_store: Option<Arc<Mutex<LineageStore>>>,
    pub version_store: Option<Arc<Mutex<VersionStore>>>,
    pub bridge_client: Option<Arc<BridgeClient>>,
    pub bridge_server: Option<Arc<BridgeServer>>,
    pub gpu_event_bus: Arc<GpuEventBus>,
    pub telemetry: Option<Arc<TelemetryLoop>>,
    pub fleet_manager: Arc<FleetManager>,
}

impl AppState {
    /// Create a new AppState from config.
    pub fn new(config: SynapseConfig) -> synapse_types::error::Result<Self> {
        let db = ModelDatabase::open(&config.storage.database)?;
        let backends = BackendRouter::new();
        let model_manager = ModelManager::new(config.hardware.gpu_memory_reserve_mb);
        // Initialize job store for crash recovery
        let job_store_path = config.storage.database.with_file_name("training_jobs.db");
        let job_manager = match JobStore::open(&job_store_path) {
            Ok(store) => JobManager::new_with_store(
                ExecutorKind::Subprocess,
                None,
                config.training.max_concurrent_jobs as usize,
                store,
            ),
            Err(_) => JobManager::new(
                ExecutorKind::Subprocess,
                None,
                config.training.max_concurrent_jobs as usize,
            ),
        };
        let eval_runner = EvalRunner::new();
        let marketplace_db_path = config.storage.database.with_file_name("marketplace.db");
        let marketplace_catalog = MarketplaceCatalog::open(&marketplace_db_path)?;
        let distributed_coordinator = DistributedCoordinator::new();

        // Optional feature stores — fail silently if DB can't be opened.
        // These are non-critical; the API still works without them.
        let experiment_store_path = config.storage.database.with_file_name("experiments.db");
        let experiment_store = ExperimentStore::open(&experiment_store_path).ok();

        // Initialize RAG store
        let rag_store_path = config.storage.database.with_file_name("rag.db");
        let rag_store = RagStore::open(&rag_store_path).ok();

        // Initialize annotation store
        let annotation_store_path = config.storage.database.with_file_name("annotations.db");
        let annotation_store = AnnotationStore::open(&annotation_store_path).ok();

        // Initialize lineage store
        let lineage_store_path = config.storage.database.with_file_name("lineage.db");
        let lineage_store = LineageStore::open(&lineage_store_path).ok();

        // Initialize version store
        let version_store_path = config.storage.database.with_file_name("versions.db");
        let version_store = VersionStore::open(&version_store_path).ok();

        // Tenant store is REQUIRED when multi_tenant is enabled — propagate errors.
        let tenant_store = if config.security.multi_tenant {
            let tenant_store_path = config.storage.database.with_file_name("tenants.db");
            Some(TenantStore::open(&tenant_store_path).map_err(|e| {
                synapse_types::SynapseError::StorageError(format!(
                    "Failed to open tenant store: {e}"
                ))
            })?)
        } else {
            None
        };

        // GPU event bus
        let gpu_event_bus = Arc::new(GpuEventBus::new(256));

        // GPU telemetry loop
        let telemetry = if config.hardware.telemetry_interval_secs > 0 {
            Some(Arc::new(TelemetryLoop::start(TelemetryConfig {
                interval: std::time::Duration::from_secs(config.hardware.telemetry_interval_secs),
                enabled: true,
            })))
        } else {
            None
        };

        // Fleet manager
        let fleet_manager = FleetManager::new(
            std::time::Duration::from_secs(config.fleet.suspect_timeout_secs),
            std::time::Duration::from_secs(config.fleet.offline_timeout_secs),
        );
        if config.fleet.enabled {
            fleet_manager.start_health_check_loop(std::time::Duration::from_secs(
                config.fleet.health_check_interval_secs,
            ));
        }

        // Initialize bridge if enabled
        let (bridge_client, bridge_server) = if config.bridge.enabled {
            let endpoint =
                synapse_bridge::discovery::discover(config.bridge.sy_endpoint.as_deref())?;

            let protocol_config = ProtocolConfig {
                heartbeat_interval: std::time::Duration::from_secs(
                    config.bridge.heartbeat_interval_secs,
                ),
                ..ProtocolConfig::default()
            };

            let instance_id =
                std::env::var("SYNAPSE_INSTANCE_ID").unwrap_or_else(|_| config.server.bind.clone());

            let client = BridgeClient::new(endpoint.address, protocol_config.clone());
            let server = BridgeServer::new(instance_id, protocol_config);

            (Some(Arc::new(client)), Some(Arc::new(server)))
        } else {
            (None, None)
        };

        Ok(Self {
            config: Arc::new(config),
            db: Arc::new(Mutex::new(db)),
            backends: Arc::new(backends),
            model_manager: Arc::new(model_manager),
            job_manager: Arc::new(job_manager),
            eval_runner: Arc::new(eval_runner),
            marketplace_catalog: Arc::new(Mutex::new(marketplace_catalog)),
            distributed_coordinator: Arc::new(distributed_coordinator),
            experiment_store: experiment_store.map(|s| Arc::new(Mutex::new(s))),
            experiment_runners: Arc::new(Mutex::new(HashMap::new())),
            rag_store: rag_store.map(|s| Arc::new(Mutex::new(s))),
            annotation_store: annotation_store.map(|s| Arc::new(Mutex::new(s))),
            tenant_store: tenant_store.map(|s| Arc::new(Mutex::new(s))),
            lineage_store: lineage_store.map(|s| Arc::new(Mutex::new(s))),
            version_store: version_store.map(|s| Arc::new(Mutex::new(s))),
            bridge_client,
            bridge_server,
            gpu_event_bus,
            telemetry,
            fleet_manager: Arc::new(fleet_manager),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_core::config::*;

    fn test_config(tmp: &tempfile::TempDir) -> SynapseConfig {
        SynapseConfig {
            server: ServerConfig {
                bind: "127.0.0.1:0".into(),
                grpc_bind: "127.0.0.1:0".into(),
            },
            storage: StorageConfig {
                models_dir: tmp.path().join("models"),
                database: tmp.path().join("test.db"),
                cache_dir: tmp.path().join("cache"),
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
            },
            bridge: BridgeConfig {
                sy_endpoint: None,
                enabled: false,
                heartbeat_interval_secs: 10,
            },
            hardware: HardwareConfig {
                gpu_memory_reserve_mb: 512,
                telemetry_interval_secs: 0, // disabled in tests (no tokio runtime for sync tests)
            },
            security: SecurityConfig::default(),
            budget: BudgetConfig::default(),
            fleet: FleetConfig::default(),
        }
    }

    #[test]
    fn app_state_new_succeeds() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(&tmp);
        let state = AppState::new(config);
        assert!(state.is_ok());
    }

    #[test]
    fn app_state_is_clone() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(&tmp);
        let state = AppState::new(config).unwrap();
        let cloned = state.clone();
        assert!(Arc::ptr_eq(&state.config, &cloned.config));
        assert!(Arc::ptr_eq(&state.backends, &cloned.backends));
    }

    #[test]
    fn app_state_bridge_disabled() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(&tmp);
        let state = AppState::new(config).unwrap();
        assert!(state.bridge_client.is_none());
        assert!(state.bridge_server.is_none());
        assert!(!state.config.bridge.enabled);
    }

    #[test]
    fn app_state_creates_databases() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(&tmp);
        let _state = AppState::new(config).unwrap();
        assert!(tmp.path().join("test.db").exists());
        assert!(tmp.path().join("marketplace.db").exists());
    }

    #[tokio::test]
    async fn app_state_model_manager_starts_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(&tmp);
        let state = AppState::new(config).unwrap();
        let loaded = state.model_manager.list_loaded(None).await;
        assert!(loaded.is_empty());
    }

    #[test]
    fn app_state_no_tenant_store_by_default() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(&tmp);
        let state = AppState::new(config).unwrap();
        assert!(state.tenant_store.is_none());
    }

    #[test]
    fn app_state_tenant_store_when_multi_tenant() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut config = test_config(&tmp);
        config.security.multi_tenant = true;
        let state = AppState::new(config).unwrap();
        assert!(state.tenant_store.is_some());
    }
}
