use crate::backends::BackendRouter;
use crate::bridge::client::BridgeClient;
use crate::bridge::protocol::ProtocolConfig;
use crate::bridge::server::BridgeServer;
use crate::config::IfranConfig;
use crate::eval::runner::EvalRunner;
use crate::experiment::store::ExperimentStore;
use crate::fleet::manager::FleetManager;
use crate::hardware::events::GpuEventBus;
use crate::hardware::telemetry::{TelemetryConfig, TelemetryLoop};
use crate::lifecycle::manager::ModelManager;
use crate::lineage::store::LineageStore;
use crate::marketplace::catalog::MarketplaceCatalog;
use crate::rag::store::RagStore;
use crate::rlhf::store::AnnotationStore;
use crate::storage::db::ModelDatabase;
use crate::tenant::store::TenantStore;
use crate::train::dataset::labeler::AutoLabeler;
use crate::train::distributed::coordinator::DistributedCoordinator;
use crate::train::executor::ExecutorKind;
use crate::train::experiment::runner::ExperimentHandle;
use crate::train::job::manager::JobManager;
use crate::train::job::store::JobStore;
use crate::training_events::TrainingEventBus;
use crate::types::experiment::ExperimentId;
use crate::versioning::store::VersionStore;
use majra::pubsub::PubSub;
use majra::ws::{WsBridge, WsBridgeConfig};
/// Shared application state accessible across all API handlers.
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Application state shared across all handlers via Axum's State extractor.
///
/// All SQLite stores use r2d2 connection pools and are Send+Sync, so no Mutex needed.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<IfranConfig>,
    pub db: Arc<ModelDatabase>,
    pub backends: Arc<BackendRouter>,
    pub model_manager: Arc<ModelManager>,
    pub job_manager: Arc<JobManager>,
    pub eval_runner: Arc<EvalRunner>,
    pub marketplace_catalog: Arc<MarketplaceCatalog>,
    pub auto_labeler: Arc<AutoLabeler>,
    pub distributed_coordinator: Arc<DistributedCoordinator>,
    pub experiment_store: Option<Arc<ExperimentStore>>,
    pub experiment_runners: Arc<Mutex<HashMap<ExperimentId, ExperimentHandle>>>,
    pub rag_store: Option<Arc<RagStore>>,
    pub annotation_store: Option<Arc<AnnotationStore>>,
    pub tenant_store: Option<Arc<TenantStore>>,
    pub lineage_store: Option<Arc<LineageStore>>,
    pub version_store: Option<Arc<VersionStore>>,
    pub bridge_client: Option<Arc<BridgeClient>>,
    pub bridge_server: Option<Arc<BridgeServer>>,
    pub event_hub: Arc<PubSub>,
    pub training_event_bus: Arc<TrainingEventBus>,
    pub gpu_event_bus: Arc<GpuEventBus>,
    pub telemetry: Option<Arc<TelemetryLoop>>,
    pub fleet_manager: Arc<FleetManager>,
    pub prometheus_registry: Arc<prometheus::Registry>,
}

impl AppState {
    /// Create a new AppState from config.
    pub fn new(config: IfranConfig) -> crate::types::error::Result<Self> {
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
        let job_manager = Arc::new(job_manager);

        let eval_runner = EvalRunner::new();
        let marketplace_db_path = config.storage.database.with_file_name("marketplace.db");
        let marketplace_catalog = MarketplaceCatalog::open(&marketplace_db_path)?;
        let auto_labeler = AutoLabeler::new();
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
                crate::types::IfranError::StorageError(format!("Failed to open tenant store: {e}"))
            })?)
        } else {
            None
        };

        let event_hub = Arc::new(PubSub::with_capacity(512));

        let training_event_bus = Arc::new(TrainingEventBus::with_hub(256, event_hub.clone()));

        let gpu_event_bus = Arc::new(GpuEventBus::with_hub(256, event_hub.clone()));

        // GPU telemetry loop
        let telemetry = if config.hardware.telemetry_interval_secs > 0 {
            Some(Arc::new(TelemetryLoop::start(TelemetryConfig {
                interval: std::time::Duration::from_secs(config.hardware.telemetry_interval_secs),
                enabled: true,
            })))
        } else {
            None
        };

        let prometheus_registry = Arc::new(prometheus::Registry::new());

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
                crate::bridge::discovery::discover(config.bridge.sy_endpoint.as_deref())?;

            let protocol_config = ProtocolConfig {
                heartbeat_interval: std::time::Duration::from_secs(
                    config.bridge.heartbeat_interval_secs,
                ),
                ..ProtocolConfig::default()
            };

            let instance_id =
                std::env::var("IFRAN_INSTANCE_ID").unwrap_or_else(|_| config.server.bind.clone());

            let client = BridgeClient::new(endpoint.address, protocol_config.clone());
            let server = BridgeServer::new(instance_id, protocol_config);

            (Some(Arc::new(client)), Some(Arc::new(server)))
        } else {
            (None, None)
        };

        if let Some(ws_addr) = &config.server.ws_bind {
            if let Ok(addr) = ws_addr.parse() {
                let bridge = Arc::new(WsBridge::new(event_hub.clone(), WsBridgeConfig::default()));
                bridge.spawn(addr);
            }
        }

        Ok(Self {
            config: Arc::new(config),
            db: Arc::new(db),
            backends: Arc::new(backends),
            model_manager: Arc::new(model_manager),
            job_manager,
            eval_runner: Arc::new(eval_runner),
            marketplace_catalog: Arc::new(marketplace_catalog),
            auto_labeler: Arc::new(auto_labeler),
            distributed_coordinator: Arc::new(distributed_coordinator),
            experiment_store: experiment_store.map(Arc::new),
            experiment_runners: Arc::new(Mutex::new(HashMap::new())),
            rag_store: rag_store.map(Arc::new),
            annotation_store: annotation_store.map(Arc::new),
            tenant_store: tenant_store.map(Arc::new),
            lineage_store: lineage_store.map(Arc::new),
            version_store: version_store.map(Arc::new),
            bridge_client,
            bridge_server,
            event_hub,
            training_event_bus,
            gpu_event_bus,
            telemetry,
            fleet_manager: Arc::new(fleet_manager),
            prometheus_registry,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn test_config(tmp: &tempfile::TempDir) -> IfranConfig {
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
