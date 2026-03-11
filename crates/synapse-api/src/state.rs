/// Shared application state accessible across all API handlers.
use std::sync::Arc;
use synapse_backends::BackendRouter;
use synapse_bridge::client::BridgeClient;
use synapse_bridge::protocol::ProtocolConfig;
use synapse_bridge::server::BridgeServer;
use synapse_core::config::SynapseConfig;
use synapse_core::eval::runner::EvalRunner;
use synapse_core::lifecycle::manager::ModelManager;
use synapse_core::marketplace::catalog::MarketplaceCatalog;
use synapse_core::storage::db::ModelDatabase;
use synapse_train::distributed::coordinator::DistributedCoordinator;
use synapse_train::executor::ExecutorKind;
use synapse_train::job::manager::JobManager;
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
    pub bridge_client: Option<Arc<BridgeClient>>,
    pub bridge_server: Option<Arc<BridgeServer>>,
}

impl AppState {
    /// Create a new AppState from config.
    pub fn new(config: SynapseConfig) -> synapse_types::error::Result<Self> {
        let db = ModelDatabase::open(&config.storage.database)?;
        let backends = BackendRouter::new();
        let model_manager = ModelManager::new(config.hardware.gpu_memory_reserve_mb);
        let job_manager = JobManager::new(
            ExecutorKind::Subprocess,
            None,
            config.training.max_concurrent_jobs as usize,
        );
        let eval_runner = EvalRunner::new();
        let marketplace_db_path = config.storage.database.with_file_name("marketplace.db");
        let marketplace_catalog = MarketplaceCatalog::open(&marketplace_db_path)?;
        let distributed_coordinator = DistributedCoordinator::new();

        // Initialize bridge if enabled
        let (bridge_client, bridge_server) = if config.bridge.enabled {
            let endpoint = synapse_bridge::discovery::discover(
                config.bridge.sy_endpoint.as_deref(),
            )?;

            let protocol_config = ProtocolConfig {
                heartbeat_interval: std::time::Duration::from_secs(
                    config.bridge.heartbeat_interval_secs,
                ),
                ..ProtocolConfig::default()
            };

            let instance_id = std::env::var("SYNAPSE_INSTANCE_ID")
                .unwrap_or_else(|_| config.server.bind.clone());

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
            bridge_client,
            bridge_server,
        })
    }
}
