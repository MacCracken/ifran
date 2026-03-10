/// Shared application state accessible across all API handlers.
use std::sync::Arc;
use synapse_backends::BackendRouter;
use synapse_core::config::SynapseConfig;
use synapse_core::eval::runner::EvalRunner;
use synapse_core::lifecycle::manager::ModelManager;
use synapse_core::marketplace::catalog::MarketplaceCatalog;
use synapse_core::storage::db::ModelDatabase;
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

        Ok(Self {
            config: Arc::new(config),
            db: Arc::new(Mutex::new(db)),
            backends: Arc::new(backends),
            model_manager: Arc::new(model_manager),
            job_manager: Arc::new(job_manager),
            eval_runner: Arc::new(eval_runner),
            marketplace_catalog: Arc::new(Mutex::new(marketplace_catalog)),
        })
    }
}
