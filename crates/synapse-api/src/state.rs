/// Shared application state accessible across all API handlers.

use std::sync::Arc;
use synapse_backends::BackendRouter;
use synapse_core::config::SynapseConfig;
use synapse_core::lifecycle::manager::ModelManager;
use synapse_core::storage::db::ModelDatabase;
use synapse_train::job::manager::JobManager;
use synapse_train::executor::ExecutorKind;
use tokio::sync::Mutex;

/// Application state shared across all handlers via Axum's State extractor.
///
/// `ModelDatabase` (rusqlite) is not Send, so it's wrapped in a tokio Mutex
/// which is held briefly for each DB operation.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<SynapseConfig>,
    pub db: Arc<Mutex<ModelDatabase>>,
    pub backends: Arc<BackendRouter>,
    pub model_manager: Arc<ModelManager>,
    pub job_manager: Arc<JobManager>,
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

        Ok(Self {
            config: Arc::new(config),
            db: Arc::new(Mutex::new(db)),
            backends: Arc::new(backends),
            model_manager: Arc::new(model_manager),
            job_manager: Arc::new(job_manager),
        })
    }
}
