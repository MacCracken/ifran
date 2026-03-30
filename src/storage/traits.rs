//! Database store trait definitions.
//!
//! These traits abstract the storage layer so implementations can be swapped
//! (e.g. SQLite for development, Postgres for production).

use crate::types::PagedResult;
use crate::types::TenantId;
use crate::types::error::Result;
use crate::types::eval::{EvalResult, EvalRunId};
use crate::types::experiment::{
    Direction, ExperimentId, ExperimentProgram, ExperimentStatus, TrialId, TrialResult,
};
use crate::types::lineage::{LineageNode, PipelineStage};
use crate::types::marketplace::{MarketplaceEntry, MarketplaceQuery};
use crate::types::model::{ModelId, ModelInfo};
use crate::types::rag::{ChunkInfo, DocumentId, DocumentInfo, RagPipelineConfig, RagPipelineId};
use crate::types::rlhf::{AnnotationPair, AnnotationSession, AnnotationStats, Preference};
use crate::types::training::{TrainingJobId, TrainingStatus};
use crate::types::versioning::{ModelVersion, ModelVersionId};
use uuid::Uuid;

#[cfg(feature = "sqlite")]
use crate::experiment::store::{ExperimentRecord, ExperimentSummary};
#[cfg(feature = "sqlite")]
use crate::preference::store::PreferencePair;
#[cfg(feature = "sqlite")]
use crate::tenant::store::TenantRecord;
use crate::train::job::status::JobState;

/// Model catalog store.
pub trait ModelStore: Send + Sync {
    fn insert(&self, model: &ModelInfo, tenant_id: &TenantId) -> Result<()>;
    fn get(&self, id: ModelId, tenant_id: &TenantId) -> Result<ModelInfo>;
    fn get_by_name(&self, name: &str, tenant_id: &TenantId) -> Result<ModelInfo>;
    fn list(&self, tenant_id: &TenantId, limit: u32, offset: u32)
    -> Result<PagedResult<ModelInfo>>;
    fn update(&self, model: &ModelInfo, tenant_id: &TenantId) -> Result<()>;
    fn delete(&self, id: ModelId, tenant_id: &TenantId) -> Result<()>;
    fn count(&self, tenant_id: &TenantId) -> Result<usize>;
}

/// Marketplace catalog store.
pub trait MarketplaceStore: Send + Sync {
    fn publish(&self, entry: &MarketplaceEntry, tenant_id: &TenantId) -> Result<()>;
    fn get(&self, model_name: &str, tenant_id: &TenantId) -> Result<MarketplaceEntry>;
    fn search(
        &self,
        query: &MarketplaceQuery,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<MarketplaceEntry>>;
    fn list(
        &self,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<MarketplaceEntry>>;
    fn unpublish(&self, model_name: &str, tenant_id: &TenantId) -> Result<()>;
    fn count(&self, tenant_id: &TenantId) -> Result<usize>;
}

/// Experiment and trial store.
#[cfg(feature = "sqlite")]
pub trait ExperimentStore: Send + Sync {
    fn insert_experiment(
        &self,
        id: ExperimentId,
        name: &str,
        program: &ExperimentProgram,
        tenant_id: &TenantId,
    ) -> Result<()>;
    fn update_experiment_status(
        &self,
        id: ExperimentId,
        status: ExperimentStatus,
        tenant_id: &TenantId,
    ) -> Result<()>;
    fn update_best_trial(
        &self,
        experiment_id: ExperimentId,
        trial_id: TrialId,
        score: f64,
        tenant_id: &TenantId,
    ) -> Result<()>;
    fn insert_trial(&self, trial: &TrialResult) -> Result<()>;
    fn update_trial(&self, trial: &TrialResult) -> Result<()>;
    fn get_experiment(&self, id: ExperimentId, tenant_id: &TenantId) -> Result<ExperimentRecord>;
    fn list_experiments(
        &self,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<ExperimentSummary>>;
    fn get_trials(
        &self,
        experiment_id: ExperimentId,
        tenant_id: &TenantId,
    ) -> Result<Vec<TrialResult>>;
    fn get_leaderboard(
        &self,
        experiment_id: ExperimentId,
        direction: Direction,
        limit: usize,
        tenant_id: &TenantId,
    ) -> Result<Vec<TrialResult>>;
}

/// RAG pipeline and document store.
pub trait RagStore: Send + Sync {
    fn create_pipeline(
        &self,
        id: RagPipelineId,
        config: &RagPipelineConfig,
        tenant_id: &TenantId,
    ) -> Result<()>;
    fn get_pipeline(&self, id: RagPipelineId, tenant_id: &TenantId) -> Result<RagPipelineConfig>;
    fn list_pipelines(
        &self,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<(RagPipelineId, RagPipelineConfig)>>;
    fn delete_pipeline(&self, id: RagPipelineId, tenant_id: &TenantId) -> Result<()>;
    fn insert_document(&self, doc: &DocumentInfo) -> Result<()>;
    fn insert_chunk(&self, chunk: &ChunkInfo) -> Result<()>;
    fn get_chunks_for_document(&self, document_id: DocumentId) -> Result<Vec<ChunkInfo>>;
    fn get_all_chunks_for_pipeline(&self, pipeline_id: RagPipelineId) -> Result<Vec<ChunkInfo>>;
}

/// RLHF annotation store.
pub trait AnnotationStore: Send + Sync {
    fn create_session(
        &self,
        name: &str,
        model_name: &str,
        tenant_id: &TenantId,
    ) -> Result<AnnotationSession>;
    fn get_session(&self, id: Uuid, tenant_id: &TenantId) -> Result<AnnotationSession>;
    fn list_sessions(
        &self,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<AnnotationSession>>;
    fn add_pairs(&self, pairs: &[AnnotationPair]) -> Result<()>;
    fn get_pairs(&self, session_id: Uuid, tenant_id: &TenantId) -> Result<Vec<AnnotationPair>>;
    fn get_next_unannotated(
        &self,
        session_id: Uuid,
        tenant_id: &TenantId,
    ) -> Result<Option<AnnotationPair>>;
    fn annotate_pair(&self, pair_id: Uuid, preference: Preference) -> Result<()>;
    fn get_stats(&self, session_id: Uuid, tenant_id: &TenantId) -> Result<AnnotationStats>;
    fn export_session(&self, session_id: Uuid, tenant_id: &TenantId)
    -> Result<Vec<AnnotationPair>>;
}

/// Multi-tenant store.
#[cfg(feature = "sqlite")]
pub trait TenantStore: Send + Sync {
    fn create_tenant(&self, name: &str) -> Result<(TenantRecord, String)>;
    fn resolve_by_key(&self, raw_key: &str) -> Result<TenantRecord>;
    fn list_tenants(&self, limit: u32, offset: u32) -> Result<PagedResult<TenantRecord>>;
    fn disable_tenant(&self, id: &TenantId) -> Result<()>;
    fn enable_tenant(&self, id: &TenantId) -> Result<()>;
}

/// Model lineage store.
pub trait LineageStore: Send + Sync {
    fn record(&self, node: &LineageNode, tenant_id: &TenantId) -> Result<()>;
    fn get(&self, id: Uuid, tenant_id: &TenantId) -> Result<LineageNode>;
    fn get_ancestry(
        &self,
        id: Uuid,
        tenant_id: &TenantId,
        max_depth: Option<u32>,
    ) -> Result<Vec<LineageNode>>;
    fn list(
        &self,
        tenant_id: &TenantId,
        stage: Option<PipelineStage>,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<LineageNode>>;
    fn find_by_artifact(
        &self,
        artifact_ref: &str,
        tenant_id: &TenantId,
    ) -> Result<Vec<LineageNode>>;
}

/// Model version store.
pub trait VersionStore: Send + Sync {
    fn create(&self, version: &ModelVersion, tenant_id: &TenantId) -> Result<()>;
    fn get(&self, id: ModelVersionId, tenant_id: &TenantId) -> Result<ModelVersion>;
    fn list_by_family(
        &self,
        family: &str,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<ModelVersion>>;
    fn list(
        &self,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<ModelVersion>>;
    fn latest(&self, family: &str, tenant_id: &TenantId) -> Result<ModelVersion>;
    fn get_lineage(&self, id: ModelVersionId, tenant_id: &TenantId) -> Result<Vec<ModelVersion>>;
}

/// Training job persistence store.
pub trait JobStore: Send + Sync {
    fn save_job(&self, job: &JobState) -> Result<()>;
    fn get_job(&self, id: TrainingJobId) -> Result<Option<JobState>>;
    fn list_jobs(
        &self,
        status_filter: Option<TrainingStatus>,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<JobState>>;
    fn delete_job(&self, id: TrainingJobId) -> Result<()>;
    fn recover_jobs(&self) -> Result<Vec<JobState>>;
}

/// Evaluation result store.
pub trait EvalStore: Send + Sync {
    fn insert(&self, result: &EvalResult, tenant_id: &TenantId) -> Result<()>;
    fn get_run(&self, run_id: EvalRunId, tenant_id: &TenantId) -> Result<Vec<EvalResult>>;
    fn get_by_model(&self, model_name: &str, tenant_id: &TenantId) -> Result<Vec<EvalResult>>;
    fn list(&self, tenant_id: &TenantId) -> Result<Vec<EvalResult>>;
}

/// Preference pair store for DPO/RLHF training data.
#[cfg(feature = "sqlite")]
pub trait PreferenceStore: Send + Sync {
    fn add(&self, pair: &PreferencePair, tenant_id: &TenantId) -> Result<()>;
    fn list(&self, tenant_id: &TenantId, limit: u32) -> Result<Vec<PreferencePair>>;
    fn export_dpo(&self, tenant_id: &TenantId) -> Result<Vec<serde_json::Value>>;
    fn count(&self, tenant_id: &TenantId) -> Result<u64>;
    fn add_batch(&self, pairs: &[PreferencePair], tenant_id: &TenantId) -> Result<u64>;
}
