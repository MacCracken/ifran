//! DAG-based ML workflow orchestration backed by majra's WorkflowEngine.
//!
//! Wraps [`majra::dag::WorkflowEngine`] with training-specific step types
//! (`Curate`, `Train`, `Evaluate`, `Approve`, `Deploy`) and a
//! [`TrainingStepExecutor`] that dispatches based on step config.

use async_trait::async_trait;
use majra::dag::{
    InMemoryWorkflowStorage, StepExecutor, WorkflowContext, WorkflowDefinition, WorkflowEngine,
    WorkflowEngineConfig, WorkflowRun, WorkflowStep, WorkflowStorage,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Re-export majra types used by consumers.
pub use majra::dag::StepRunStatus;
pub use majra::dag::{ErrorPolicy, RetryPolicy, TriggerMode, WorkflowRunStatus};

/// Ifran-specific ML step types.
///
/// Each variant maps to a phase of the training pipeline. The step type
/// is stored in `WorkflowStep::config` under the `"step_type"` key so
/// the executor can dispatch accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StepType {
    Curate,
    Train,
    Evaluate,
    Approve,
    Deploy,
}

/// Executes a single training workflow step by reading the `"step_type"`
/// field from `step.config` and dispatching to the appropriate handler.
///
/// This is the ifran-side implementation of [`majra::dag::StepExecutor`].
pub struct TrainingStepExecutor;

#[async_trait]
impl StepExecutor for TrainingStepExecutor {
    async fn execute(
        &self,
        step: &WorkflowStep,
        context: &WorkflowContext,
    ) -> Result<serde_json::Value, String> {
        let step_type: StepType = step
            .config
            .get("step_type")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .ok_or_else(|| {
                format!(
                    "step '{}' missing or invalid 'step_type' in config",
                    step.id
                )
            })?;

        tracing::info!(
            step_id = %step.id,
            step_name = %step.name,
            step_type = ?step_type,
            "executing training workflow step"
        );

        match step_type {
            StepType::Curate => execute_curate(step, context).await,
            StepType::Train => execute_train(step, context).await,
            StepType::Evaluate => execute_evaluate(step, context).await,
            StepType::Approve => execute_approve(step, context).await,
            StepType::Deploy => execute_deploy(step, context).await,
        }
    }
}

/// Curate step: data selection, filtering, preprocessing.
async fn execute_curate(
    step: &WorkflowStep,
    _context: &WorkflowContext,
) -> Result<serde_json::Value, String> {
    tracing::debug!(step_id = %step.id, "curate: processing dataset");
    Ok(serde_json::json!({ "status": "curated", "step_id": step.id }))
}

/// Train step: model training execution.
async fn execute_train(
    step: &WorkflowStep,
    _context: &WorkflowContext,
) -> Result<serde_json::Value, String> {
    tracing::debug!(step_id = %step.id, "train: running training job");
    Ok(serde_json::json!({ "status": "trained", "step_id": step.id }))
}

/// Evaluate step: model evaluation and metrics collection.
async fn execute_evaluate(
    step: &WorkflowStep,
    _context: &WorkflowContext,
) -> Result<serde_json::Value, String> {
    tracing::debug!(step_id = %step.id, "evaluate: running evaluation");
    Ok(serde_json::json!({ "status": "evaluated", "step_id": step.id }))
}

/// Approve step: human or automated approval gate.
async fn execute_approve(
    step: &WorkflowStep,
    _context: &WorkflowContext,
) -> Result<serde_json::Value, String> {
    tracing::debug!(step_id = %step.id, "approve: auto-approving");
    Ok(serde_json::json!({ "status": "approved", "step_id": step.id }))
}

/// Deploy step: model deployment to serving infrastructure.
async fn execute_deploy(
    step: &WorkflowStep,
    _context: &WorkflowContext,
) -> Result<serde_json::Value, String> {
    tracing::debug!(step_id = %step.id, "deploy: deploying model");
    Ok(serde_json::json!({ "status": "deployed", "step_id": step.id }))
}

/// Training workflow engine wrapping majra's [`WorkflowEngine`] with
/// in-memory storage and the [`TrainingStepExecutor`].
pub struct TrainingWorkflow {
    engine: WorkflowEngine<InMemoryWorkflowStorage, TrainingStepExecutor>,
    storage: Arc<InMemoryWorkflowStorage>,
}

impl TrainingWorkflow {
    /// Create a new training workflow engine with default configuration.
    #[must_use]
    pub fn new() -> Self {
        let storage = Arc::new(InMemoryWorkflowStorage::new());
        let executor = Arc::new(TrainingStepExecutor);
        let engine =
            WorkflowEngine::new(storage.clone(), executor, WorkflowEngineConfig::default());
        Self { engine, storage }
    }

    /// Create a new training workflow engine with custom configuration.
    #[must_use]
    pub fn with_config(config: WorkflowEngineConfig) -> Self {
        let storage = Arc::new(InMemoryWorkflowStorage::new());
        let executor = Arc::new(TrainingStepExecutor);
        let engine = WorkflowEngine::new(storage.clone(), executor, config);
        Self { engine, storage }
    }

    /// Create a workflow definition and store it.
    ///
    /// Validates the DAG structure before persisting.
    pub async fn create_workflow(
        &self,
        definition: WorkflowDefinition,
    ) -> Result<(), majra::error::MajraError> {
        WorkflowEngine::<InMemoryWorkflowStorage, TrainingStepExecutor>::validate(&definition)?;
        self.storage.create_definition(&definition).await?;
        Ok(())
    }

    /// Validate a workflow definition without storing it.
    pub fn validate(definition: &WorkflowDefinition) -> Result<(), majra::error::MajraError> {
        WorkflowEngine::<InMemoryWorkflowStorage, TrainingStepExecutor>::validate(definition)
    }

    /// Execute a workflow to completion.
    ///
    /// Creates a run, walks tiers in topological order, executes steps
    /// with retry and error policies, and returns the completed run.
    pub async fn execute(
        &self,
        definition: &WorkflowDefinition,
        input: Option<serde_json::Value>,
        triggered_by: &str,
    ) -> Result<WorkflowRun, majra::error::MajraError> {
        self.engine.execute(definition, input, triggered_by).await
    }

    /// Cancel a running workflow. Steps already in flight will complete,
    /// but no new tiers will be started.
    pub async fn cancel(&self, run_id: &str) -> Result<(), majra::error::MajraError> {
        self.engine.cancel(run_id).await
    }

    /// Access the underlying storage for queries.
    #[must_use]
    pub fn storage(&self) -> &Arc<InMemoryWorkflowStorage> {
        &self.storage
    }
}

impl Default for TrainingWorkflow {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to build a [`WorkflowStep`] with a given ifran [`StepType`].
#[must_use]
pub fn training_step(
    id: &str,
    name: &str,
    step_type: StepType,
    depends_on: Vec<String>,
) -> WorkflowStep {
    WorkflowStep {
        id: id.into(),
        name: name.into(),
        depends_on,
        trigger_mode: TriggerMode::default(),
        config: serde_json::json!({ "step_type": step_type }),
        error_policy: ErrorPolicy::default(),
        retry_policy: RetryPolicy::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_definition(steps: Vec<WorkflowStep>) -> WorkflowDefinition {
        let now = chrono::Utc::now().timestamp_millis();
        WorkflowDefinition {
            id: uuid::Uuid::new_v4().to_string(),
            name: "test-workflow".into(),
            description: None,
            steps,
            enabled: true,
            version: 1,
            created_by: "test".into(),
            created_at: now,
            updated_at: now,
        }
    }

    #[test]
    fn step_type_serde() {
        for t in [
            StepType::Curate,
            StepType::Train,
            StepType::Evaluate,
            StepType::Approve,
            StepType::Deploy,
        ] {
            let json = serde_json::to_string(&t).unwrap();
            let back: StepType = serde_json::from_str(&json).unwrap();
            assert_eq!(t, back);
        }
    }

    #[test]
    fn valid_pipeline() {
        let s1 = training_step("curate", "curate", StepType::Curate, vec![]);
        let s2 = training_step("train", "train", StepType::Train, vec!["curate".into()]);
        let s3 = training_step("eval", "eval", StepType::Evaluate, vec!["train".into()]);
        let s4 = training_step("approve", "approve", StepType::Approve, vec!["eval".into()]);
        let s5 = training_step("deploy", "deploy", StepType::Deploy, vec!["approve".into()]);

        let def = make_definition(vec![s1, s2, s3, s4, s5]);
        assert!(TrainingWorkflow::validate(&def).is_ok());
    }

    #[test]
    fn cycle_detection() {
        let s1 = training_step("a", "a", StepType::Curate, vec!["b".into()]);
        let s2 = training_step("b", "b", StepType::Train, vec!["a".into()]);
        let def = make_definition(vec![s1, s2]);
        assert!(TrainingWorkflow::validate(&def).is_err());
    }

    #[test]
    fn missing_dependency() {
        let s1 = training_step("a", "a", StepType::Curate, vec!["nonexistent".into()]);
        let def = make_definition(vec![s1]);
        assert!(TrainingWorkflow::validate(&def).is_err());
    }

    #[test]
    fn empty_pipeline_valid() {
        let def = make_definition(vec![]);
        assert!(TrainingWorkflow::validate(&def).is_ok());
    }

    #[test]
    fn diamond_dependency() {
        let a = training_step("a", "A", StepType::Curate, vec![]);
        let b = training_step("b", "B", StepType::Train, vec!["a".into()]);
        let c = training_step("c", "C", StepType::Evaluate, vec!["a".into()]);
        let d = training_step("d", "D", StepType::Deploy, vec!["b".into(), "c".into()]);

        let def = make_definition(vec![a, b, c, d]);
        assert!(TrainingWorkflow::validate(&def).is_ok());
    }

    #[test]
    fn single_step_pipeline() {
        let s = training_step("only", "only", StepType::Deploy, vec![]);
        let def = make_definition(vec![s]);
        assert!(TrainingWorkflow::validate(&def).is_ok());
    }

    #[test]
    fn parallel_steps() {
        let s1 = training_step("a", "curate-a", StepType::Curate, vec![]);
        let s2 = training_step("b", "curate-b", StepType::Curate, vec![]);
        let def = make_definition(vec![s1, s2]);
        assert!(TrainingWorkflow::validate(&def).is_ok());
    }

    #[tokio::test]
    async fn execute_linear_workflow() {
        let tw = TrainingWorkflow::new();
        let s1 = training_step("curate", "curate", StepType::Curate, vec![]);
        let s2 = training_step("train", "train", StepType::Train, vec!["curate".into()]);
        let s3 = training_step("eval", "eval", StepType::Evaluate, vec!["train".into()]);

        let def = make_definition(vec![s1, s2, s3]);
        tw.create_workflow(def.clone()).await.unwrap();

        let run = tw.execute(&def, None, "test").await.unwrap();
        assert_eq!(run.status, WorkflowRunStatus::Completed);
        assert!(run.output.is_some());
    }

    #[tokio::test]
    async fn execute_diamond_workflow() {
        let tw = TrainingWorkflow::new();
        let a = training_step("a", "curate", StepType::Curate, vec![]);
        let b = training_step("b", "train", StepType::Train, vec!["a".into()]);
        let c = training_step("c", "eval", StepType::Evaluate, vec!["a".into()]);
        let d = training_step(
            "d",
            "deploy",
            StepType::Deploy,
            vec!["b".into(), "c".into()],
        );

        let def = make_definition(vec![a, b, c, d]);
        tw.create_workflow(def.clone()).await.unwrap();

        let run = tw.execute(&def, None, "test").await.unwrap();
        assert_eq!(run.status, WorkflowRunStatus::Completed);

        let output = run.output.unwrap();
        let map = output.as_object().unwrap();
        assert_eq!(map.len(), 4);
    }

    #[tokio::test]
    async fn create_invalid_workflow_fails() {
        let tw = TrainingWorkflow::new();
        let s1 = training_step("a", "a", StepType::Curate, vec!["b".into()]);
        let s2 = training_step("b", "b", StepType::Train, vec!["a".into()]);
        let def = make_definition(vec![s1, s2]);
        assert!(tw.create_workflow(def).await.is_err());
    }

    #[test]
    fn step_run_status_values() {
        // Verify StepRunStatus re-export works and has expected variants.
        let _pending = StepRunStatus::Pending;
        let _running = StepRunStatus::Running;
        let _completed = StepRunStatus::Completed;
        let _failed = StepRunStatus::Failed;
        let _skipped = StepRunStatus::Skipped;
    }

    #[test]
    fn training_step_helper() {
        let step = training_step("my-step", "My Step", StepType::Train, vec!["dep1".into()]);
        assert_eq!(step.id, "my-step");
        assert_eq!(step.name, "My Step");
        assert_eq!(step.depends_on, vec!["dep1".to_string()]);
        let st: StepType = serde_json::from_value(step.config["step_type"].clone()).unwrap();
        assert_eq!(st, StepType::Train);
    }
}
