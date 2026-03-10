//! Evaluation runner — orchestrates benchmark execution against a loaded model.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use synapse_types::SynapseError;
use synapse_types::error::Result;
use synapse_types::eval::*;

/// Tracks active and completed eval runs.
pub struct EvalRunner {
    runs: Arc<RwLock<HashMap<EvalRunId, EvalRunState>>>,
}

/// Internal state of an eval run.
#[derive(Debug, Clone)]
pub struct EvalRunState {
    pub run_id: EvalRunId,
    pub config: EvalConfig,
    pub status: EvalStatus,
    pub results: Vec<EvalResult>,
    pub error: Option<String>,
}

impl EvalRunner {
    pub fn new() -> Self {
        Self {
            runs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new eval run. Returns the run ID.
    pub async fn create_run(&self, config: EvalConfig) -> Result<EvalRunId> {
        let run_id = uuid::Uuid::new_v4();
        let state = EvalRunState {
            run_id,
            config,
            status: EvalStatus::Queued,
            results: Vec::new(),
            error: None,
        };
        self.runs.write().await.insert(run_id, state);
        Ok(run_id)
    }

    /// Get the state of an eval run.
    pub async fn get_run(&self, run_id: EvalRunId) -> Result<EvalRunState> {
        self.runs
            .read()
            .await
            .get(&run_id)
            .cloned()
            .ok_or_else(|| SynapseError::EvalError(format!("Eval run {run_id} not found")))
    }

    /// List all eval runs.
    pub async fn list_runs(&self) -> Vec<EvalRunState> {
        self.runs.read().await.values().cloned().collect()
    }

    /// Mark a run as started.
    pub async fn start_run(&self, run_id: EvalRunId) -> Result<()> {
        let mut runs = self.runs.write().await;
        let state = runs
            .get_mut(&run_id)
            .ok_or_else(|| SynapseError::EvalError(format!("Eval run {run_id} not found")))?;
        state.status = EvalStatus::Running;
        Ok(())
    }

    /// Record a benchmark result for a run.
    pub async fn record_result(&self, run_id: EvalRunId, result: EvalResult) -> Result<()> {
        let mut runs = self.runs.write().await;
        let state = runs
            .get_mut(&run_id)
            .ok_or_else(|| SynapseError::EvalError(format!("Eval run {run_id} not found")))?;
        state.results.push(result);
        Ok(())
    }

    /// Mark a run as completed.
    pub async fn complete_run(&self, run_id: EvalRunId) -> Result<()> {
        let mut runs = self.runs.write().await;
        let state = runs
            .get_mut(&run_id)
            .ok_or_else(|| SynapseError::EvalError(format!("Eval run {run_id} not found")))?;
        state.status = EvalStatus::Completed;
        Ok(())
    }

    /// Mark a run as failed.
    pub async fn fail_run(&self, run_id: EvalRunId, error: String) -> Result<()> {
        let mut runs = self.runs.write().await;
        let state = runs
            .get_mut(&run_id)
            .ok_or_else(|| SynapseError::EvalError(format!("Eval run {run_id} not found")))?;
        state.status = EvalStatus::Failed;
        state.error = Some(error);
        Ok(())
    }

    /// Run a custom (exact-match) benchmark against a model.
    ///
    /// This is the MVP eval path: load samples from JSONL, run inference on each,
    /// compare output to expected, return accuracy.
    ///
    /// The actual inference call is provided as a closure so the runner
    /// doesn't depend on any specific backend.
    pub async fn run_custom_benchmark<F, Fut>(
        &self,
        run_id: EvalRunId,
        dataset_path: &str,
        sample_limit: Option<usize>,
        model_name: &str,
        infer_fn: F,
    ) -> Result<EvalResult>
    where
        F: Fn(String) -> Fut,
        Fut: std::future::Future<Output = Result<String>>,
    {
        let samples = super::benchmarks::load_samples(dataset_path, sample_limit)?;
        let start = Instant::now();

        let mut predictions = Vec::new();
        for sample in &samples {
            match infer_fn(sample.prompt.clone()).await {
                Ok(output) => predictions.push((output, sample.expected.clone())),
                Err(e) => {
                    tracing::warn!(prompt = %sample.prompt, error = %e, "Eval inference failed");
                }
            }
        }

        let score = super::benchmarks::score_contains_match(&predictions);
        let duration = start.elapsed().as_secs_f64();

        let result = EvalResult {
            run_id,
            model_name: model_name.to_string(),
            benchmark: BenchmarkKind::Custom,
            score,
            details: Some(serde_json::json!({
                "total_samples": samples.len(),
                "successful_inferences": predictions.len(),
                "scoring_method": "contains_match",
            })),
            samples_evaluated: predictions.len() as u64,
            duration_secs: duration,
            evaluated_at: chrono::Utc::now(),
        };

        self.record_result(run_id, result.clone()).await?;
        Ok(result)
    }
}

impl Default for EvalRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn create_and_get_run() {
        let runner = EvalRunner::new();
        let config = EvalConfig {
            model_name: "test-model".into(),
            benchmarks: vec![BenchmarkKind::Custom],
            sample_limit: Some(10),
            dataset_path: Some("/tmp/eval.jsonl".into()),
        };
        let run_id = runner.create_run(config).await.unwrap();
        let state = runner.get_run(run_id).await.unwrap();
        assert_eq!(state.status, EvalStatus::Queued);
    }

    #[tokio::test]
    async fn lifecycle() {
        let runner = EvalRunner::new();
        let config = EvalConfig {
            model_name: "test-model".into(),
            benchmarks: vec![BenchmarkKind::Perplexity],
            sample_limit: None,
            dataset_path: None,
        };
        let run_id = runner.create_run(config).await.unwrap();
        runner.start_run(run_id).await.unwrap();
        assert_eq!(
            runner.get_run(run_id).await.unwrap().status,
            EvalStatus::Running
        );
        runner.complete_run(run_id).await.unwrap();
        assert_eq!(
            runner.get_run(run_id).await.unwrap().status,
            EvalStatus::Completed
        );
    }

    #[tokio::test]
    async fn fail_run() {
        let runner = EvalRunner::new();
        let config = EvalConfig {
            model_name: "test".into(),
            benchmarks: vec![],
            sample_limit: None,
            dataset_path: None,
        };
        let run_id = runner.create_run(config).await.unwrap();
        runner.fail_run(run_id, "OOM".into()).await.unwrap();
        let state = runner.get_run(run_id).await.unwrap();
        assert_eq!(state.status, EvalStatus::Failed);
        assert_eq!(state.error, Some("OOM".into()));
    }

    #[tokio::test]
    async fn list_runs() {
        let runner = EvalRunner::new();
        let config = EvalConfig {
            model_name: "m".into(),
            benchmarks: vec![],
            sample_limit: None,
            dataset_path: None,
        };
        runner.create_run(config.clone()).await.unwrap();
        runner.create_run(config).await.unwrap();
        assert_eq!(runner.list_runs().await.len(), 2);
    }
}
