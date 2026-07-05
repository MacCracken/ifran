//! Evaluation runner — orchestrates benchmark execution against a loaded model.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use crate::types::IfranError;
use crate::types::error::Result;
use crate::types::eval::*;

/// Maximum number of `EvalResult`s kept per run before oldest entries are pruned.
const MAX_RESULTS_BUFFER: usize = 10_000;

/// Tracks active and completed eval runs.
pub struct EvalRunner {
    runs: Arc<RwLock<HashMap<EvalRunId, EvalRunState>>>,
}

/// Internal state of an eval run.
#[derive(Debug, Clone)]
pub struct EvalRunState {
    pub run_id: EvalRunId,
    pub config: EvalConfig,
    pub tenant_id: String,
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

    /// Create a new eval run scoped to a tenant. Returns the run ID.
    pub async fn create_run(&self, config: EvalConfig, tenant_id: &str) -> Result<EvalRunId> {
        let run_id = uuid::Uuid::new_v4();
        let state = EvalRunState {
            run_id,
            config,
            tenant_id: tenant_id.to_string(),
            status: EvalStatus::Queued,
            results: Vec::new(),
            error: None,
        };
        self.runs.write().await.insert(run_id, state);
        Ok(run_id)
    }

    /// Get the state of an eval run, verifying tenant ownership.
    pub async fn get_run(&self, run_id: EvalRunId, tenant_id: &str) -> Result<EvalRunState> {
        let runs = self.runs.read().await;
        let run = runs
            .get(&run_id)
            .ok_or_else(|| IfranError::EvalError(format!("Eval run {run_id} not found")))?;
        if run.tenant_id != tenant_id {
            return Err(IfranError::EvalError(format!(
                "Eval run {run_id} not found"
            )));
        }
        Ok(run.clone())
    }

    /// List eval runs for a specific tenant.
    pub async fn list_runs(&self, tenant_id: &str) -> Vec<EvalRunState> {
        self.runs
            .read()
            .await
            .values()
            .filter(|r| r.tenant_id == tenant_id)
            .cloned()
            .collect()
    }

    /// Mark a run as started.
    pub async fn start_run(&self, run_id: EvalRunId) -> Result<()> {
        let mut runs = self.runs.write().await;
        let state = runs
            .get_mut(&run_id)
            .ok_or_else(|| IfranError::EvalError(format!("Eval run {run_id} not found")))?;
        state.status = EvalStatus::Running;
        Ok(())
    }

    /// Record a benchmark result for a run.
    pub async fn record_result(&self, run_id: EvalRunId, result: EvalResult) -> Result<()> {
        let mut runs = self.runs.write().await;
        let state = runs
            .get_mut(&run_id)
            .ok_or_else(|| IfranError::EvalError(format!("Eval run {run_id} not found")))?;
        if state.results.len() >= MAX_RESULTS_BUFFER {
            tracing::warn!(
                run_id = %run_id,
                "Eval result buffer full ({MAX_RESULTS_BUFFER}), dropping oldest results"
            );
            // Keep the most recent half
            let drain_count = MAX_RESULTS_BUFFER / 2;
            state.results.drain(..drain_count);
        }
        state.results.push(result);
        Ok(())
    }

    /// Mark a run as completed.
    pub async fn complete_run(&self, run_id: EvalRunId) -> Result<()> {
        let mut runs = self.runs.write().await;
        let state = runs
            .get_mut(&run_id)
            .ok_or_else(|| IfranError::EvalError(format!("Eval run {run_id} not found")))?;
        state.status = EvalStatus::Completed;
        Ok(())
    }

    /// Mark a run as failed.
    pub async fn fail_run(&self, run_id: EvalRunId, error: String) -> Result<()> {
        let mut runs = self.runs.write().await;
        let state = runs
            .get_mut(&run_id)
            .ok_or_else(|| IfranError::EvalError(format!("Eval run {run_id} not found")))?;
        state.status = EvalStatus::Failed;
        state.error = Some(error);
        Ok(())
    }

    /// Run a benchmark of the given kind. Dispatches to the appropriate method.
    ///
    /// The `infer_fn` closure is called with a prompt string and returns
    /// the model's text output, keeping the runner backend-agnostic.
    pub async fn run_benchmark<F, Fut>(
        &self,
        run_id: EvalRunId,
        kind: BenchmarkKind,
        dataset_path: &str,
        sample_limit: Option<usize>,
        model_name: &str,
        infer_fn: &F,
    ) -> Result<EvalResult>
    where
        F: Fn(String) -> Fut,
        Fut: std::future::Future<Output = Result<String>>,
    {
        match kind {
            BenchmarkKind::Custom => {
                self.run_custom_benchmark(run_id, dataset_path, sample_limit, model_name, |p| {
                    infer_fn(p)
                })
                .await
            }
            BenchmarkKind::Mmlu => {
                self.run_mmlu_benchmark(run_id, dataset_path, sample_limit, model_name, |p| {
                    infer_fn(p)
                })
                .await
            }
            BenchmarkKind::HellaSwag => {
                self.run_hellaswag_benchmark(run_id, dataset_path, sample_limit, model_name, |p| {
                    infer_fn(p)
                })
                .await
            }
            BenchmarkKind::HumanEval => {
                self.run_humaneval_benchmark(run_id, dataset_path, sample_limit, model_name, |p| {
                    infer_fn(p)
                })
                .await
            }
            BenchmarkKind::Perplexity => {
                self.run_perplexity_benchmark(run_id, dataset_path, sample_limit, model_name, |p| {
                    infer_fn(p)
                })
                .await
            }
        }
    }

    /// Run a custom (contains-match) benchmark against a model.
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

        let mut predictions = Vec::with_capacity(samples.len());
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

    /// Run an MMLU-style multiple-choice benchmark.
    pub async fn run_mmlu_benchmark<F, Fut>(
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

        let mut predictions = Vec::with_capacity(samples.len());
        for sample in &samples {
            let prompt = super::benchmarks::format_mmlu_prompt(sample);
            let expected = super::benchmarks::mmlu_expected_letter(sample);
            match infer_fn(prompt).await {
                Ok(output) => predictions.push((output, expected)),
                Err(e) => {
                    tracing::warn!(error = %e, "MMLU inference failed");
                }
            }
        }

        let score = super::benchmarks::score_mmlu(&predictions);
        let duration = start.elapsed().as_secs_f64();

        let result = EvalResult {
            run_id,
            model_name: model_name.to_string(),
            benchmark: BenchmarkKind::Mmlu,
            score,
            details: Some(serde_json::json!({
                "total_samples": samples.len(),
                "successful_inferences": predictions.len(),
                "scoring_method": "mmlu_letter_match",
            })),
            samples_evaluated: predictions.len() as u64,
            duration_secs: duration,
            evaluated_at: chrono::Utc::now(),
        };

        self.record_result(run_id, result.clone()).await?;
        Ok(result)
    }

    /// Run a HellaSwag-style commonsense completion benchmark.
    pub async fn run_hellaswag_benchmark<F, Fut>(
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

        let mut predictions = Vec::with_capacity(samples.len());
        for sample in &samples {
            let prompt = super::benchmarks::format_hellaswag_prompt(sample);
            match infer_fn(prompt).await {
                Ok(output) => predictions.push((output, sample.expected.clone())),
                Err(e) => {
                    tracing::warn!(error = %e, "HellaSwag inference failed");
                }
            }
        }

        let score = super::benchmarks::score_contains_match(&predictions);
        let duration = start.elapsed().as_secs_f64();

        let result = EvalResult {
            run_id,
            model_name: model_name.to_string(),
            benchmark: BenchmarkKind::HellaSwag,
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

    /// Run a HumanEval-style code generation benchmark.
    pub async fn run_humaneval_benchmark<F, Fut>(
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

        let mut predictions = Vec::with_capacity(samples.len());
        for sample in &samples {
            let prompt = super::benchmarks::format_humaneval_prompt(sample);
            match infer_fn(prompt).await {
                Ok(output) => predictions.push((output, sample.expected.clone())),
                Err(e) => {
                    tracing::warn!(error = %e, "HumanEval inference failed");
                }
            }
        }

        let score = super::benchmarks::score_contains_match(&predictions);
        let duration = start.elapsed().as_secs_f64();

        let result = EvalResult {
            run_id,
            model_name: model_name.to_string(),
            benchmark: BenchmarkKind::HumanEval,
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

    /// Run a perplexity benchmark (approximate, using sliding-window prediction).
    pub async fn run_perplexity_benchmark<F, Fut>(
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

        let mut predictions = Vec::with_capacity(samples.len());
        for sample in &samples {
            let prompt = super::benchmarks::format_perplexity_prompt(sample);
            // Expected: second half of the text
            let words: Vec<&str> = sample.expected.split_whitespace().collect();
            let split = words.len() / 2;
            let expected_continuation = if split < words.len() {
                words[split..].join(" ")
            } else {
                sample.expected.clone()
            };

            match infer_fn(prompt).await {
                Ok(output) => predictions.push((output, expected_continuation)),
                Err(e) => {
                    tracing::warn!(error = %e, "Perplexity inference failed");
                }
            }
        }

        let score = super::benchmarks::score_perplexity(&predictions);
        let duration = start.elapsed().as_secs_f64();

        let result = EvalResult {
            run_id,
            model_name: model_name.to_string(),
            benchmark: BenchmarkKind::Perplexity,
            score,
            details: Some(serde_json::json!({
                "total_samples": samples.len(),
                "successful_inferences": predictions.len(),
                "scoring_method": "approximate_perplexity",
                "note": "Approximate perplexity via sliding-window contains-match. Lower is better.",
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

    const TENANT: &str = "default";
    const TENANT_B: &str = "acme";

    #[tokio::test]
    async fn create_and_get_run() {
        let runner = EvalRunner::new();
        let config = EvalConfig {
            model_name: "test-model".into(),
            benchmarks: vec![BenchmarkKind::Custom],
            sample_limit: Some(10),
            dataset_path: Some("/tmp/eval.jsonl".into()),
        };
        let run_id = runner.create_run(config, TENANT).await.unwrap();
        let state = runner.get_run(run_id, TENANT).await.unwrap();
        assert_eq!(state.status, EvalStatus::Queued);
        assert_eq!(state.tenant_id, TENANT);
    }

    #[tokio::test]
    async fn tenant_isolation() {
        let runner = EvalRunner::new();
        let config = EvalConfig {
            model_name: "m".into(),
            benchmarks: vec![],
            sample_limit: None,
            dataset_path: None,
        };
        let run_a = runner.create_run(config.clone(), TENANT).await.unwrap();
        let run_b = runner.create_run(config, TENANT_B).await.unwrap();

        assert_eq!(runner.list_runs(TENANT).await.len(), 1);
        assert_eq!(runner.list_runs(TENANT_B).await.len(), 1);

        assert!(runner.get_run(run_a, TENANT_B).await.is_err());
        assert!(runner.get_run(run_b, TENANT).await.is_err());
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
        let run_id = runner.create_run(config, TENANT).await.unwrap();
        runner.start_run(run_id).await.unwrap();
        assert_eq!(
            runner.get_run(run_id, TENANT).await.unwrap().status,
            EvalStatus::Running
        );
        runner.complete_run(run_id).await.unwrap();
        assert_eq!(
            runner.get_run(run_id, TENANT).await.unwrap().status,
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
        let run_id = runner.create_run(config, TENANT).await.unwrap();
        runner.fail_run(run_id, "OOM".into()).await.unwrap();
        let state = runner.get_run(run_id, TENANT).await.unwrap();
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
        runner.create_run(config.clone(), TENANT).await.unwrap();
        runner.create_run(config, TENANT).await.unwrap();
        assert_eq!(runner.list_runs(TENANT).await.len(), 2);
    }

    #[tokio::test]
    async fn run_custom_benchmark_with_mock() {
        // Create a temp JSONL file
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(
            &mut std::io::BufWriter::new(tmp.as_file()),
            b"{\"prompt\":\"capital of France?\",\"expected\":\"Paris\"}\n\
              {\"prompt\":\"capital of Germany?\",\"expected\":\"Berlin\"}\n",
        )
        .unwrap();

        let runner = EvalRunner::new();
        let config = EvalConfig {
            model_name: "test".into(),
            benchmarks: vec![BenchmarkKind::Custom],
            sample_limit: None,
            dataset_path: Some(tmp.path().to_string_lossy().to_string()),
        };
        let run_id = runner.create_run(config, TENANT).await.unwrap();
        runner.start_run(run_id).await.unwrap();

        let mock_infer = |_prompt: String| async { Ok("Paris is the answer".to_string()) };

        let result = runner
            .run_custom_benchmark(
                run_id,
                &tmp.path().to_string_lossy(),
                None,
                "test",
                mock_infer,
            )
            .await
            .unwrap();

        assert_eq!(result.samples_evaluated, 2);
        assert!(result.score > 0.0); // "Paris" matches one
    }

    #[tokio::test]
    async fn run_mmlu_benchmark_with_mock() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(
            &mut std::io::BufWriter::new(tmp.as_file()),
            b"{\"prompt\":\"What is 2+2?\",\"expected\":\"A\",\"choices\":[\"4\",\"5\",\"6\",\"7\"],\"answer_index\":0}\n",
        )
        .unwrap();

        let runner = EvalRunner::new();
        let config = EvalConfig {
            model_name: "test".into(),
            benchmarks: vec![BenchmarkKind::Mmlu],
            sample_limit: None,
            dataset_path: None,
        };
        let run_id = runner.create_run(config, TENANT).await.unwrap();
        runner.start_run(run_id).await.unwrap();

        let mock_infer = |_prompt: String| async { Ok("A".to_string()) };

        let result = runner
            .run_mmlu_benchmark(
                run_id,
                &tmp.path().to_string_lossy(),
                None,
                "test",
                mock_infer,
            )
            .await
            .unwrap();

        assert_eq!(result.samples_evaluated, 1);
        assert!((result.score - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn run_benchmark_dispatcher() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(
            &mut std::io::BufWriter::new(tmp.as_file()),
            b"{\"prompt\":\"test\",\"expected\":\"yes\"}\n",
        )
        .unwrap();

        let runner = EvalRunner::new();
        let config = EvalConfig {
            model_name: "test".into(),
            benchmarks: vec![BenchmarkKind::Custom],
            sample_limit: None,
            dataset_path: None,
        };
        let run_id = runner.create_run(config, TENANT).await.unwrap();
        runner.start_run(run_id).await.unwrap();

        let mock_infer = |_prompt: String| async { Ok("yes".to_string()) };

        let result = runner
            .run_benchmark(
                run_id,
                BenchmarkKind::Custom,
                &tmp.path().to_string_lossy(),
                None,
                "test",
                &mock_infer,
            )
            .await
            .unwrap();

        assert_eq!(result.benchmark, BenchmarkKind::Custom);
        assert!((result.score - 1.0).abs() < 1e-6);
    }
}
