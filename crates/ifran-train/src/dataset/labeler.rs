//! Auto-labeling pipeline — uses model inference to label unlabeled datasets.
//!
//! Reads a source JSONL file, calls an inference function for each sample,
//! and writes a labeled JSONL file with the model's output as the label.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::Arc;
use tokio::sync::RwLock;

use chrono::{DateTime, Utc};
use ifran_types::IfranError;
use ifran_types::error::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type AutoLabelJobId = Uuid;

/// Configuration for an auto-labeling job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoLabelConfig {
    /// Path to the source (unlabeled) JSONL file.
    pub source_path: String,
    /// Model name to use for labeling.
    pub model_name: String,
    /// JSON field name for the generated label (default: "expected").
    #[serde(default = "default_label_field")]
    pub label_field: String,
    /// JSON field containing the prompt/input text.
    #[serde(default = "default_prompt_field")]
    pub prompt_field: String,
    /// Optional system prompt to guide the labeler model.
    pub system_prompt: Option<String>,
    /// Max tokens for the labeling inference.
    pub max_tokens: Option<u32>,
    /// Temperature for labeling inference.
    pub temperature: Option<f32>,
    /// Output path for the labeled dataset.
    pub output_path: Option<String>,
}

fn default_label_field() -> String {
    "expected".into()
}

fn default_prompt_field() -> String {
    "prompt".into()
}

/// Status of an auto-labeling job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoLabelStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

/// State of an auto-labeling job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoLabelJobState {
    pub job_id: AutoLabelJobId,
    pub config: AutoLabelConfig,
    pub status: AutoLabelStatus,
    pub labeled_count: u64,
    pub total_count: u64,
    pub output_path: Option<String>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Manages auto-labeling jobs.
pub struct AutoLabeler {
    jobs: Arc<RwLock<HashMap<AutoLabelJobId, AutoLabelJobState>>>,
}

impl AutoLabeler {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new auto-labeling job. Returns the job ID.
    pub async fn create_job(&self, config: AutoLabelConfig) -> Result<AutoLabelJobId> {
        let job_id = Uuid::new_v4();
        let state = AutoLabelJobState {
            job_id,
            config,
            status: AutoLabelStatus::Queued,
            labeled_count: 0,
            total_count: 0,
            output_path: None,
            error: None,
            created_at: Utc::now(),
        };
        self.jobs.write().await.insert(job_id, state);
        Ok(job_id)
    }

    /// Get the state of an auto-labeling job.
    pub async fn get_job(&self, job_id: AutoLabelJobId) -> Result<AutoLabelJobState> {
        self.jobs
            .read()
            .await
            .get(&job_id)
            .cloned()
            .ok_or_else(|| IfranError::TrainingError(format!("Auto-label job {job_id} not found")))
    }

    /// List all auto-labeling jobs.
    pub async fn list_jobs(&self) -> Vec<AutoLabelJobState> {
        self.jobs.read().await.values().cloned().collect()
    }

    /// Mark a job as running.
    pub async fn start_job(&self, job_id: AutoLabelJobId) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = jobs.get_mut(&job_id).ok_or_else(|| {
            IfranError::TrainingError(format!("Auto-label job {job_id} not found"))
        })?;
        state.status = AutoLabelStatus::Running;
        Ok(())
    }

    /// Update progress on a running job.
    pub async fn update_progress(
        &self,
        job_id: AutoLabelJobId,
        labeled: u64,
        total: u64,
    ) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = jobs.get_mut(&job_id).ok_or_else(|| {
            IfranError::TrainingError(format!("Auto-label job {job_id} not found"))
        })?;
        state.labeled_count = labeled;
        state.total_count = total;
        Ok(())
    }

    /// Mark a job as completed.
    pub async fn complete_job(&self, job_id: AutoLabelJobId, output_path: String) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = jobs.get_mut(&job_id).ok_or_else(|| {
            IfranError::TrainingError(format!("Auto-label job {job_id} not found"))
        })?;
        state.status = AutoLabelStatus::Completed;
        state.output_path = Some(output_path);
        Ok(())
    }

    /// Mark a job as failed.
    pub async fn fail_job(&self, job_id: AutoLabelJobId, error: String) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let state = jobs.get_mut(&job_id).ok_or_else(|| {
            IfranError::TrainingError(format!("Auto-label job {job_id} not found"))
        })?;
        state.status = AutoLabelStatus::Failed;
        state.error = Some(error);
        Ok(())
    }
}

impl Default for AutoLabeler {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute the labeling pipeline: reads source JSONL, calls infer_fn per sample,
/// writes labeled JSONL. Counts samples in a single pass (no double-read).
///
/// The `infer_fn` takes a prompt string and returns the model's label text.
pub async fn run_labeling<F, Fut>(
    config: &AutoLabelConfig,
    output_path: &str,
    labeler: &AutoLabeler,
    job_id: AutoLabelJobId,
    infer_fn: F,
) -> Result<String>
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    labeler.update_progress(job_id, 0, 0).await?;

    let file = std::fs::File::open(&config.source_path)
        .map_err(|e| IfranError::TrainingError(format!("Failed to open source: {e}")))?;
    let reader = std::io::BufReader::new(file);

    let out_file = std::fs::File::create(output_path)
        .map_err(|e| IfranError::TrainingError(format!("Failed to create output: {e}")))?;
    let mut writer = std::io::BufWriter::new(out_file);

    let label_key = config.label_field.clone();
    let mut processed = 0u64;
    let mut total = 0u64;

    for line in reader.lines() {
        let line = line.map_err(|e| IfranError::TrainingError(format!("Read error: {e}")))?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        total += 1;

        let mut obj: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&line)
            .map_err(|e| IfranError::TrainingError(format!("Invalid JSON: {e}")))?;

        // Extract prompt text
        let prompt_text = match obj.get(&config.prompt_field) {
            Some(serde_json::Value::String(s)) => s.clone(),
            _ => {
                // Write unchanged if no prompt field
                writeln!(writer, "{}", serde_json::to_string(&obj).unwrap())
                    .map_err(|e| IfranError::TrainingError(format!("Write error: {e}")))?;
                continue;
            }
        };

        // Build full prompt with optional system prompt
        let full_prompt = match &config.system_prompt {
            Some(sys) => format!("{sys}\n\n{prompt_text}"),
            None => prompt_text,
        };

        // Call inference to get label
        match infer_fn(full_prompt).await {
            Ok(label) => {
                obj.insert(
                    label_key.clone(),
                    serde_json::Value::String(label.trim().to_string()),
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "Auto-label inference failed for sample");
                // Write unchanged on failure
            }
        }

        writeln!(writer, "{}", serde_json::to_string(&obj).unwrap())
            .map_err(|e| IfranError::TrainingError(format!("Write error: {e}")))?;

        processed += 1;
        #[allow(clippy::manual_is_multiple_of)]
        if processed % 10 == 0 {
            let _ = labeler.update_progress(job_id, processed, total).await;
        }
    }

    writer
        .flush()
        .map_err(|e| IfranError::TrainingError(format!("Flush failed: {e}")))?;

    // Final progress update
    labeler.update_progress(job_id, processed, total).await?;

    Ok(output_path.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn auto_label_config_defaults() {
        let json = r#"{"source_path":"/data.jsonl","model_name":"llama"}"#;
        let config: AutoLabelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.label_field, "expected");
        assert_eq!(config.prompt_field, "prompt");
        assert!(config.system_prompt.is_none());
    }

    #[test]
    fn auto_label_status_serde() {
        for s in [
            AutoLabelStatus::Queued,
            AutoLabelStatus::Running,
            AutoLabelStatus::Completed,
            AutoLabelStatus::Failed,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: AutoLabelStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[tokio::test]
    async fn create_and_get_job() {
        let labeler = AutoLabeler::new();
        let config = AutoLabelConfig {
            source_path: "/data.jsonl".into(),
            model_name: "test".into(),
            label_field: "expected".into(),
            prompt_field: "prompt".into(),
            system_prompt: None,
            max_tokens: None,
            temperature: None,
            output_path: None,
        };
        let id = labeler.create_job(config).await.unwrap();
        let state = labeler.get_job(id).await.unwrap();
        assert_eq!(state.status, AutoLabelStatus::Queued);
    }

    #[tokio::test]
    async fn job_lifecycle() {
        let labeler = AutoLabeler::new();
        let config = AutoLabelConfig {
            source_path: "/data.jsonl".into(),
            model_name: "test".into(),
            label_field: "expected".into(),
            prompt_field: "prompt".into(),
            system_prompt: None,
            max_tokens: None,
            temperature: None,
            output_path: None,
        };
        let id = labeler.create_job(config).await.unwrap();
        labeler.start_job(id).await.unwrap();
        assert_eq!(
            labeler.get_job(id).await.unwrap().status,
            AutoLabelStatus::Running
        );

        labeler.update_progress(id, 5, 10).await.unwrap();
        let state = labeler.get_job(id).await.unwrap();
        assert_eq!(state.labeled_count, 5);
        assert_eq!(state.total_count, 10);

        labeler.complete_job(id, "/out.jsonl".into()).await.unwrap();
        let state = labeler.get_job(id).await.unwrap();
        assert_eq!(state.status, AutoLabelStatus::Completed);
        assert_eq!(state.output_path, Some("/out.jsonl".into()));
    }

    #[tokio::test]
    async fn job_failure() {
        let labeler = AutoLabeler::new();
        let config = AutoLabelConfig {
            source_path: "/data.jsonl".into(),
            model_name: "test".into(),
            label_field: "expected".into(),
            prompt_field: "prompt".into(),
            system_prompt: None,
            max_tokens: None,
            temperature: None,
            output_path: None,
        };
        let id = labeler.create_job(config).await.unwrap();
        labeler.fail_job(id, "OOM".into()).await.unwrap();
        let state = labeler.get_job(id).await.unwrap();
        assert_eq!(state.status, AutoLabelStatus::Failed);
        assert_eq!(state.error, Some("OOM".into()));
    }

    #[tokio::test]
    async fn run_labeling_end_to_end() {
        let input = tempfile::NamedTempFile::new().unwrap();
        writeln!(&input, r#"{{"prompt":"What is 2+2?"}}"#).unwrap();
        writeln!(&input, r#"{{"prompt":"Capital of France?"}}"#).unwrap();

        let output = tempfile::NamedTempFile::new().unwrap();

        let labeler = AutoLabeler::new();
        let config = AutoLabelConfig {
            source_path: input.path().to_string_lossy().to_string(),
            model_name: "test".into(),
            label_field: "expected".into(),
            prompt_field: "prompt".into(),
            system_prompt: None,
            max_tokens: None,
            temperature: None,
            output_path: Some(output.path().to_string_lossy().to_string()),
        };
        let id = labeler.create_job(config.clone()).await.unwrap();
        labeler.start_job(id).await.unwrap();

        let mock_infer = |prompt: String| async move {
            if prompt.contains("2+2") {
                Ok("4".to_string())
            } else {
                Ok("Paris".to_string())
            }
        };

        let result = run_labeling(
            &config,
            &output.path().to_string_lossy(),
            &labeler,
            id,
            mock_infer,
        )
        .await
        .unwrap();

        labeler.complete_job(id, result.clone()).await.unwrap();

        // Verify output
        let content = std::fs::read_to_string(output.path()).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2);

        let obj1: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(obj1["expected"], "4");

        let obj2: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(obj2["expected"], "Paris");
    }

    #[tokio::test]
    async fn run_labeling_with_system_prompt() {
        let input = tempfile::NamedTempFile::new().unwrap();
        writeln!(&input, r#"{{"prompt":"hello"}}"#).unwrap();

        let output = tempfile::NamedTempFile::new().unwrap();

        let labeler = AutoLabeler::new();
        let config = AutoLabelConfig {
            source_path: input.path().to_string_lossy().to_string(),
            model_name: "test".into(),
            label_field: "label".into(),
            prompt_field: "prompt".into(),
            system_prompt: Some("Classify sentiment:".into()),
            max_tokens: None,
            temperature: None,
            output_path: None,
        };
        let id = labeler.create_job(config.clone()).await.unwrap();
        labeler.start_job(id).await.unwrap();

        let mock_infer = |prompt: String| async move {
            // Verify system prompt is prepended
            assert!(prompt.starts_with("Classify sentiment:"));
            Ok("positive".to_string())
        };

        run_labeling(
            &config,
            &output.path().to_string_lossy(),
            &labeler,
            id,
            mock_infer,
        )
        .await
        .unwrap();

        let content = std::fs::read_to_string(output.path()).unwrap();
        let obj: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(obj["label"], "positive");
    }

    #[tokio::test]
    async fn list_jobs() {
        let labeler = AutoLabeler::new();
        let config = AutoLabelConfig {
            source_path: "/a.jsonl".into(),
            model_name: "test".into(),
            label_field: "expected".into(),
            prompt_field: "prompt".into(),
            system_prompt: None,
            max_tokens: None,
            temperature: None,
            output_path: None,
        };
        labeler.create_job(config.clone()).await.unwrap();
        labeler.create_job(config).await.unwrap();
        assert_eq!(labeler.list_jobs().await.len(), 2);
    }
}
