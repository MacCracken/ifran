//! REST handlers for dataset operations — auto-labeling and augmentation.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use synapse_train::dataset::labeler::{AutoLabelJobId, AutoLabelStatus};
use synapse_train::dataset::processor::AugmentationStrategy;
use synapse_types::inference::InferenceRequest;

use crate::state::AppState;

// --- Auto-labeling ---

/// Request to create an auto-labeling job.
#[derive(Deserialize)]
pub struct CreateAutoLabelRequest {
    pub source_path: String,
    pub model_name: String,
    #[serde(default = "default_label_field")]
    pub label_field: String,
    #[serde(default = "default_prompt_field")]
    pub prompt_field: String,
    pub system_prompt: Option<String>,
    pub output_path: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

fn default_label_field() -> String {
    "expected".into()
}
fn default_prompt_field() -> String {
    "prompt".into()
}

/// Response for an auto-labeling job.
#[derive(Serialize)]
pub struct AutoLabelJobResponse {
    pub job_id: AutoLabelJobId,
    pub status: AutoLabelStatus,
    pub labeled_count: u64,
    pub total_count: u64,
    pub output_path: Option<String>,
    pub error: Option<String>,
}

/// POST /datasets/auto-label — create and start an auto-labeling job.
pub async fn create_auto_label(
    State(state): State<AppState>,
    Json(req): Json<CreateAutoLabelRequest>,
) -> Result<(StatusCode, Json<AutoLabelJobResponse>), (StatusCode, String)> {
    let output_path = req.output_path.unwrap_or_else(|| {
        format!(
            "{}.labeled.jsonl",
            req.source_path.trim_end_matches(".jsonl")
        )
    });

    let config = synapse_train::dataset::labeler::AutoLabelConfig {
        source_path: req.source_path.clone(),
        model_name: req.model_name.clone(),
        label_field: req.label_field.clone(),
        prompt_field: req.prompt_field.clone(),
        system_prompt: req.system_prompt.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        output_path: Some(output_path.clone()),
    };

    let job_id = state
        .auto_labeler
        .create_job(config)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Spawn background task
    let labeler = state.auto_labeler.clone();
    let backends = state.backends.clone();
    let model_manager = state.model_manager.clone();
    let model_name = req.model_name.clone();
    let source_path = req.source_path;
    let label_field = req.label_field;
    let prompt_field = req.prompt_field;
    let system_prompt = req.system_prompt;
    let max_tokens = req.max_tokens.unwrap_or(256);
    let temperature = req.temperature.unwrap_or(0.0);

    tokio::spawn(async move {
        if let Err(e) = labeler.start_job(job_id).await {
            tracing::error!(error = %e, "Failed to start auto-label job");
            return;
        }

        let infer_fn = |prompt: String| {
            let backends = backends.clone();
            let model_manager = model_manager.clone();
            let model_name = model_name.clone();
            async move {
                let loaded = model_manager.list_loaded(None).await;
                let loaded_model = loaded
                    .iter()
                    .find(|m| m.model_name == model_name)
                    .or_else(|| loaded.first())
                    .ok_or_else(|| {
                        synapse_types::SynapseError::TrainingError(
                            "No model loaded for auto-labeling".into(),
                        )
                    })?;

                let backend_id = synapse_types::backend::BackendId(loaded_model.backend_id.clone());
                let backend = backends.get(&backend_id).ok_or_else(|| {
                    synapse_types::SynapseError::TrainingError(format!(
                        "Backend '{}' not available",
                        loaded_model.backend_id
                    ))
                })?;

                let handle = synapse_backends::ModelHandle(loaded_model.handle.clone());
                let req = InferenceRequest {
                    prompt,
                    max_tokens: Some(max_tokens),
                    temperature: Some(temperature),
                    top_p: None,
                    top_k: None,
                    stop_sequences: None,
                    system_prompt: None,
                    sensitivity: None,
                };

                let resp = backend.infer(&handle, &req).await?;
                Ok(resp.text)
            }
        };

        match synapse_train::dataset::labeler::run_labeling(
            &source_path,
            &output_path,
            &prompt_field,
            &label_field,
            system_prompt.as_deref(),
            &labeler,
            job_id,
            infer_fn,
        )
        .await
        {
            Ok(path) => {
                let _ = labeler.complete_job(job_id, path).await;
            }
            Err(e) => {
                tracing::error!(error = %e, "Auto-label job failed");
                let _ = labeler.fail_job(job_id, e.to_string()).await;
            }
        }
    });

    let job = state
        .auto_labeler
        .get_job(job_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(job_to_auto_label_response(&job))))
}

/// GET /datasets/auto-label/jobs — list all auto-labeling jobs.
pub async fn list_auto_label_jobs(
    State(state): State<AppState>,
) -> Json<Vec<AutoLabelJobResponse>> {
    let jobs = state.auto_labeler.list_jobs().await;
    Json(jobs.iter().map(job_to_auto_label_response).collect())
}

/// GET /datasets/auto-label/jobs/:id — get auto-labeling job status.
pub async fn get_auto_label_job(
    State(state): State<AppState>,
    Path(id): Path<AutoLabelJobId>,
) -> Result<Json<AutoLabelJobResponse>, (StatusCode, String)> {
    let job = state
        .auto_labeler
        .get_job(id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(Json(job_to_auto_label_response(&job)))
}

fn job_to_auto_label_response(
    job: &synapse_train::dataset::labeler::AutoLabelJobState,
) -> AutoLabelJobResponse {
    AutoLabelJobResponse {
        job_id: job.job_id,
        status: job.status,
        labeled_count: job.labeled_count,
        total_count: job.total_count,
        output_path: job.output_path.clone(),
        error: job.error.clone(),
    }
}

// --- Augmentation ---

/// Request to augment a dataset.
#[derive(Deserialize)]
pub struct AugmentRequest {
    pub input_path: String,
    pub output_path: Option<String>,
    pub strategies: Vec<AugmentationStrategy>,
    #[serde(default = "default_factor")]
    pub augment_factor: usize,
    #[serde(default = "default_text_field_name")]
    pub text_field: String,
    pub word_probability: Option<f64>,
    pub seed: Option<u64>,
}

fn default_factor() -> usize {
    1
}
fn default_text_field_name() -> String {
    "text".into()
}

/// Response for an augmentation operation.
#[derive(Serialize)]
pub struct AugmentResponse {
    pub original_count: usize,
    pub augmented_count: usize,
    pub output_path: String,
}

/// POST /datasets/augment — augment a dataset with text augmentation strategies.
pub async fn augment_dataset(
    State(_state): State<AppState>,
    Json(req): Json<AugmentRequest>,
) -> Result<Json<AugmentResponse>, (StatusCode, String)> {
    let output_path = req.output_path.unwrap_or_else(|| {
        format!(
            "{}.augmented.jsonl",
            req.input_path.trim_end_matches(".jsonl")
        )
    });

    let config = synapse_train::dataset::processor::AugmentationConfig {
        strategies: req.strategies,
        augment_factor: req.augment_factor,
        text_field: req.text_field,
        preserve_labels: true,
        word_probability: req.word_probability.unwrap_or(0.1),
        seed: req.seed,
    };

    let input = req.input_path.clone();
    let out = output_path.clone();

    // Run in blocking task since it's CPU-bound file I/O
    let result = tokio::task::spawn_blocking(move || {
        synapse_train::dataset::processor::augment_dataset(
            std::path::Path::new(&input),
            std::path::Path::new(&out),
            &config,
        )
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
    })?
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    Ok(Json(AugmentResponse {
        original_count: result.original_count,
        augmented_count: result.augmented_count,
        output_path: result.output_path,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_auto_label_request_deserialize() {
        let json = r#"{
            "source_path": "/data/unlabeled.jsonl",
            "model_name": "llama-7b",
            "system_prompt": "Classify sentiment as positive or negative"
        }"#;
        let req: CreateAutoLabelRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.source_path, "/data/unlabeled.jsonl");
        assert_eq!(req.model_name, "llama-7b");
        assert_eq!(req.label_field, "expected");
        assert_eq!(req.prompt_field, "prompt");
        assert!(req.system_prompt.is_some());
    }

    #[test]
    fn auto_label_response_serializes() {
        let resp = AutoLabelJobResponse {
            job_id: uuid::Uuid::nil(),
            status: AutoLabelStatus::Running,
            labeled_count: 50,
            total_count: 100,
            output_path: Some("/output.jsonl".into()),
            error: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["labeled_count"], 50);
        assert_eq!(json["total_count"], 100);
        assert_eq!(json["status"], "running");
    }

    #[test]
    fn augment_request_deserialize() {
        let json = r#"{
            "input_path": "/data/train.jsonl",
            "strategies": ["synonym_replacement", "random_deletion"],
            "augment_factor": 3,
            "word_probability": 0.2
        }"#;
        let req: AugmentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.strategies.len(), 2);
        assert_eq!(req.augment_factor, 3);
        assert_eq!(req.text_field, "text");
    }

    #[test]
    fn augment_response_serializes() {
        let resp = AugmentResponse {
            original_count: 100,
            augmented_count: 250,
            output_path: "/out.jsonl".into(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["original_count"], 100);
        assert_eq!(json["augmented_count"], 250);
    }
}
