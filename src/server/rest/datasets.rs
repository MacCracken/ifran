//! REST handlers for dataset operations — auto-labeling and augmentation.

use crate::train::dataset::labeler::{AutoLabelJobId, AutoLabelStatus};
use crate::train::dataset::processor::AugmentationStrategy;
use crate::types::inference::InferenceRequest;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};

use super::pagination::{PaginatedResponse, PaginationQuery};
use crate::server::state::AppState;

fn validate_path(path: &str) -> Result<(), (StatusCode, String)> {
    if path.contains("..") {
        return Err((
            StatusCode::BAD_REQUEST,
            "path must not contain '..' components".into(),
        ));
    }
    Ok(())
}

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
#[derive(Debug, Serialize)]
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
    validate_path(&req.source_path)?;
    if let Some(ref p) = req.output_path {
        validate_path(p)?;
    }

    let output_path = req.output_path.unwrap_or_else(|| {
        format!(
            "{}.labeled.jsonl",
            req.source_path.trim_end_matches(".jsonl")
        )
    });

    let config = crate::train::dataset::labeler::AutoLabelConfig {
        source_path: req.source_path.clone(),
        model_name: req.model_name.clone(),
        label_field: req.label_field.clone(),
        prompt_field: req.prompt_field.clone(),
        system_prompt: req.system_prompt.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        output_path: Some(output_path.clone()),
    };

    let label_config = config.clone();

    let job_id = state
        .auto_labeler
        .create_job(config)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Spawn background task
    let labeler = state.auto_labeler.clone();
    let backends = state.backends.clone();
    let model_manager = state.model_manager.clone();
    let model_name = label_config.model_name.clone();
    let max_tokens = label_config.max_tokens.unwrap_or(256);
    let temperature = label_config.temperature.unwrap_or(0.0);

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
                        crate::types::IfranError::TrainingError(
                            "No model loaded for auto-labeling".into(),
                        )
                    })?;

                let backend_id = crate::types::backend::BackendId(loaded_model.backend_id.clone());
                let backend = backends.get(&backend_id).ok_or_else(|| {
                    crate::types::IfranError::TrainingError(format!(
                        "Backend '{}' not available",
                        loaded_model.backend_id
                    ))
                })?;

                let handle = crate::backends::ModelHandle(loaded_model.handle.clone());
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

        match crate::train::dataset::labeler::run_labeling(
            &label_config,
            &output_path,
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

/// GET /datasets/auto-label/jobs — list auto-labeling jobs with pagination.
pub async fn list_auto_label_jobs(
    State(state): State<AppState>,
    Query(page): Query<PaginationQuery>,
) -> Json<PaginatedResponse<AutoLabelJobResponse>> {
    let jobs = state.auto_labeler.list_jobs().await;
    Json(PaginatedResponse::from_slice(
        &jobs,
        &page,
        job_to_auto_label_response,
    ))
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
    job: &crate::train::dataset::labeler::AutoLabelJobState,
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
#[derive(Debug, Serialize)]
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
    validate_path(&req.input_path)?;
    if let Some(ref p) = req.output_path {
        validate_path(p)?;
    }

    if req.augment_factor < 1 || req.augment_factor > 100 {
        return Err((
            StatusCode::BAD_REQUEST,
            "augment_factor must be between 1 and 100".into(),
        ));
    }

    let output_path = req.output_path.unwrap_or_else(|| {
        format!(
            "{}.augmented.jsonl",
            req.input_path.trim_end_matches(".jsonl")
        )
    });

    let config = crate::train::dataset::processor::AugmentationConfig {
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
        crate::train::dataset::processor::augment_dataset(
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

// --- Validation & Preview ---

/// Request to validate a dataset file.
#[derive(Deserialize)]
pub struct ValidateDatasetRequest {
    pub path: String,
    pub format: crate::types::training::DatasetFormat,
}

/// Response for dataset validation.
#[derive(Debug, Serialize)]
pub struct ValidateDatasetResponse {
    pub valid: bool,
    pub total_rows: usize,
    pub invalid_rows: usize,
    pub errors: Vec<String>,
}

/// POST /datasets/validate — validate a dataset file for format compliance.
pub async fn validate_dataset(
    State(_state): State<AppState>,
    Json(req): Json<ValidateDatasetRequest>,
) -> Result<Json<ValidateDatasetResponse>, (StatusCode, String)> {
    validate_path(&req.path)?;

    let path = req.path.clone();
    let format = req.format;

    let result = tokio::task::spawn_blocking(move || {
        crate::train::dataset::validator::validate(std::path::Path::new(&path), format)
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
    })?
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    Ok(Json(ValidateDatasetResponse {
        valid: result.valid,
        total_rows: result.total_rows,
        invalid_rows: result.invalid_rows,
        errors: result.errors,
    }))
}

/// Request to preview a dataset file.
#[derive(Deserialize)]
pub struct PreviewDatasetRequest {
    pub path: String,
    pub format: crate::types::training::DatasetFormat,
    #[serde(default = "default_preview_limit")]
    pub limit: usize,
}

fn default_preview_limit() -> usize {
    5
}

/// Response for dataset preview.
#[derive(Debug, Serialize)]
pub struct PreviewDatasetResponse {
    pub samples: Vec<serde_json::Value>,
    pub has_more: bool,
    pub format: crate::types::training::DatasetFormat,
}

/// POST /datasets/preview — preview first N rows of a dataset file.
pub async fn preview_dataset(
    State(_state): State<AppState>,
    Json(req): Json<PreviewDatasetRequest>,
) -> Result<Json<PreviewDatasetResponse>, (StatusCode, String)> {
    validate_path(&req.path)?;

    let path = req.path.clone();
    let format = req.format;
    let limit = req.limit.clamp(1, 50);

    let result =
        tokio::task::spawn_blocking(move || -> Result<(Vec<serde_json::Value>, bool), String> {
            use std::io::BufRead;
            let file =
                std::fs::File::open(&path).map_err(|e| format!("Failed to open file: {e}"))?;
            let reader = std::io::BufReader::new(file);

            match format {
                crate::types::training::DatasetFormat::Jsonl => {
                    let mut samples = Vec::with_capacity(limit);
                    let mut has_more = false;
                    for line in reader.lines() {
                        let line = line.map_err(|e| format!("Read error: {e}"))?;
                        if line.trim().is_empty() {
                            continue;
                        }
                        if samples.len() >= limit {
                            has_more = true;
                            break;
                        }
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line.trim()) {
                            samples.push(val);
                        }
                    }
                    Ok((samples, has_more))
                }
                crate::types::training::DatasetFormat::Csv => {
                    let mut lines_iter = reader.lines();
                    let header_line = match lines_iter.next() {
                        Some(Ok(h)) if !h.trim().is_empty() => h,
                        _ => return Ok((Vec::new(), false)),
                    };
                    let headers: Vec<String> = header_line
                        .split(',')
                        .map(|h| h.trim().to_string())
                        .collect();
                    let mut samples = Vec::with_capacity(limit);
                    let mut has_more = false;
                    for line in lines_iter {
                        let line = line.map_err(|e| format!("Read error: {e}"))?;
                        if line.trim().is_empty() {
                            continue;
                        }
                        if samples.len() >= limit {
                            has_more = true;
                            break;
                        }
                        let values: Vec<&str> = line.split(',').collect();
                        let mut obj = serde_json::Map::new();
                        for (i, header) in headers.iter().enumerate() {
                            let val = values.get(i).unwrap_or(&"");
                            obj.insert(header.clone(), serde_json::Value::String(val.to_string()));
                        }
                        samples.push(serde_json::Value::Object(obj));
                    }
                    Ok((samples, has_more))
                }
                _ => Err(format!("Preview not supported for format {:?}", format)),
            }
        })
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Task failed: {e}"),
            )
        })?
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    Ok(Json(PreviewDatasetResponse {
        samples: result.0,
        has_more: result.1,
        format,
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

    #[test]
    fn validate_request_deserialize() {
        let json = r#"{"path": "/data/train.jsonl", "format": "jsonl"}"#;
        let req: ValidateDatasetRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.path, "/data/train.jsonl");
        assert_eq!(req.format, crate::types::training::DatasetFormat::Jsonl);
    }

    #[test]
    fn validate_request_csv_format() {
        let json = r#"{"path": "/data/train.csv", "format": "csv"}"#;
        let req: ValidateDatasetRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.format, crate::types::training::DatasetFormat::Csv);
    }

    #[test]
    fn validate_response_serializes() {
        let resp = ValidateDatasetResponse {
            valid: true,
            total_rows: 1000,
            invalid_rows: 2,
            errors: vec!["Line 5: invalid json".into()],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["valid"], true);
        assert_eq!(json["total_rows"], 1000);
        assert_eq!(json["invalid_rows"], 2);
        assert_eq!(json["errors"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn preview_request_deserialize_defaults() {
        let json = r#"{"path": "/data/train.jsonl", "format": "jsonl"}"#;
        let req: PreviewDatasetRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.path, "/data/train.jsonl");
        assert_eq!(req.limit, 5); // default
    }

    #[test]
    fn preview_request_deserialize_with_limit() {
        let json = r#"{"path": "/data/train.jsonl", "format": "jsonl", "limit": 10}"#;
        let req: PreviewDatasetRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.limit, 10);
    }

    #[test]
    fn preview_response_serializes() {
        let samples = vec![
            serde_json::json!({"text": "hello"}),
            serde_json::json!({"text": "world"}),
        ];
        let resp = PreviewDatasetResponse {
            samples,
            has_more: true,
            format: crate::types::training::DatasetFormat::Jsonl,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["has_more"], true);
        assert_eq!(json["format"], "jsonl");
        assert_eq!(json["samples"].as_array().unwrap().len(), 2);
    }

    // --- validate_path tests ---

    #[test]
    fn validate_path_rejects_dot_dot() {
        let err = validate_path("/data/../etc/passwd").unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.contains(".."));
    }

    #[test]
    fn validate_path_accepts_normal() {
        assert!(validate_path("/data/train.jsonl").is_ok());
        assert!(validate_path("relative/path.jsonl").is_ok());
    }

    #[test]
    fn validate_path_rejects_embedded_dot_dot() {
        assert!(validate_path("foo/..bar").is_err());
        assert!(validate_path("a/../b").is_err());
    }

    // --- CreateAutoLabelRequest edge cases ---

    #[test]
    fn auto_label_request_missing_required_fields() {
        let json = r#"{"source_path": "/data.jsonl"}"#;
        let result = serde_json::from_str::<CreateAutoLabelRequest>(json);
        assert!(result.is_err(), "model_name is required");
    }

    #[test]
    fn auto_label_request_all_optional_fields() {
        let json = r#"{
            "source_path": "/data.jsonl",
            "model_name": "test",
            "label_field": "label",
            "prompt_field": "input",
            "system_prompt": "classify",
            "output_path": "/out.jsonl",
            "max_tokens": 128,
            "temperature": 0.5
        }"#;
        let req: CreateAutoLabelRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.label_field, "label");
        assert_eq!(req.prompt_field, "input");
        assert_eq!(req.output_path, Some("/out.jsonl".into()));
        assert_eq!(req.max_tokens, Some(128));
        assert_eq!(req.temperature, Some(0.5));
    }

    // --- AugmentRequest edge cases ---

    #[test]
    fn augment_request_defaults() {
        let json = r#"{
            "input_path": "/data.jsonl",
            "strategies": ["synonym_replacement"]
        }"#;
        let req: AugmentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.augment_factor, 1);
        assert_eq!(req.text_field, "text");
        assert!(req.output_path.is_none());
        assert!(req.word_probability.is_none());
        assert!(req.seed.is_none());
    }

    #[test]
    fn augment_request_missing_strategies() {
        let json = r#"{"input_path": "/data.jsonl"}"#;
        let result = serde_json::from_str::<AugmentRequest>(json);
        assert!(result.is_err(), "strategies is required");
    }

    #[test]
    fn augment_request_empty_strategies() {
        let json = r#"{"input_path": "/data.jsonl", "strategies": []}"#;
        let req: AugmentRequest = serde_json::from_str(json).unwrap();
        assert!(req.strategies.is_empty());
    }

    // --- ValidateDatasetRequest edge cases ---

    #[test]
    fn validate_request_missing_format() {
        let json = r#"{"path": "/data.jsonl"}"#;
        let result = serde_json::from_str::<ValidateDatasetRequest>(json);
        assert!(result.is_err(), "format is required");
    }

    // --- PreviewDatasetRequest edge cases ---

    #[test]
    fn preview_request_custom_limit() {
        let json = r#"{"path": "/data.csv", "format": "csv", "limit": 25}"#;
        let req: PreviewDatasetRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.limit, 25);
        assert_eq!(req.format, crate::types::training::DatasetFormat::Csv);
    }

    // --- Handler tests using test_state ---

    use crate::server::test_helpers::helpers::test_state;

    #[tokio::test]
    async fn list_auto_label_jobs_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let page = super::super::pagination::PaginationQuery {
            limit: super::super::pagination::DEFAULT_LIMIT,
            offset: 0,
        };
        let result = list_auto_label_jobs(State(state), Query(page)).await;
        assert!(result.0.data.is_empty());
    }

    #[tokio::test]
    async fn get_auto_label_job_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let fake_id = uuid::Uuid::new_v4();
        let result = get_auto_label_job(State(state), Path(fake_id)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn create_auto_label_rejects_path_traversal() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let req = CreateAutoLabelRequest {
            source_path: "/data/../etc/passwd".into(),
            model_name: "test".into(),
            label_field: "expected".into(),
            prompt_field: "prompt".into(),
            system_prompt: None,
            output_path: None,
            max_tokens: None,
            temperature: None,
        };
        let result = create_auto_label(State(state), Json(req)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn create_auto_label_rejects_output_path_traversal() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let req = CreateAutoLabelRequest {
            source_path: "/data/clean.jsonl".into(),
            model_name: "test".into(),
            label_field: "expected".into(),
            prompt_field: "prompt".into(),
            system_prompt: None,
            output_path: Some("/tmp/../etc/shadow".into()),
            max_tokens: None,
            temperature: None,
        };
        let result = create_auto_label(State(state), Json(req)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn augment_rejects_path_traversal() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let req = AugmentRequest {
            input_path: "/data/../secrets".into(),
            output_path: None,
            strategies: vec![],
            augment_factor: 1,
            text_field: "text".into(),
            word_probability: None,
            seed: None,
        };
        let result = augment_dataset(State(state), Json(req)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn augment_rejects_factor_too_high() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let req = AugmentRequest {
            input_path: "/data/ok.jsonl".into(),
            output_path: None,
            strategies: vec![],
            augment_factor: 200,
            text_field: "text".into(),
            word_probability: None,
            seed: None,
        };
        let result = augment_dataset(State(state), Json(req)).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.contains("augment_factor"));
    }

    #[tokio::test]
    async fn augment_rejects_factor_zero() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let req = AugmentRequest {
            input_path: "/data/ok.jsonl".into(),
            output_path: None,
            strategies: vec![],
            augment_factor: 0,
            text_field: "text".into(),
            word_probability: None,
            seed: None,
        };
        let result = augment_dataset(State(state), Json(req)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn validate_rejects_path_traversal() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let req = ValidateDatasetRequest {
            path: "/data/../etc/passwd".into(),
            format: crate::types::training::DatasetFormat::Jsonl,
        };
        let result = validate_dataset(State(state), Json(req)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn preview_rejects_path_traversal() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let req = PreviewDatasetRequest {
            path: "../../etc/passwd".into(),
            format: crate::types::training::DatasetFormat::Jsonl,
            limit: 5,
        };
        let result = preview_dataset(State(state), Json(req)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn validate_nonexistent_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let req = ValidateDatasetRequest {
            path: "/nonexistent/file.jsonl".into(),
            format: crate::types::training::DatasetFormat::Jsonl,
        };
        let result = validate_dataset(State(state), Json(req)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn preview_nonexistent_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let req = PreviewDatasetRequest {
            path: "/nonexistent/file.jsonl".into(),
            format: crate::types::training::DatasetFormat::Jsonl,
            limit: 5,
        };
        let result = preview_dataset(State(state), Json(req)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn preview_jsonl_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let file_path = tmp.path().join("test.jsonl");
        std::fs::write(
            &file_path,
            "{\"text\":\"hello\"}\n{\"text\":\"world\"}\n{\"text\":\"foo\"}\n",
        )
        .unwrap();
        let req = PreviewDatasetRequest {
            path: file_path.to_string_lossy().to_string(),
            format: crate::types::training::DatasetFormat::Jsonl,
            limit: 2,
        };
        let result = preview_dataset(State(state), Json(req)).await.unwrap();
        assert_eq!(result.0.samples.len(), 2);
        assert!(result.0.has_more);
        assert_eq!(result.0.samples[0]["text"], "hello");
    }

    #[tokio::test]
    async fn preview_csv_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let file_path = tmp.path().join("test.csv");
        std::fs::write(&file_path, "name,age\nAlice,30\nBob,25\n").unwrap();
        let req = PreviewDatasetRequest {
            path: file_path.to_string_lossy().to_string(),
            format: crate::types::training::DatasetFormat::Csv,
            limit: 10,
        };
        let result = preview_dataset(State(state), Json(req)).await.unwrap();
        assert_eq!(result.0.samples.len(), 2);
        assert!(!result.0.has_more);
        assert_eq!(result.0.samples[0]["name"], "Alice");
        assert_eq!(result.0.samples[0]["age"], "30");
    }

    #[tokio::test]
    async fn preview_csv_empty_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let file_path = tmp.path().join("empty.csv");
        std::fs::write(&file_path, "").unwrap();
        let req = PreviewDatasetRequest {
            path: file_path.to_string_lossy().to_string(),
            format: crate::types::training::DatasetFormat::Csv,
            limit: 5,
        };
        let result = preview_dataset(State(state), Json(req)).await.unwrap();
        assert!(result.0.samples.is_empty());
    }

    #[tokio::test]
    async fn preview_limit_clamped() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let file_path = tmp.path().join("big.jsonl");
        // Write 60 lines but request limit=100 (should be clamped to 50)
        let content: String = (0..60).map(|i| format!("{{\"i\":{i}}}\n")).collect();
        std::fs::write(&file_path, &content).unwrap();
        let req = PreviewDatasetRequest {
            path: file_path.to_string_lossy().to_string(),
            format: crate::types::training::DatasetFormat::Jsonl,
            limit: 100,
        };
        let result = preview_dataset(State(state), Json(req)).await.unwrap();
        assert_eq!(result.0.samples.len(), 50);
        assert!(result.0.has_more);
    }
}
