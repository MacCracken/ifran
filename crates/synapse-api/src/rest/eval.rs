//! REST handlers for model evaluation.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use synapse_types::eval::*;
use synapse_types::inference::InferenceRequest;

use crate::state::AppState;

/// Response for an eval run.
#[derive(Serialize)]
pub struct EvalRunResponse {
    pub run_id: EvalRunId,
    pub model_name: String,
    pub status: EvalStatus,
    pub benchmarks: Vec<BenchmarkKind>,
    pub results: Vec<EvalResultResponse>,
    pub error: Option<String>,
}

/// Response for an eval result.
#[derive(Serialize)]
pub struct EvalResultResponse {
    pub benchmark: BenchmarkKind,
    pub score: f64,
    pub samples_evaluated: u64,
    pub duration_secs: f64,
    pub evaluated_at: String,
}

/// Request to create an eval run.
#[derive(Deserialize)]
pub struct CreateEvalRequest {
    pub model_name: String,
    pub benchmarks: Vec<BenchmarkKind>,
    pub sample_limit: Option<usize>,
    pub dataset_path: Option<String>,
}

/// POST /eval/runs — create a new eval run and execute benchmarks.
pub async fn create_run(
    State(state): State<AppState>,
    Json(req): Json<CreateEvalRequest>,
) -> Result<(StatusCode, Json<EvalRunResponse>), (StatusCode, String)> {
    let config = EvalConfig {
        model_name: req.model_name.clone(),
        benchmarks: req.benchmarks.clone(),
        sample_limit: req.sample_limit,
        dataset_path: req.dataset_path.clone(),
    };

    let run_id = state
        .eval_runner
        .create_run(config)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // If dataset provided, spawn background eval execution
    if let Some(ref dataset_path) = req.dataset_path {
        let runner = state.eval_runner.clone();
        let backends = state.backends.clone();
        let model_manager = state.model_manager.clone();
        let benchmarks = req.benchmarks.clone();
        let model_name = req.model_name.clone();
        let sample_limit = req.sample_limit;
        let dataset = dataset_path.clone();

        tokio::spawn(async move {
            if let Err(e) = runner.start_run(run_id).await {
                tracing::error!(error = %e, "Failed to start eval run");
                return;
            }

            // Build inference closure from backend
            let infer_fn = |prompt: String| {
                let backends = backends.clone();
                let model_manager = model_manager.clone();
                async move {
                    let loaded = model_manager.list_loaded().await;
                    let loaded_model = loaded.first().ok_or_else(|| {
                        synapse_types::SynapseError::EvalError(
                            "No model loaded for evaluation".into(),
                        )
                    })?;

                    let backend_id =
                        synapse_types::backend::BackendId(loaded_model.backend_id.clone());
                    let backend = backends.get(&backend_id).ok_or_else(|| {
                        synapse_types::SynapseError::EvalError(format!(
                            "Backend '{}' not available",
                            loaded_model.backend_id
                        ))
                    })?;

                    let handle = synapse_backends::ModelHandle(loaded_model.handle.clone());
                    let req = InferenceRequest {
                        prompt,
                        max_tokens: Some(256),
                        temperature: Some(0.0),
                        top_p: None,
                        top_k: None,
                        stop_sequences: None,
                        system_prompt: None,
                    };

                    let resp = backend.infer(&handle, &req).await?;
                    Ok(resp.text)
                }
            };

            let mut all_ok = true;
            for kind in &benchmarks {
                match runner
                    .run_benchmark(
                        run_id,
                        *kind,
                        &dataset,
                        sample_limit,
                        &model_name,
                        &infer_fn,
                    )
                    .await
                {
                    Ok(result) => {
                        tracing::info!(
                            benchmark = ?kind,
                            score = result.score,
                            "Benchmark completed"
                        );
                    }
                    Err(e) => {
                        tracing::error!(benchmark = ?kind, error = %e, "Benchmark failed");
                        all_ok = false;
                        let _ = runner.fail_run(run_id, e.to_string()).await;
                        break;
                    }
                }
            }

            if all_ok {
                let _ = runner.complete_run(run_id).await;
            }
        });
    }

    let run = state
        .eval_runner
        .get_run(run_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(run_to_response(&run))))
}

/// GET /eval/runs — list all eval runs.
pub async fn list_runs(State(state): State<AppState>) -> Json<Vec<EvalRunResponse>> {
    let runs = state.eval_runner.list_runs().await;
    Json(runs.iter().map(run_to_response).collect())
}

/// GET /eval/runs/:id — get a specific eval run.
pub async fn get_run(
    State(state): State<AppState>,
    Path(id): Path<EvalRunId>,
) -> Result<Json<EvalRunResponse>, (StatusCode, String)> {
    let run = state
        .eval_runner
        .get_run(id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(Json(run_to_response(&run)))
}

fn run_to_response(run: &synapse_core::eval::runner::EvalRunState) -> EvalRunResponse {
    EvalRunResponse {
        run_id: run.run_id,
        model_name: run.config.model_name.clone(),
        status: run.status,
        benchmarks: run.config.benchmarks.clone(),
        results: run
            .results
            .iter()
            .map(|r| EvalResultResponse {
                benchmark: r.benchmark,
                score: r.score,
                samples_evaluated: r.samples_evaluated,
                duration_secs: r.duration_secs,
                evaluated_at: r.evaluated_at.to_rfc3339(),
            })
            .collect(),
        error: run.error.clone(),
    }
}
