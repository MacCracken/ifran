//! REST handlers for experiment management.

use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use ifran_types::experiment::{ExperimentId, ExperimentProgram, ExperimentStatus, TrialResult};

use super::pagination::{PaginatedResponse, PaginationQuery};

use ifran_types::TenantId;

use crate::state::AppState;

/// Response for an experiment.
#[derive(Debug, Serialize)]
pub struct ExperimentResponse {
    pub id: ExperimentId,
    pub name: String,
    pub status: ExperimentStatus,
    pub best_score: Option<f64>,
    pub trials: Vec<TrialResponse>,
}

/// Response for a trial.
#[derive(Debug, Serialize)]
pub struct TrialResponse {
    pub trial_id: String,
    pub trial_number: u32,
    pub status: String,
    pub train_loss: Option<f64>,
    pub eval_score: Option<f64>,
    pub duration_secs: Option<f64>,
    pub learning_rate: f64,
    pub is_best: bool,
}

/// Request body for creating an experiment.
#[derive(Deserialize)]
pub struct CreateExperimentRequest {
    #[serde(flatten)]
    pub program: ExperimentProgram,
}

/// Response for experiment list.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentListItem {
    pub id: ExperimentId,
    pub name: String,
    pub status: ExperimentStatus,
    pub best_score: Option<f64>,
}

fn trial_to_response(t: &TrialResult) -> TrialResponse {
    TrialResponse {
        trial_id: t.trial_id.to_string(),
        trial_number: t.trial_number,
        status: serde_json::to_string(&t.status)
            .unwrap_or_else(|_| "unknown".into())
            .trim_matches('"')
            .to_string(),
        train_loss: t.train_loss,
        eval_score: t.eval_score,
        duration_secs: t.duration_secs,
        learning_rate: t.hyperparams.learning_rate,
        is_best: t.is_best,
    }
}

/// POST /experiments — create and start an experiment from a program JSON.
pub async fn create_experiment(
    State(state): State<AppState>,
    Extension(_tenant_id): Extension<TenantId>,
    Json(req): Json<CreateExperimentRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let store = state.experiment_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Experiment store not initialized".into(),
    ))?;

    let handle = ifran_train::experiment::runner::ExperimentRunner::start(
        state.job_manager.clone(),
        store.clone(),
        req.program,
    )
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Store the handle for later stop
    let id = handle.experiment_id;
    state.experiment_runners.lock().await.insert(id, handle);

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "id": id.to_string(), "status": "running" })),
    ))
}

/// Query parameters for listing experiments.
#[derive(Debug, Deserialize)]
pub struct ListExperimentsQuery {
    #[serde(flatten)]
    pub page: PaginationQuery,
    /// Optional status filter, e.g. `?status=running`.
    pub status: Option<ExperimentStatus>,
}

/// GET /experiments — list all experiments.
pub async fn list_experiments(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Query(query): Query<ListExperimentsQuery>,
) -> Result<Json<PaginatedResponse<ExperimentListItem>>, (StatusCode, String)> {
    let store = state.experiment_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Experiment store not initialized".into(),
    ))?;

    let s = store.lock().await;
    let experiments = s
        .list_experiments(&tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let items: Vec<ExperimentListItem> = experiments
        .into_iter()
        .filter(|(_, _, status, _)| query.status.is_none() || Some(*status) == query.status)
        .map(|(id, name, status, best_score)| ExperimentListItem {
            id,
            name,
            status,
            best_score,
        })
        .collect();

    Ok(Json(PaginatedResponse::from_slice(
        &items,
        &query.page,
        |item| item.clone(),
    )))
}

/// GET /experiments/{id} — get experiment detail with trials.
pub async fn get_experiment(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<ExperimentId>,
) -> Result<Json<ExperimentResponse>, (StatusCode, String)> {
    let store = state.experiment_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Experiment store not initialized".into(),
    ))?;

    let s = store.lock().await;
    let (exp_id, name, _, status, _, best_score) = s
        .get_experiment(id, &tenant_id)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let trials = s
        .get_trials(id, &tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(ExperimentResponse {
        id: exp_id,
        name,
        status,
        best_score,
        trials: trials.iter().map(trial_to_response).collect(),
    }))
}

/// GET /experiments/{id}/leaderboard — trials ranked by objective.
pub async fn get_leaderboard(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<ExperimentId>,
) -> Result<Json<Vec<TrialResponse>>, (StatusCode, String)> {
    let store = state.experiment_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Experiment store not initialized".into(),
    ))?;

    let s = store.lock().await;
    let (_, _, program, _, _, _) = s
        .get_experiment(id, &tenant_id)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let direction = program.objective.direction;
    let trials = s
        .get_leaderboard(id, direction, 50, &tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(trials.iter().map(trial_to_response).collect()))
}

/// POST /experiments/{id}/stop — stop a running experiment.
pub async fn stop_experiment(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<ExperimentId>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Try to stop via the in-memory handle first
    {
        let runners = state.experiment_runners.lock().await;
        if let Some(handle) = runners.get(&id) {
            handle.stop();
        }
    }

    // Also mark in the store
    if let Some(store) = &state.experiment_store {
        let s = store.lock().await;
        let _ = s.update_experiment_status(id, ExperimentStatus::Stopped, &tenant_id);
    }

    Ok(Json(
        serde_json::json!({ "id": id.to_string(), "status": "stopped" }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trial_response_serializes() {
        let resp = TrialResponse {
            trial_id: "abc".into(),
            trial_number: 1,
            status: "completed".into(),
            train_loss: Some(0.42),
            eval_score: Some(5.23),
            duration_secs: Some(295.0),
            learning_rate: 1e-4,
            is_best: true,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["trial_number"], 1);
        assert!(json["is_best"].as_bool().unwrap());
    }

    #[test]
    fn experiment_list_item_serializes() {
        let item = ExperimentListItem {
            id: uuid::Uuid::new_v4(),
            name: "test".into(),
            status: ExperimentStatus::Running,
            best_score: Some(3.25),
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["name"], "test");
        assert_eq!(json["status"], "running");
    }

    #[test]
    fn create_request_deserialize() {
        let json = r#"{
            "name": "lr-sweep",
            "base_model": "llama-8b",
            "dataset_path": "/data/train.jsonl",
            "method": "lora",
            "time_budget_secs": 300,
            "objective": {"metric": "perplexity", "direction": "minimize"},
            "search": {"strategy": "grid"},
            "search_space": [{"name": "learning_rate", "values": [0.00001, 0.0001]}],
            "base_hyperparams": {
                "learning_rate": 0.0002,
                "epochs": 3,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "max_seq_length": 2048
            }
        }"#;
        let req: CreateExperimentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.program.name, "lr-sweep");
        assert_eq!(req.program.time_budget_secs, 300);
    }
}
