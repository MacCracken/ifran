//! REST handlers for experiment management.

use crate::types::experiment::{ExperimentId, ExperimentProgram, ExperimentStatus, TrialResult};
use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};

use super::pagination::{PaginatedResponse, PaginationQuery};

use crate::types::TenantId;

use crate::server::state::AppState;

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

    let handle = crate::train::experiment::runner::ExperimentRunner::start(
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

    let safe_limit = query.page.safe_limit();
    let paged = store
        .list_experiments(&tenant_id, safe_limit, query.page.offset)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let items: Vec<ExperimentListItem> = paged
        .items
        .into_iter()
        .filter(|(_, _, status, _)| query.status.is_none() || Some(*status) == query.status)
        .map(|(id, name, status, best_score)| ExperimentListItem {
            id,
            name,
            status,
            best_score,
        })
        .collect();

    Ok(Json(PaginatedResponse::pre_sliced(
        items,
        paged.total,
        safe_limit,
        query.page.offset,
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

    let (exp_id, name, _, status, _, best_score) = store
        .get_experiment(id, &tenant_id)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let trials = store
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

    let (_, _, program, _, _, _) = store
        .get_experiment(id, &tenant_id)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let direction = program.objective.direction;
    let trials = store
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
        let _ = store.update_experiment_status(id, ExperimentStatus::Stopped, &tenant_id);
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

    #[test]
    fn create_request_missing_required_fields() {
        // Missing "name" which is required by ExperimentProgram
        let json = r#"{
            "base_model": "llama-8b",
            "dataset_path": "/data/train.jsonl",
            "method": "lora",
            "time_budget_secs": 300,
            "objective": {"metric": "perplexity", "direction": "minimize"},
            "search": {"strategy": "grid"},
            "search_space": [],
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
        let result = serde_json::from_str::<CreateExperimentRequest>(json);
        assert!(result.is_err());
    }

    #[test]
    fn experiment_response_serializes() {
        let resp = ExperimentResponse {
            id: uuid::Uuid::nil(),
            name: "test-exp".into(),
            status: ExperimentStatus::Completed,
            best_score: Some(1.23),
            trials: vec![TrialResponse {
                trial_id: "t-1".into(),
                trial_number: 0,
                status: "completed".into(),
                train_loss: Some(0.5),
                eval_score: Some(1.23),
                duration_secs: Some(60.0),
                learning_rate: 1e-4,
                is_best: true,
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["name"], "test-exp");
        assert_eq!(json["status"], "completed");
        assert_eq!(json["best_score"], 1.23);
        assert_eq!(json["trials"].as_array().unwrap().len(), 1);
        assert_eq!(json["trials"][0]["is_best"], true);
    }

    #[test]
    fn experiment_status_values() {
        let item_running = ExperimentListItem {
            id: uuid::Uuid::new_v4(),
            name: "r".into(),
            status: ExperimentStatus::Running,
            best_score: None,
        };
        let item_stopped = ExperimentListItem {
            id: uuid::Uuid::new_v4(),
            name: "s".into(),
            status: ExperimentStatus::Stopped,
            best_score: None,
        };
        let item_failed = ExperimentListItem {
            id: uuid::Uuid::new_v4(),
            name: "f".into(),
            status: ExperimentStatus::Failed,
            best_score: None,
        };
        assert_eq!(
            serde_json::to_value(&item_running).unwrap()["status"],
            "running"
        );
        assert_eq!(
            serde_json::to_value(&item_stopped).unwrap()["status"],
            "stopped"
        );
        assert_eq!(
            serde_json::to_value(&item_failed).unwrap()["status"],
            "failed"
        );
    }

    use crate::server::test_helpers::helpers::test_state;

    #[tokio::test]
    async fn list_experiments_store_unavailable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut state = test_state(&tmp);
        state.experiment_store = None;

        let query = ListExperimentsQuery {
            page: PaginationQuery {
                limit: super::super::pagination::DEFAULT_LIMIT,
                offset: 0,
            },
            status: None,
        };
        let result = list_experiments(
            State(state),
            Extension(TenantId::default_tenant()),
            Query(query),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn get_experiment_store_unavailable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut state = test_state(&tmp);
        state.experiment_store = None;

        let result = get_experiment(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(uuid::Uuid::new_v4()),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn get_experiment_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        // experiment_store might be Some — the experiment simply doesn't exist
        if state.experiment_store.is_some() {
            let result = get_experiment(
                State(state),
                Extension(TenantId::default_tenant()),
                Path(uuid::Uuid::new_v4()),
            )
            .await;
            assert!(result.is_err());
            assert_eq!(result.unwrap_err().0, StatusCode::NOT_FOUND);
        }
    }

    #[tokio::test]
    async fn create_experiment_store_unavailable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut state = test_state(&tmp);
        state.experiment_store = None;

        let json = r#"{
            "name": "test",
            "base_model": "m",
            "dataset_path": "/d.jsonl",
            "method": "lora",
            "time_budget_secs": 60,
            "objective": {"metric": "perplexity", "direction": "minimize"},
            "search": {"strategy": "grid"},
            "search_space": [{"name": "learning_rate", "values": [0.0001]}],
            "base_hyperparams": {
                "learning_rate": 0.0002,
                "epochs": 1,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
                "weight_decay": 0.0,
                "max_seq_length": 512
            }
        }"#;
        let req: CreateExperimentRequest = serde_json::from_str(json).unwrap();
        let result = create_experiment(
            State(state),
            Extension(TenantId::default_tenant()),
            Json(req),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn get_leaderboard_store_unavailable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut state = test_state(&tmp);
        state.experiment_store = None;

        let result = get_leaderboard(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(uuid::Uuid::new_v4()),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn stop_experiment_no_handle() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let id = uuid::Uuid::new_v4();
        // Should succeed even if experiment doesn't exist (idempotent)
        let result = stop_experiment(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(id),
        )
        .await;
        assert!(result.is_ok());
        let json = result.unwrap().0;
        assert_eq!(json["status"], "stopped");
    }

    #[tokio::test]
    async fn stop_experiment_returns_id_in_response() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let id = uuid::Uuid::new_v4();

        let result = stop_experiment(
            State(state),
            Extension(TenantId::default_tenant()),
            Path(id),
        )
        .await;
        assert!(result.is_ok());
        let json = result.unwrap().0;
        assert_eq!(json["id"], id.to_string());
        assert_eq!(json["status"], "stopped");
    }

    #[test]
    fn trial_to_response_completed() {
        use crate::types::experiment::{TrialResult, TrialStatus};
        use crate::types::training::HyperParams;

        let trial = TrialResult {
            trial_id: uuid::Uuid::new_v4(),
            experiment_id: uuid::Uuid::new_v4(),
            trial_number: 2,
            hyperparams: HyperParams {
                learning_rate: 5e-5,
                epochs: 3,
                batch_size: 4,
                gradient_accumulation_steps: 4,
                warmup_steps: 100,
                weight_decay: 0.01,
                max_seq_length: 2048,
            },
            train_loss: Some(0.35),
            eval_score: Some(4.12),
            status: TrialStatus::Completed,
            duration_secs: Some(180.5),
            started_at: Some(chrono::Utc::now()),
            completed_at: Some(chrono::Utc::now()),
            checkpoint_path: Some("/checkpoints/trial-2".into()),
            is_best: true,
        };

        let resp = trial_to_response(&trial);
        assert_eq!(resp.trial_number, 2);
        assert_eq!(resp.train_loss, Some(0.35));
        assert_eq!(resp.eval_score, Some(4.12));
        assert_eq!(resp.duration_secs, Some(180.5));
        assert!((resp.learning_rate - 5e-5).abs() < f64::EPSILON);
        assert!(resp.is_best);
        assert!(resp.status.contains("completed"));
    }

    #[test]
    fn trial_to_response_failed() {
        use crate::types::experiment::{TrialResult, TrialStatus};
        use crate::types::training::HyperParams;

        let trial = TrialResult {
            trial_id: uuid::Uuid::new_v4(),
            experiment_id: uuid::Uuid::new_v4(),
            trial_number: 5,
            hyperparams: HyperParams {
                learning_rate: 1e-3,
                epochs: 1,
                batch_size: 8,
                gradient_accumulation_steps: 1,
                warmup_steps: 0,
                weight_decay: 0.0,
                max_seq_length: 512,
            },
            train_loss: None,
            eval_score: None,
            status: TrialStatus::Failed,
            duration_secs: Some(10.0),
            started_at: Some(chrono::Utc::now()),
            completed_at: Some(chrono::Utc::now()),
            checkpoint_path: None,
            is_best: false,
        };

        let resp = trial_to_response(&trial);
        assert_eq!(resp.trial_number, 5);
        assert!(resp.train_loss.is_none());
        assert!(resp.eval_score.is_none());
        assert!(!resp.is_best);
        assert!(resp.status.contains("failed"));
    }

    #[test]
    fn trial_to_response_training() {
        use crate::types::experiment::{TrialResult, TrialStatus};
        use crate::types::training::HyperParams;

        let trial = TrialResult {
            trial_id: uuid::Uuid::new_v4(),
            experiment_id: uuid::Uuid::new_v4(),
            trial_number: 1,
            hyperparams: HyperParams {
                learning_rate: 2e-4,
                epochs: 3,
                batch_size: 4,
                gradient_accumulation_steps: 4,
                warmup_steps: 100,
                weight_decay: 0.01,
                max_seq_length: 2048,
            },
            train_loss: None,
            eval_score: None,
            status: TrialStatus::Training,
            duration_secs: None,
            started_at: Some(chrono::Utc::now()),
            completed_at: None,
            checkpoint_path: None,
            is_best: false,
        };

        let resp = trial_to_response(&trial);
        assert!(resp.status.contains("training"));
        assert!(resp.duration_secs.is_none());
    }

    #[test]
    fn experiment_response_empty_trials() {
        let resp = ExperimentResponse {
            id: uuid::Uuid::nil(),
            name: "empty-exp".into(),
            status: ExperimentStatus::Running,
            best_score: None,
            trials: vec![],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["name"], "empty-exp");
        assert_eq!(json["status"], "running");
        assert!(json["best_score"].is_null());
        assert!(json["trials"].as_array().unwrap().is_empty());
    }

    #[test]
    fn experiment_list_item_no_score() {
        let item = ExperimentListItem {
            id: uuid::Uuid::new_v4(),
            name: "no-score".into(),
            status: ExperimentStatus::Completed,
            best_score: None,
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["status"], "completed");
        assert!(json["best_score"].is_null());
    }

    #[tokio::test]
    async fn get_leaderboard_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        if state.experiment_store.is_some() {
            let result = get_leaderboard(
                State(state),
                Extension(TenantId::default_tenant()),
                Path(uuid::Uuid::new_v4()),
            )
            .await;
            assert!(result.is_err());
            assert_eq!(result.unwrap_err().0, StatusCode::NOT_FOUND);
        }
    }

    #[tokio::test]
    async fn list_experiments_with_store() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        if state.experiment_store.is_none() {
            return;
        }

        let query = ListExperimentsQuery {
            page: PaginationQuery {
                limit: super::super::pagination::DEFAULT_LIMIT,
                offset: 0,
            },
            status: None,
        };
        let result = list_experiments(
            State(state),
            Extension(TenantId::default_tenant()),
            Query(query),
        )
        .await;
        assert!(result.is_ok());
        // Empty store should return empty list
        assert!(result.unwrap().0.data.is_empty());
    }

    #[tokio::test]
    async fn list_experiments_with_status_filter() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        if state.experiment_store.is_none() {
            return;
        }

        let query = ListExperimentsQuery {
            page: PaginationQuery {
                limit: super::super::pagination::DEFAULT_LIMIT,
                offset: 0,
            },
            status: Some(ExperimentStatus::Running),
        };
        let result = list_experiments(
            State(state),
            Extension(TenantId::default_tenant()),
            Query(query),
        )
        .await;
        assert!(result.is_ok());
    }
}
