//! SQLite storage for experiment and trial results.

use rusqlite::{Connection, params};
use synapse_types::SynapseError;
use synapse_types::error::Result;
use synapse_types::experiment::{
    Direction, ExperimentId, ExperimentProgram, ExperimentStatus, TrialId, TrialResult, TrialStatus,
};
use uuid::Uuid;

/// Full experiment record returned by `get_experiment`.
pub type ExperimentRecord = (ExperimentId, String, ExperimentProgram, ExperimentStatus, Option<TrialId>, Option<f64>);

/// Summary record returned by `list_experiments`.
pub type ExperimentSummary = (ExperimentId, String, ExperimentStatus, Option<f64>);

/// Manages experiment and trial storage in SQLite.
pub struct ExperimentStore {
    conn: Connection,
}

impl ExperimentStore {
    /// Open (or create) the experiment database at the given path.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn =
            Connection::open(path).map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Create an in-memory store (for testing).
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS experiments (
                    id              TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    program_json    TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'running',
                    best_trial_id   TEXT,
                    best_score      REAL,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiment_trials (
                    trial_id        TEXT PRIMARY KEY,
                    experiment_id   TEXT NOT NULL,
                    trial_number    INTEGER NOT NULL,
                    hyperparams_json TEXT NOT NULL,
                    train_loss      REAL,
                    eval_score      REAL,
                    status          TEXT NOT NULL DEFAULT 'training',
                    duration_secs   REAL,
                    started_at      TEXT,
                    completed_at    TEXT,
                    checkpoint_path TEXT,
                    is_best         INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );
                CREATE INDEX IF NOT EXISTS idx_trials_experiment
                    ON experiment_trials(experiment_id);",
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Insert a new experiment.
    pub fn insert_experiment(
        &self,
        id: ExperimentId,
        name: &str,
        program: &ExperimentProgram,
    ) -> Result<()> {
        let program_json = serde_json::to_string(program)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let now = chrono::Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT INTO experiments (id, name, program_json, status, created_at, updated_at)
                 VALUES (?1, ?2, ?3, 'running', ?4, ?5)",
                params![id.to_string(), name, program_json, now, now],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Update experiment status.
    pub fn update_experiment_status(
        &self,
        id: ExperimentId,
        status: ExperimentStatus,
    ) -> Result<()> {
        let status_str = serde_json::to_string(&status)
            .unwrap()
            .trim_matches('"')
            .to_string();
        let now = chrono::Utc::now().to_rfc3339();
        self.conn
            .execute(
                "UPDATE experiments SET status = ?1, updated_at = ?2 WHERE id = ?3",
                params![status_str, now, id.to_string()],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Update the best trial for an experiment.
    pub fn update_best_trial(
        &self,
        experiment_id: ExperimentId,
        trial_id: TrialId,
        score: f64,
    ) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn
            .execute(
                "UPDATE experiments SET best_trial_id = ?1, best_score = ?2, updated_at = ?3 WHERE id = ?4",
                params![trial_id.to_string(), score, now, experiment_id.to_string()],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        // Mark the trial as best, unmark previous best
        self.conn
            .execute(
                "UPDATE experiment_trials SET is_best = 0 WHERE experiment_id = ?1 AND trial_id != ?2",
                params![experiment_id.to_string(), trial_id.to_string()],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        self.conn
            .execute(
                "UPDATE experiment_trials SET is_best = 1 WHERE trial_id = ?1",
                params![trial_id.to_string()],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Insert a new trial.
    pub fn insert_trial(&self, trial: &TrialResult) -> Result<()> {
        let hp_json = serde_json::to_string(&trial.hyperparams)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let status_str = serde_json::to_string(&trial.status)
            .unwrap()
            .trim_matches('"')
            .to_string();
        self.conn
            .execute(
                "INSERT INTO experiment_trials
                    (trial_id, experiment_id, trial_number, hyperparams_json, train_loss,
                     eval_score, status, duration_secs, started_at, completed_at,
                     checkpoint_path, is_best)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    trial.trial_id.to_string(),
                    trial.experiment_id.to_string(),
                    trial.trial_number,
                    hp_json,
                    trial.train_loss,
                    trial.eval_score,
                    status_str,
                    trial.duration_secs,
                    trial.started_at.map(|t| t.to_rfc3339()),
                    trial.completed_at.map(|t| t.to_rfc3339()),
                    trial.checkpoint_path,
                    trial.is_best as i32,
                ],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Update an existing trial.
    pub fn update_trial(&self, trial: &TrialResult) -> Result<()> {
        let status_str = serde_json::to_string(&trial.status)
            .unwrap()
            .trim_matches('"')
            .to_string();
        self.conn
            .execute(
                "UPDATE experiment_trials SET
                    train_loss = ?1, eval_score = ?2, status = ?3,
                    duration_secs = ?4, completed_at = ?5, checkpoint_path = ?6, is_best = ?7
                 WHERE trial_id = ?8",
                params![
                    trial.train_loss,
                    trial.eval_score,
                    status_str,
                    trial.duration_secs,
                    trial.completed_at.map(|t| t.to_rfc3339()),
                    trial.checkpoint_path,
                    trial.is_best as i32,
                    trial.trial_id.to_string(),
                ],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Get an experiment by ID, returns (id, name, program, status, best_trial_id, best_score).
    pub fn get_experiment(
        &self,
        id: ExperimentId,
    ) -> Result<ExperimentRecord>
    {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, program_json, status, best_trial_id, best_score FROM experiments WHERE id = ?1")
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        stmt.query_row(params![id.to_string()], |row| {
            let id_str: String = row.get(0)?;
            let name: String = row.get(1)?;
            let program_json: String = row.get(2)?;
            let status_str: String = row.get(3)?;
            let best_trial_str: Option<String> = row.get(4)?;
            let best_score: Option<f64> = row.get(5)?;

            let id = Uuid::parse_str(&id_str).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
            })?;
            let program: ExperimentProgram =
                serde_json::from_str(&program_json).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        2,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
            let status: ExperimentStatus =
                serde_json::from_str(&format!("\"{status_str}\"")).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
            let best_trial_id = best_trial_str.and_then(|s| Uuid::parse_str(&s).ok());

            Ok((id, name, program, status, best_trial_id, best_score))
        })
        .map_err(|e| SynapseError::StorageError(e.to_string()))
    }

    /// List all experiments.
    pub fn list_experiments(
        &self,
    ) -> Result<Vec<ExperimentSummary>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, status, best_score FROM experiments ORDER BY created_at DESC")
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map([], |row| {
                let id_str: String = row.get(0)?;
                let name: String = row.get(1)?;
                let status_str: String = row.get(2)?;
                let best_score: Option<f64> = row.get(3)?;

                let id = Uuid::parse_str(&id_str).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
                let status: ExperimentStatus =
                    serde_json::from_str(&format!("\"{status_str}\"")).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            2,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?;

                Ok((id, name, status, best_score))
            })
            .map_err(|e| SynapseError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        Ok(results)
    }

    /// Get all trials for an experiment.
    pub fn get_trials(&self, experiment_id: ExperimentId) -> Result<Vec<TrialResult>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT trial_id, experiment_id, trial_number, hyperparams_json, train_loss,
                        eval_score, status, duration_secs, started_at, completed_at,
                        checkpoint_path, is_best
                 FROM experiment_trials WHERE experiment_id = ?1
                 ORDER BY trial_number ASC",
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(params![experiment_id.to_string()], row_to_trial)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        Ok(results)
    }

    /// Get trials sorted by score (leaderboard).
    pub fn get_leaderboard(
        &self,
        experiment_id: ExperimentId,
        direction: Direction,
        limit: usize,
    ) -> Result<Vec<TrialResult>> {
        let order = match direction {
            Direction::Minimize => "ASC",
            Direction::Maximize => "DESC",
        };
        let sql = format!(
            "SELECT trial_id, experiment_id, trial_number, hyperparams_json, train_loss,
                    eval_score, status, duration_secs, started_at, completed_at,
                    checkpoint_path, is_best
             FROM experiment_trials
             WHERE experiment_id = ?1 AND eval_score IS NOT NULL AND status = 'completed'
             ORDER BY eval_score {order}
             LIMIT ?2"
        );

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(
                params![experiment_id.to_string(), limit as i64],
                row_to_trial,
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        Ok(results)
    }
}

fn row_to_trial(row: &rusqlite::Row) -> rusqlite::Result<TrialResult> {
    let trial_id_str: String = row.get(0)?;
    let experiment_id_str: String = row.get(1)?;
    let hp_json: String = row.get(3)?;
    let status_str: String = row.get(6)?;
    let started_str: Option<String> = row.get(8)?;
    let completed_str: Option<String> = row.get(9)?;

    let trial_id = Uuid::parse_str(&trial_id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let experiment_id = Uuid::parse_str(&experiment_id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let hyperparams = serde_json::from_str(&hp_json).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let status: TrialStatus =
        serde_json::from_str(&format!("\"{status_str}\"")).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(6, rusqlite::types::Type::Text, Box::new(e))
        })?;
    let started_at = started_str.and_then(|s| {
        chrono::DateTime::parse_from_rfc3339(&s)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .ok()
    });
    let completed_at = completed_str.and_then(|s| {
        chrono::DateTime::parse_from_rfc3339(&s)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .ok()
    });

    Ok(TrialResult {
        trial_id,
        experiment_id,
        trial_number: row.get(2)?,
        hyperparams,
        train_loss: row.get(4)?,
        eval_score: row.get(5)?,
        status,
        duration_secs: row.get(7)?,
        started_at,
        completed_at,
        checkpoint_path: row.get(10)?,
        is_best: row.get::<_, i32>(11)? != 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_types::eval::BenchmarkKind;
    use synapse_types::experiment::*;
    use synapse_types::training::*;

    fn sample_program() -> ExperimentProgram {
        ExperimentProgram {
            name: "test-experiment".into(),
            base_model: "llama-8b".into(),
            dataset_path: "/data/train.jsonl".into(),
            dataset_format: "jsonl".into(),
            method: TrainingMethod::Lora,
            time_budget_secs: 300,
            objective: ExperimentObjective {
                metric: BenchmarkKind::Perplexity,
                direction: Direction::Minimize,
            },
            search: SearchStrategy::Grid,
            search_space: vec![],
            base_hyperparams: HyperParams {
                learning_rate: 2e-4,
                epochs: 3,
                batch_size: 4,
                gradient_accumulation_steps: 4,
                warmup_steps: 100,
                weight_decay: 0.01,
                max_seq_length: 2048,
            },
            max_trials: None,
            eval_sample_limit: None,
        }
    }

    fn sample_trial(experiment_id: ExperimentId, num: u32) -> TrialResult {
        TrialResult {
            trial_id: Uuid::new_v4(),
            experiment_id,
            trial_number: num,
            hyperparams: HyperParams {
                learning_rate: 1e-4,
                epochs: 3,
                batch_size: 4,
                gradient_accumulation_steps: 4,
                warmup_steps: 100,
                weight_decay: 0.01,
                max_seq_length: 2048,
            },
            train_loss: Some(0.42),
            eval_score: Some(5.23),
            status: TrialStatus::Completed,
            duration_secs: Some(295.0),
            started_at: Some(chrono::Utc::now()),
            completed_at: Some(chrono::Utc::now()),
            checkpoint_path: Some("/checkpoints/trial".into()),
            is_best: false,
        }
    }

    #[test]
    fn insert_and_get_experiment() {
        let store = ExperimentStore::open_in_memory().unwrap();
        let id = Uuid::new_v4();
        let program = sample_program();
        store.insert_experiment(id, "test", &program).unwrap();

        let (got_id, name, got_program, status, best_trial, best_score) =
            store.get_experiment(id).unwrap();
        assert_eq!(got_id, id);
        assert_eq!(name, "test");
        assert_eq!(got_program.base_model, "llama-8b");
        assert_eq!(status, ExperimentStatus::Running);
        assert!(best_trial.is_none());
        assert!(best_score.is_none());
    }

    #[test]
    fn update_experiment_status() {
        let store = ExperimentStore::open_in_memory().unwrap();
        let id = Uuid::new_v4();
        store
            .insert_experiment(id, "test", &sample_program())
            .unwrap();

        store
            .update_experiment_status(id, ExperimentStatus::Completed)
            .unwrap();

        let (_, _, _, status, _, _) = store.get_experiment(id).unwrap();
        assert_eq!(status, ExperimentStatus::Completed);
    }

    #[test]
    fn insert_and_get_trials() {
        let store = ExperimentStore::open_in_memory().unwrap();
        let exp_id = Uuid::new_v4();
        store
            .insert_experiment(exp_id, "test", &sample_program())
            .unwrap();

        let trial1 = sample_trial(exp_id, 1);
        let trial2 = sample_trial(exp_id, 2);
        store.insert_trial(&trial1).unwrap();
        store.insert_trial(&trial2).unwrap();

        let trials = store.get_trials(exp_id).unwrap();
        assert_eq!(trials.len(), 2);
        assert_eq!(trials[0].trial_number, 1);
        assert_eq!(trials[1].trial_number, 2);
    }

    #[test]
    fn update_best_trial() {
        let store = ExperimentStore::open_in_memory().unwrap();
        let exp_id = Uuid::new_v4();
        store
            .insert_experiment(exp_id, "test", &sample_program())
            .unwrap();

        let trial = sample_trial(exp_id, 1);
        store.insert_trial(&trial).unwrap();
        store
            .update_best_trial(exp_id, trial.trial_id, 5.23)
            .unwrap();

        let (_, _, _, _, best_id, best_score) = store.get_experiment(exp_id).unwrap();
        assert_eq!(best_id, Some(trial.trial_id));
        assert_eq!(best_score, Some(5.23));

        let trials = store.get_trials(exp_id).unwrap();
        assert!(trials[0].is_best);
    }

    #[test]
    fn list_experiments() {
        let store = ExperimentStore::open_in_memory().unwrap();
        store
            .insert_experiment(Uuid::new_v4(), "exp-1", &sample_program())
            .unwrap();
        store
            .insert_experiment(Uuid::new_v4(), "exp-2", &sample_program())
            .unwrap();

        let list = store.list_experiments().unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn leaderboard_ordering() {
        let store = ExperimentStore::open_in_memory().unwrap();
        let exp_id = Uuid::new_v4();
        store
            .insert_experiment(exp_id, "test", &sample_program())
            .unwrap();

        let mut t1 = sample_trial(exp_id, 1);
        t1.eval_score = Some(10.0);
        let mut t2 = sample_trial(exp_id, 2);
        t2.eval_score = Some(5.0);
        let mut t3 = sample_trial(exp_id, 3);
        t3.eval_score = Some(8.0);
        store.insert_trial(&t1).unwrap();
        store.insert_trial(&t2).unwrap();
        store.insert_trial(&t3).unwrap();

        // Minimize → lowest first
        let lb = store
            .get_leaderboard(exp_id, Direction::Minimize, 10)
            .unwrap();
        assert_eq!(lb.len(), 3);
        assert_eq!(lb[0].eval_score, Some(5.0));
        assert_eq!(lb[1].eval_score, Some(8.0));

        // Maximize → highest first
        let lb = store
            .get_leaderboard(exp_id, Direction::Maximize, 2)
            .unwrap();
        assert_eq!(lb.len(), 2);
        assert_eq!(lb[0].eval_score, Some(10.0));
    }

    #[test]
    fn update_trial() {
        let store = ExperimentStore::open_in_memory().unwrap();
        let exp_id = Uuid::new_v4();
        store
            .insert_experiment(exp_id, "test", &sample_program())
            .unwrap();

        let mut trial = sample_trial(exp_id, 1);
        trial.status = TrialStatus::Training;
        trial.eval_score = None;
        store.insert_trial(&trial).unwrap();

        trial.status = TrialStatus::Completed;
        trial.eval_score = Some(3.14);
        store.update_trial(&trial).unwrap();

        let trials = store.get_trials(exp_id).unwrap();
        assert_eq!(trials[0].status, TrialStatus::Completed);
        assert_eq!(trials[0].eval_score, Some(3.14));
    }
}
