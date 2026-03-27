//! SQLite-backed persistence for training jobs (crash recovery).

use crate::job::status::JobState;
use chrono::{DateTime, Utc};
use ifran_types::IfranError;
use ifran_types::PagedResult;
use ifran_types::TenantId;
use ifran_types::error::Result;
use ifran_types::training::{TrainingJobConfig, TrainingJobId, TrainingStatus};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use std::path::Path;
use uuid::Uuid;

/// Persists training job state to SQLite for crash recovery.
pub struct JobStore {
    pool: Pool<SqliteConnectionManager>,
}

impl JobStore {
    /// Open (or create) the job store database at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let manager = SqliteConnectionManager::file(path).with_init(|conn| {
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
            Ok(())
        });
        let pool = Pool::builder()
            .max_size(4)
            .build(manager)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let store = Self { pool };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS training_jobs (
                    id              TEXT PRIMARY KEY,
                    tenant_id       TEXT NOT NULL DEFAULT 'default',
                    config_json     TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    current_step    INTEGER NOT NULL DEFAULT 0,
                    total_steps     INTEGER NOT NULL DEFAULT 0,
                    current_epoch   REAL NOT NULL DEFAULT 0.0,
                    current_loss    REAL,
                    created_at      TEXT NOT NULL,
                    started_at      TEXT,
                    completed_at    TEXT,
                    error           TEXT
                );",
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Upsert a job state (INSERT OR REPLACE).
    pub fn save_job(&self, job: &JobState) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let config_json = serde_json::to_string(&job.config)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let status_str = serde_json::to_string(&job.status)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        // serde_json wraps in quotes for string enums, trim them
        let status_str = status_str.trim_matches('"');

        conn.execute(
            "INSERT OR REPLACE INTO training_jobs
                    (id, tenant_id, config_json, status, current_step, total_steps, current_epoch,
                     current_loss, created_at, started_at, completed_at, error)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                job.id.to_string(),
                job.tenant_id.0.as_str(),
                config_json,
                status_str,
                job.current_step as i64,
                job.total_steps as i64,
                job.current_epoch as f64,
                job.current_loss,
                job.created_at.to_rfc3339(),
                job.started_at.map(|t| t.to_rfc3339()),
                job.completed_at.map(|t| t.to_rfc3339()),
                job.error,
            ],
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Look up a single job by ID.
    pub fn get_job(&self, id: TrainingJobId) -> Result<Option<JobState>> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT * FROM training_jobs WHERE id = ?1")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![id.to_string()], row_to_job_state)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        match rows.next() {
            Some(row) => Ok(Some(
                row.map_err(|e| IfranError::StorageError(e.to_string()))?,
            )),
            None => Ok(None),
        }
    }

    /// List jobs, optionally filtered by status, with pagination.
    pub fn list_jobs(
        &self,
        status_filter: Option<TrainingStatus>,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<JobState>> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let (total, items) = match status_filter {
            Some(status) => {
                let status_str = serde_json::to_string(&status)
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;
                let status_str = status_str.trim_matches('"').to_string();

                let total: usize = conn
                    .query_row(
                        "SELECT COUNT(*) FROM training_jobs WHERE status = ?1",
                        params![status_str],
                        |row| row.get::<_, i64>(0),
                    )
                    .map(|c| c as usize)
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;

                let mut stmt = conn
                    .prepare(
                        "SELECT * FROM training_jobs WHERE status = ?1 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3",
                    )
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;
                let items = stmt
                    .query_map(params![status_str, limit, offset], row_to_job_state)
                    .map_err(|e| IfranError::StorageError(e.to_string()))?
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;

                (total, items)
            }
            None => {
                let total: usize = conn
                    .query_row("SELECT COUNT(*) FROM training_jobs", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .map(|c| c as usize)
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;

                let mut stmt = conn
                    .prepare(
                        "SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT ?1 OFFSET ?2",
                    )
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;
                let items = stmt
                    .query_map(params![limit, offset], row_to_job_state)
                    .map_err(|e| IfranError::StorageError(e.to_string()))?
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;

                (total, items)
            }
        };

        Ok(PagedResult { items, total })
    }

    /// Delete a job from the store.
    pub fn delete_job(&self, id: TrainingJobId) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.execute(
            "DELETE FROM training_jobs WHERE id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Load all non-terminal jobs for crash recovery.
    /// Terminal statuses: completed, failed, cancelled.
    pub fn recover_jobs(&self) -> Result<Vec<JobState>> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT * FROM training_jobs WHERE status NOT IN ('completed', 'failed', 'cancelled')
                 ORDER BY created_at ASC",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map([], row_to_job_state)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(results)
    }
}

fn row_to_job_state(row: &rusqlite::Row) -> rusqlite::Result<JobState> {
    let id_str: String = row.get(0)?;
    let tenant_id_str: String = row.get(1)?;
    let config_json: String = row.get(2)?;
    let status_str: String = row.get(3)?;
    let current_step: i64 = row.get(4)?;
    let total_steps: i64 = row.get(5)?;
    let current_epoch: f64 = row.get(6)?;
    let current_loss: Option<f64> = row.get(7)?;
    let created_at_str: String = row.get(8)?;
    let started_at_str: Option<String> = row.get(9)?;
    let completed_at_str: Option<String> = row.get(10)?;
    let error: Option<String> = row.get(11)?;

    let id = Uuid::parse_str(&id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;

    let config: TrainingJobConfig = serde_json::from_str(&config_json).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(2, rusqlite::types::Type::Text, Box::new(e))
    })?;

    let status: TrainingStatus =
        serde_json::from_str(&format!("\"{status_str}\"")).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e))
        })?;

    let created_at = parse_rfc3339(&created_at_str, 8)?;
    let started_at = started_at_str
        .as_deref()
        .map(|s| parse_rfc3339(s, 9))
        .transpose()?;
    let completed_at = completed_at_str
        .as_deref()
        .map(|s| parse_rfc3339(s, 10))
        .transpose()?;

    Ok(JobState {
        id,
        tenant_id: TenantId(tenant_id_str),
        config,
        status,
        current_step: current_step as u64,
        total_steps: total_steps as u64,
        current_epoch: current_epoch as f32,
        current_loss,
        checkpoints: Vec::new(), // Checkpoints are not persisted in this table
        created_at,
        started_at,
        completed_at,
        error,
    })
}

fn parse_rfc3339(s: &str, col: usize) -> rusqlite::Result<DateTime<Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(col, rusqlite::types::Type::Text, Box::new(e))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ifran_types::training::*;

    fn test_store() -> JobStore {
        let manager = SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
            Ok(())
        });
        let pool = Pool::builder().max_size(4).build(manager).unwrap();
        let store = JobStore { pool };
        store.migrate().unwrap();
        store
    }

    fn test_config() -> TrainingJobConfig {
        TrainingJobConfig {
            base_model: "test-model".into(),
            dataset: DatasetConfig {
                path: "/tmp/data.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: Some(100),
            },
            method: TrainingMethod::Lora,
            hyperparams: HyperParams {
                learning_rate: 2e-4,
                epochs: 1,
                batch_size: 4,
                gradient_accumulation_steps: 1,
                warmup_steps: 0,
                weight_decay: 0.0,
                max_seq_length: 512,
            },
            output_name: None,
            lora: None,
            max_steps: None,
            time_budget_secs: None,
        }
    }

    fn test_job_state() -> JobState {
        JobState::new(
            Uuid::new_v4(),
            TenantId::default_tenant(),
            test_config(),
            100,
        )
    }

    #[test]
    fn save_and_load_roundtrip() {
        let store = test_store();
        let job = test_job_state();
        let id = job.id;

        store.save_job(&job).unwrap();
        let loaded = store.get_job(id).unwrap().expect("job should exist");

        assert_eq!(loaded.id, id);
        assert_eq!(loaded.status, TrainingStatus::Queued);
        assert_eq!(loaded.config.base_model, "test-model");
        assert_eq!(loaded.total_steps, 100);
        assert_eq!(loaded.current_step, 0);
    }

    #[test]
    fn save_updates_existing() {
        let store = test_store();
        let mut job = test_job_state();
        let id = job.id;

        store.save_job(&job).unwrap();
        job.start();
        job.update_progress(50, 0.5, 0.42);
        store.save_job(&job).unwrap();

        let loaded = store.get_job(id).unwrap().expect("job should exist");
        assert_eq!(loaded.status, TrainingStatus::Running);
        assert_eq!(loaded.current_step, 50);
        assert_eq!(loaded.current_loss, Some(0.42));
    }

    #[test]
    fn get_job_not_found() {
        let store = test_store();
        let result = store.get_job(Uuid::new_v4()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn list_jobs_all() {
        let store = test_store();
        store.save_job(&test_job_state()).unwrap();
        store.save_job(&test_job_state()).unwrap();
        let all = store.list_jobs(None, 100, 0).unwrap();
        assert_eq!(all.items.len(), 2);
        assert_eq!(all.total, 2);
    }

    #[test]
    fn list_jobs_filtered() {
        let store = test_store();
        let mut job1 = test_job_state();
        let job2 = test_job_state();
        job1.start();

        store.save_job(&job1).unwrap();
        store.save_job(&job2).unwrap();

        let running = store
            .list_jobs(Some(TrainingStatus::Running), 100, 0)
            .unwrap();
        assert_eq!(running.items.len(), 1);
        assert_eq!(running.total, 1);

        let queued = store
            .list_jobs(Some(TrainingStatus::Queued), 100, 0)
            .unwrap();
        assert_eq!(queued.items.len(), 1);
        assert_eq!(queued.total, 1);

        let failed = store
            .list_jobs(Some(TrainingStatus::Failed), 100, 0)
            .unwrap();
        assert!(failed.items.is_empty());
        assert_eq!(failed.total, 0);
    }

    #[test]
    fn delete_job() {
        let store = test_store();
        let job = test_job_state();
        let id = job.id;

        store.save_job(&job).unwrap();
        assert!(store.get_job(id).unwrap().is_some());

        store.delete_job(id).unwrap();
        assert!(store.get_job(id).unwrap().is_none());
    }

    #[test]
    fn recover_jobs_returns_non_terminal() {
        let store = test_store();

        // Queued job — should be recovered
        let queued = test_job_state();
        store.save_job(&queued).unwrap();

        // Running job — should be recovered
        let mut running = test_job_state();
        running.start();
        store.save_job(&running).unwrap();

        // Completed job — should NOT be recovered
        let mut completed = test_job_state();
        completed.start();
        completed.complete();
        store.save_job(&completed).unwrap();

        // Failed job — should NOT be recovered
        let mut failed = test_job_state();
        failed.start();
        failed.fail("error".into());
        store.save_job(&failed).unwrap();

        // Cancelled job — should NOT be recovered
        let mut cancelled = test_job_state();
        cancelled.cancel();
        store.save_job(&cancelled).unwrap();

        let recovered = store.recover_jobs().unwrap();
        assert_eq!(recovered.len(), 2);
    }

    #[test]
    fn recovery_marks_running_as_failed() {
        let store = test_store();
        let mut job = test_job_state();
        job.start();
        store.save_job(&job).unwrap();

        // Simulate crash recovery: load non-terminal, mark running/preparing as failed
        let mut recovered = store.recover_jobs().unwrap();
        for j in &mut recovered {
            if j.status == TrainingStatus::Running || j.status == TrainingStatus::Preparing {
                j.fail("Process crashed — job interrupted".into());
                store.save_job(j).unwrap();
            }
        }

        let loaded = store.get_job(job.id).unwrap().unwrap();
        assert_eq!(loaded.status, TrainingStatus::Failed);
        assert_eq!(
            loaded.error.as_deref(),
            Some("Process crashed — job interrupted")
        );
    }

    #[test]
    fn open_creates_file_and_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("subdir").join("jobs.db");
        let store = JobStore::open(&db_path).unwrap();
        assert!(db_path.exists());
        // Should be able to save/load
        let job = test_job_state();
        store.save_job(&job).unwrap();
        let loaded = store.get_job(job.id).unwrap();
        assert!(loaded.is_some());
    }
}
