//! SQLite storage for evaluation results.

use ifran_types::IfranError;
use ifran_types::TenantId;
use ifran_types::error::Result;
use ifran_types::eval::{BenchmarkKind, EvalResult, EvalRunId};
use rusqlite::{Connection, params};
use uuid::Uuid;

/// Manages eval result storage in SQLite.
pub struct EvalStore {
    conn: Connection,
}

impl EvalStore {
    /// Open (or create) the eval database at the given path.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS eval_results (
                    run_id              TEXT NOT NULL,
                    model_name          TEXT NOT NULL,
                    benchmark           TEXT NOT NULL,
                    score               REAL NOT NULL,
                    details             TEXT,
                    samples_evaluated   INTEGER NOT NULL,
                    duration_secs       REAL NOT NULL,
                    evaluated_at        TEXT NOT NULL,
                    PRIMARY KEY (run_id, benchmark)
                );
                CREATE INDEX IF NOT EXISTS idx_eval_model ON eval_results(model_name);",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        // Add tenant_id column (idempotent — ignore if already exists)
        let _ = self.conn.execute_batch(
            "ALTER TABLE eval_results ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';",
        );
        self.conn
            .execute_batch("CREATE INDEX IF NOT EXISTS idx_eval_tenant ON eval_results(tenant_id);")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(())
    }

    /// Insert an eval result.
    pub fn insert(&self, result: &EvalResult, tenant_id: &TenantId) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO eval_results (run_id, model_name, benchmark, score, details,
                    samples_evaluated, duration_secs, evaluated_at, tenant_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    result.run_id.to_string(),
                    result.model_name,
                    serde_json::to_string(&result.benchmark)
                        .unwrap()
                        .trim_matches('"'),
                    result.score,
                    result.details.as_ref().map(|d| d.to_string()),
                    result.samples_evaluated as i64,
                    result.duration_secs,
                    result.evaluated_at.to_rfc3339(),
                    tenant_id.0,
                ],
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Get all results for a specific eval run.
    pub fn get_run(&self, run_id: EvalRunId, tenant_id: &TenantId) -> Result<Vec<EvalResult>> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM eval_results WHERE run_id = ?1 AND tenant_id = ?2")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(params![run_id.to_string(), tenant_id.0], row_to_eval_result)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(results)
    }

    /// Get all eval results for a model.
    pub fn get_by_model(&self, model_name: &str, tenant_id: &TenantId) -> Result<Vec<EvalResult>> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM eval_results WHERE model_name = ?1 AND tenant_id = ?2 ORDER BY evaluated_at DESC")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(params![model_name, tenant_id.0], row_to_eval_result)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(results)
    }

    /// List all eval results.
    pub fn list(&self, tenant_id: &TenantId) -> Result<Vec<EvalResult>> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM eval_results WHERE tenant_id = ?1 ORDER BY evaluated_at DESC")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(params![tenant_id.0], row_to_eval_result)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(results)
    }
}

fn row_to_eval_result(row: &rusqlite::Row) -> rusqlite::Result<EvalResult> {
    let run_id_str: String = row.get(0)?;
    let benchmark_str: String = row.get(2)?;
    let details_str: Option<String> = row.get(4)?;
    let evaluated_str: String = row.get(7)?;

    let run_id = Uuid::parse_str(&run_id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;

    let benchmark: BenchmarkKind =
        serde_json::from_str(&format!("\"{benchmark_str}\"")).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(2, rusqlite::types::Type::Text, Box::new(e))
        })?;

    let details = details_str.and_then(|s| serde_json::from_str(&s).ok());

    let evaluated_at = chrono::DateTime::parse_from_rfc3339(&evaluated_str)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(7, rusqlite::types::Type::Text, Box::new(e))
        })?;

    Ok(EvalResult {
        run_id,
        model_name: row.get(1)?,
        benchmark,
        score: row.get(3)?,
        details,
        samples_evaluated: row.get::<_, i64>(5)? as u64,
        duration_secs: row.get(6)?,
        evaluated_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn test_store() -> EvalStore {
        let conn = Connection::open_in_memory().unwrap();
        let store = EvalStore { conn };
        store.migrate().unwrap();
        store
    }

    fn sample_result() -> EvalResult {
        EvalResult {
            run_id: Uuid::new_v4(),
            model_name: "llama-3.1-8b".into(),
            benchmark: BenchmarkKind::Perplexity,
            score: 5.23,
            details: None,
            samples_evaluated: 1000,
            duration_secs: 42.5,
            evaluated_at: Utc::now(),
        }
    }

    #[test]
    fn insert_and_get_by_model() {
        let store = test_store();
        let tenant = TenantId::default_tenant();
        let result = sample_result();
        store.insert(&result, &tenant).unwrap();
        let fetched = store.get_by_model("llama-3.1-8b", &tenant).unwrap();
        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].score, 5.23);
    }

    #[test]
    fn get_run() {
        let store = test_store();
        let tenant = TenantId::default_tenant();
        let result = sample_result();
        let run_id = result.run_id;
        store.insert(&result, &tenant).unwrap();
        let fetched = store.get_run(run_id, &tenant).unwrap();
        assert_eq!(fetched.len(), 1);
    }

    #[test]
    fn list_all() {
        let store = test_store();
        let tenant = TenantId::default_tenant();
        store.insert(&sample_result(), &tenant).unwrap();
        store
            .insert(
                &EvalResult {
                    benchmark: BenchmarkKind::Custom,
                    ..sample_result()
                },
                &tenant,
            )
            .unwrap();
        let all = store.list(&tenant).unwrap();
        assert_eq!(all.len(), 2);
    }
}
