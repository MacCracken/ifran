//! Z-score-based inference quality drift detection.

use rusqlite::{Connection, params};
use synapse_types::SynapseError;
use synapse_types::TenantId;
use synapse_types::drift::{BaselineSnapshot, DriftResult, DriftSeverity};
use synapse_types::error::Result;
use uuid::Uuid;

pub struct DriftDetector {
    conn: Connection,
    /// Z-score threshold for drift alert (default 2.0).
    threshold: f64,
}

impl DriftDetector {
    pub fn open(path: &std::path::Path, threshold: f64) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let det = Self { conn, threshold };
        det.migrate()?;
        Ok(det)
    }

    #[cfg(test)]
    pub fn open_in_memory(threshold: f64) -> Result<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let det = Self { conn, threshold };
        det.migrate()?;
        Ok(det)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS drift_baselines (
                id           TEXT PRIMARY KEY,
                tenant_id    TEXT NOT NULL DEFAULT 'default',
                model_name   TEXT NOT NULL,
                mean_score   REAL NOT NULL,
                std_dev      REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                created_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_drift_model ON drift_baselines(tenant_id, model_name);",
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Record a baseline snapshot from evaluation scores.
    pub fn record_baseline(
        &self,
        model_name: &str,
        scores: &[f64],
        tenant_id: &TenantId,
    ) -> Result<BaselineSnapshot> {
        if scores.is_empty() {
            return Err(SynapseError::EvalError(
                "Cannot create baseline from empty scores".into(),
            ));
        }

        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let snapshot = BaselineSnapshot {
            id: Uuid::new_v4(),
            model_name: model_name.into(),
            mean_score: mean,
            std_dev,
            sample_count: scores.len() as u64,
            created_at: chrono::Utc::now(),
        };

        self.conn
            .execute(
                "INSERT INTO drift_baselines (id, tenant_id, model_name, mean_score, std_dev, sample_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    snapshot.id.to_string(),
                    tenant_id.0,
                    snapshot.model_name,
                    snapshot.mean_score,
                    snapshot.std_dev,
                    snapshot.sample_count as i64,
                    snapshot.created_at.to_rfc3339()
                ],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        Ok(snapshot)
    }

    /// Check for drift against the latest baseline for a model.
    pub fn check_drift(
        &self,
        model_name: &str,
        current_scores: &[f64],
        tenant_id: &TenantId,
    ) -> Result<DriftResult> {
        if current_scores.is_empty() {
            return Err(SynapseError::EvalError(
                "Cannot check drift with empty scores".into(),
            ));
        }

        let baseline = self.get_latest_baseline(model_name, tenant_id)?;
        let current_mean = current_scores.iter().sum::<f64>() / current_scores.len() as f64;

        let z_score = if baseline.std_dev > f64::EPSILON {
            (current_mean - baseline.mean_score) / baseline.std_dev
        } else {
            0.0
        };

        let severity = DriftSeverity::from_z_score(z_score);

        Ok(DriftResult {
            snapshot_id: baseline.id,
            current_mean,
            z_score,
            drifted: z_score.abs() > self.threshold,
            severity,
            checked_at: chrono::Utc::now(),
        })
    }

    /// Get the latest baseline for a model.
    pub fn get_latest_baseline(
        &self,
        model_name: &str,
        tenant_id: &TenantId,
    ) -> Result<BaselineSnapshot> {
        self.conn
            .query_row(
                "SELECT id, model_name, mean_score, std_dev, sample_count, created_at
             FROM drift_baselines WHERE model_name = ?1 AND tenant_id = ?2
             ORDER BY created_at DESC LIMIT 1",
                params![model_name, tenant_id.0],
                |row| {
                    let id_str: String = row.get(0)?;
                    let created_str: String = row.get(5)?;
                    Ok(BaselineSnapshot {
                        id: Uuid::parse_str(&id_str).unwrap_or_default(),
                        model_name: row.get(1)?,
                        mean_score: row.get(2)?,
                        std_dev: row.get(3)?,
                        sample_count: row.get::<_, i64>(4)? as u64,
                        created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now()),
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    SynapseError::EvalError(format!("No baseline for model {model_name}"))
                }
                other => SynapseError::StorageError(other.to_string()),
            })
    }

    /// List all baselines for a model.
    pub fn list_baselines(
        &self,
        model_name: &str,
        tenant_id: &TenantId,
    ) -> Result<Vec<BaselineSnapshot>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, model_name, mean_score, std_dev, sample_count, created_at
             FROM drift_baselines WHERE model_name = ?1 AND tenant_id = ?2
             ORDER BY created_at DESC",
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(params![model_name, tenant_id.0], |row| {
                let id_str: String = row.get(0)?;
                let created_str: String = row.get(5)?;
                Ok(BaselineSnapshot {
                    id: Uuid::parse_str(&id_str).unwrap_or_default(),
                    model_name: row.get(1)?,
                    mean_score: row.get(2)?,
                    std_dev: row.get(3)?,
                    sample_count: row.get::<_, i64>(4)? as u64,
                    created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                })
            })
            .map_err(|e| SynapseError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> TenantId {
        TenantId::default_tenant()
    }

    #[test]
    fn record_and_get_baseline() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        let scores = vec![0.80, 0.85, 0.82, 0.88, 0.84];
        let baseline = det.record_baseline("llama-8b", &scores, &t()).unwrap();
        assert_eq!(baseline.model_name, "llama-8b");
        assert_eq!(baseline.sample_count, 5);
        assert!((baseline.mean_score - 0.838).abs() < 0.01);
    }

    #[test]
    fn check_no_drift() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        det.record_baseline("model", &vec![0.80, 0.85, 0.82, 0.88, 0.84], &t())
            .unwrap();
        let result = det
            .check_drift("model", &vec![0.83, 0.84, 0.82], &t())
            .unwrap();
        assert!(!result.drifted);
        assert_eq!(result.severity, DriftSeverity::None);
    }

    #[test]
    fn check_drift_detected() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        det.record_baseline("model", &vec![0.80, 0.85, 0.82, 0.88, 0.84], &t())
            .unwrap();
        // Much lower scores -> drift
        let result = det
            .check_drift("model", &vec![0.50, 0.52, 0.48], &t())
            .unwrap();
        assert!(result.drifted);
        assert!(result.z_score.abs() > 2.0);
    }

    #[test]
    fn empty_scores_error() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        assert!(det.record_baseline("m", &[], &t()).is_err());
    }

    #[test]
    fn no_baseline_error() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        assert!(det.check_drift("nonexistent", &[0.5], &t()).is_err());
    }

    #[test]
    fn list_baselines() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        det.record_baseline("m", &[0.8, 0.9], &t()).unwrap();
        det.record_baseline("m", &[0.7, 0.8], &t()).unwrap();
        let baselines = det.list_baselines("m", &t()).unwrap();
        assert_eq!(baselines.len(), 2);
    }

    #[test]
    fn zero_std_dev_no_drift() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        det.record_baseline("m", &[0.8, 0.8, 0.8], &t()).unwrap();
        let result = det.check_drift("m", &[0.5], &t()).unwrap();
        // Zero std_dev -> z_score = 0 -> no drift
        assert!(!result.drifted);
    }
}
