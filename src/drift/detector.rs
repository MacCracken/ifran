//! Z-score-based inference quality drift detection.

use crate::types::IfranError;
use crate::types::TenantId;
use crate::types::drift::{BaselineSnapshot, DriftResult, DriftSeverity};
use crate::types::error::Result;
use rusqlite::{Connection, params};
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
        let conn = Connection::open(path).map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let det = Self { conn, threshold };
        det.migrate()?;
        Ok(det)
    }

    #[cfg(test)]
    pub fn open_in_memory(threshold: f64) -> Result<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| IfranError::StorageError(e.to_string()))?;
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
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
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
            return Err(IfranError::EvalError(
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
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

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
            return Err(IfranError::EvalError(
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
                    IfranError::EvalError(format!("No baseline for model {model_name}"))
                }
                other => IfranError::StorageError(other.to_string()),
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
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

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
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

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
        det.record_baseline("model", &[0.80, 0.85, 0.82, 0.88, 0.84], &t())
            .unwrap();
        let result = det.check_drift("model", &[0.83, 0.84, 0.82], &t()).unwrap();
        assert!(!result.drifted);
        assert_eq!(result.severity, DriftSeverity::None);
    }

    #[test]
    fn check_drift_detected() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        det.record_baseline("model", &[0.80, 0.85, 0.82, 0.88, 0.84], &t())
            .unwrap();
        // Much lower scores -> drift
        let result = det.check_drift("model", &[0.50, 0.52, 0.48], &t()).unwrap();
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

    #[test]
    fn multiple_baselines_latest_used() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        // First baseline: mean ~0.5
        det.record_baseline("m", &[0.4, 0.5, 0.6], &t()).unwrap();
        // Second baseline: mean ~0.9  (the latest)
        det.record_baseline("m", &[0.88, 0.90, 0.92], &t()).unwrap();

        // Check drift against the latest baseline (mean ~0.9).
        // Scores near 0.5 should drift relative to 0.9 baseline.
        let result = det.check_drift("m", &[0.5, 0.5, 0.5], &t()).unwrap();
        // z_score should be negative (current < baseline mean)
        assert!(result.z_score < 0.0);
    }

    #[test]
    fn negative_drift() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        det.record_baseline("m", &[0.50, 0.55, 0.52, 0.48, 0.50], &t())
            .unwrap();
        // Scores much better than baseline — z-score should be positive (improvement).
        let result = det.check_drift("m", &[0.95, 0.96, 0.97], &t()).unwrap();
        assert!(result.z_score > 0.0); // positive = improvement
    }

    #[test]
    fn single_sample_baseline() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        let baseline = det.record_baseline("m", &[0.75], &t()).unwrap();
        // Single sample => variance = 0, std_dev = 0
        assert_eq!(baseline.sample_count, 1);
        assert_eq!(baseline.std_dev, 0.0);
    }

    #[test]
    fn large_sample_baseline() {
        let det = DriftDetector::open_in_memory(2.0).unwrap();
        // 1000 samples with values 0.001..1.0
        let scores: Vec<f64> = (1..=1000).map(|i| i as f64 / 1000.0).collect();
        let baseline = det.record_baseline("large-model", &scores, &t()).unwrap();
        assert_eq!(baseline.sample_count, 1000);
        // Mean should be ~0.5005
        assert!((baseline.mean_score - 0.5005).abs() < 0.01);
        // Std dev should be ~0.2887 for uniform
        assert!(baseline.std_dev > 0.2);
        assert!(baseline.std_dev < 0.35);
    }

    #[test]
    fn drift_severity_boundary_values() {
        use crate::types::drift::DriftSeverity;
        // Exact boundary values: >1.0, >2.0, >3.0, >4.0
        // At exactly 1.0, should be None (not >1.0)
        assert_eq!(DriftSeverity::from_z_score(1.0), DriftSeverity::None);
        // At exactly 2.0, should be Low (not >2.0)
        assert_eq!(DriftSeverity::from_z_score(2.0), DriftSeverity::Low);
        // At exactly 3.0, should be Medium (not >3.0)
        assert_eq!(DriftSeverity::from_z_score(3.0), DriftSeverity::Medium);
        // At exactly 4.0, should be High (not >4.0)
        assert_eq!(DriftSeverity::from_z_score(4.0), DriftSeverity::High);

        // Just above boundaries
        assert_eq!(DriftSeverity::from_z_score(1.001), DriftSeverity::Low);
        assert_eq!(DriftSeverity::from_z_score(2.001), DriftSeverity::Medium);
        assert_eq!(DriftSeverity::from_z_score(3.001), DriftSeverity::High);
        assert_eq!(DriftSeverity::from_z_score(4.001), DriftSeverity::Critical);
    }
}
