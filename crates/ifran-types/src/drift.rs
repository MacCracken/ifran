//! Drift detection types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type SnapshotId = Uuid;

/// A baseline snapshot for drift comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    pub id: SnapshotId,
    pub model_name: String,
    pub mean_score: f64,
    pub std_dev: f64,
    pub sample_count: u64,
    pub created_at: DateTime<Utc>,
}

/// Result of a drift check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftResult {
    pub snapshot_id: SnapshotId,
    pub current_mean: f64,
    pub z_score: f64,
    pub drifted: bool,
    pub severity: DriftSeverity,
    pub checked_at: DateTime<Utc>,
}

/// Drift severity levels based on z-score thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftSeverity {
    None,
    Low,      // |z| > 1.0
    Medium,   // |z| > 2.0
    High,     // |z| > 3.0
    Critical, // |z| > 4.0
}

impl DriftSeverity {
    pub fn from_z_score(z: f64) -> Self {
        let abs_z = z.abs();
        if abs_z > 4.0 {
            Self::Critical
        } else if abs_z > 3.0 {
            Self::High
        } else if abs_z > 2.0 {
            Self::Medium
        } else if abs_z > 1.0 {
            Self::Low
        } else {
            Self::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_from_z_score() {
        assert_eq!(DriftSeverity::from_z_score(0.5), DriftSeverity::None);
        assert_eq!(DriftSeverity::from_z_score(1.5), DriftSeverity::Low);
        assert_eq!(DriftSeverity::from_z_score(-2.5), DriftSeverity::Medium);
        assert_eq!(DriftSeverity::from_z_score(3.5), DriftSeverity::High);
        assert_eq!(DriftSeverity::from_z_score(-4.5), DriftSeverity::Critical);
    }

    #[test]
    fn severity_serde_roundtrip() {
        for s in [
            DriftSeverity::None,
            DriftSeverity::Low,
            DriftSeverity::Medium,
            DriftSeverity::High,
            DriftSeverity::Critical,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: DriftSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn baseline_snapshot_serde() {
        let s = BaselineSnapshot {
            id: Uuid::new_v4(),
            model_name: "llama-8b".into(),
            mean_score: 0.85,
            std_dev: 0.05,
            sample_count: 100,
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: BaselineSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_name, "llama-8b");
    }

    #[test]
    fn drift_result_serde() {
        let r = DriftResult {
            snapshot_id: Uuid::new_v4(),
            current_mean: 0.75,
            z_score: -2.0,
            drifted: true,
            severity: DriftSeverity::Medium,
            checked_at: Utc::now(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: DriftResult = serde_json::from_str(&json).unwrap();
        assert!(back.drifted);
    }
}
