//! A/B testing types for model variant comparison.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type AbTestId = Uuid;

/// An A/B test configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTest {
    pub id: AbTestId,
    pub name: String,
    pub model_a: String,
    pub model_b: String,
    /// Traffic fraction for model B (0.0 to 1.0). Rest goes to A.
    pub traffic_split: f64,
    pub status: AbTestStatus,
    pub created_at: DateTime<Utc>,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbTestStatus {
    Active,
    Paused,
    Completed,
}

/// Recorded result for one A/B test request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestResult {
    pub test_id: AbTestId,
    pub variant: String,
    pub score: Option<f64>,
    pub latency_ms: u64,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ab_test_serde() {
        let t = AbTest {
            id: Uuid::new_v4(),
            name: "test-1".into(),
            model_a: "llama-v1".into(),
            model_b: "llama-v2".into(),
            traffic_split: 0.3,
            status: AbTestStatus::Active,
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: AbTest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.traffic_split, 0.3);
    }

    #[test]
    fn status_serde() {
        for s in [
            AbTestStatus::Active,
            AbTestStatus::Paused,
            AbTestStatus::Completed,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: AbTestStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }
}
