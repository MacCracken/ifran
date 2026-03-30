//! Dataset curation and refresh types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type DatasetId = Uuid;

/// A curated dataset entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuratedDataset {
    pub id: DatasetId,
    pub name: String,
    pub source_path: String,
    pub sample_count: u64,
    pub format: String,
    pub version: u32,
    pub fingerprint: String,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_dataset_serde() {
        let d = CuratedDataset {
            id: Uuid::new_v4(),
            name: "train-v3".into(),
            source_path: "/data/train.jsonl".into(),
            sample_count: 10000,
            format: "jsonl".into(),
            version: 3,
            fingerprint: "abc123".into(),
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: CuratedDataset = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "train-v3");
        assert_eq!(back.version, 3);
    }
}
