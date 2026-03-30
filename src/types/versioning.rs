//! Model versioning types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type ModelVersionId = Uuid;

/// A versioned model entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub id: ModelVersionId,
    /// The base model name/family (e.g., "llama-3.1-8b").
    pub model_family: String,
    /// Semantic version tag (e.g., "v1", "v2-lora-custom").
    pub version_tag: String,
    /// Reference to the model in the catalog.
    pub model_id: Option<Uuid>,
    /// Training job that produced this version (if any).
    pub training_job_id: Option<Uuid>,
    /// Parent version this was derived from (if any).
    pub parent_version_id: Option<ModelVersionId>,
    /// Who/what this model version is for.
    pub consumer: Option<String>,
    pub notes: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_version_serde() {
        let v = ModelVersion {
            id: Uuid::new_v4(),
            model_family: "llama-3.1-8b".into(),
            version_tag: "v2-lora".into(),
            model_id: Some(Uuid::new_v4()),
            training_job_id: None,
            parent_version_id: None,
            consumer: Some("chatbot".into()),
            notes: Some("Fine-tuned for support".into()),
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: ModelVersion = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_family, "llama-3.1-8b");
        assert_eq!(back.version_tag, "v2-lora");
    }
}
