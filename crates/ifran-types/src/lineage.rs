//! Pipeline lineage types for tracking provenance chains.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type LineageId = Uuid;

/// A node in the lineage graph representing one pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    pub id: LineageId,
    pub stage: PipelineStage,
    pub name: String,
    /// Reference to the artifact (model ID, dataset path, eval run ID, etc.)
    pub artifact_ref: String,
    /// Parent node IDs (inputs to this stage).
    pub parent_ids: Vec<LineageId>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

/// Pipeline stage types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    Dataset,
    Training,
    Evaluation,
    Deployment,
    Checkpoint,
    Merge,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset => write!(f, "dataset"),
            Self::Training => write!(f, "training"),
            Self::Evaluation => write!(f, "evaluation"),
            Self::Deployment => write!(f, "deployment"),
            Self::Checkpoint => write!(f, "checkpoint"),
            Self::Merge => write!(f, "merge"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_serde_roundtrip() {
        for stage in [
            PipelineStage::Dataset,
            PipelineStage::Training,
            PipelineStage::Evaluation,
            PipelineStage::Deployment,
            PipelineStage::Checkpoint,
            PipelineStage::Merge,
        ] {
            let json = serde_json::to_string(&stage).unwrap();
            let back: PipelineStage = serde_json::from_str(&json).unwrap();
            assert_eq!(stage, back);
        }
    }

    #[test]
    fn lineage_node_serde() {
        let node = LineageNode {
            id: Uuid::new_v4(),
            stage: PipelineStage::Training,
            name: "lora-run-1".into(),
            artifact_ref: "job-123".into(),
            parent_ids: vec![Uuid::new_v4()],
            metadata: serde_json::json!({"method": "lora"}),
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: LineageNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "lora-run-1");
        assert_eq!(back.stage, PipelineStage::Training);
    }

    #[test]
    fn stage_display() {
        assert_eq!(PipelineStage::Dataset.to_string(), "dataset");
        assert_eq!(PipelineStage::Deployment.to_string(), "deployment");
    }
}
