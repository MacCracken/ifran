//! Model marketplace types for sharing models between Ifran instances.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::model::{ModelFormat, QuantLevel};

/// A model published to the marketplace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceEntry {
    pub model_name: String,
    pub description: Option<String>,
    pub format: ModelFormat,
    pub quant: QuantLevel,
    pub size_bytes: u64,
    pub parameter_count: Option<u64>,
    pub architecture: Option<String>,
    /// Instance ID of the publisher.
    pub publisher_instance: String,
    /// URL to download the model files.
    pub download_url: String,
    pub sha256: Option<String>,
    pub tags: Vec<String>,
    pub published_at: DateTime<Utc>,
    /// Benchmark scores (benchmark_name → score).
    pub eval_scores: Option<HashMap<String, f64>>,
}

/// Query parameters for marketplace search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarketplaceQuery {
    pub search: Option<String>,
    pub format: Option<ModelFormat>,
    pub tags: Option<Vec<String>>,
    pub max_size_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marketplace_entry_serde() {
        let entry = MarketplaceEntry {
            model_name: "my-finetune".into(),
            description: Some("A fine-tuned model".into()),
            format: ModelFormat::Gguf,
            quant: QuantLevel::Q4KM,
            size_bytes: 4_000_000_000,
            parameter_count: Some(7_000_000_000),
            architecture: Some("llama".into()),
            publisher_instance: "node-1".into(),
            download_url: "http://node-1:8420/models/my-finetune".into(),
            sha256: Some("deadbeef".into()),
            tags: vec!["llama".into(), "chat".into()],
            published_at: Utc::now(),
            eval_scores: Some(HashMap::from([("mmlu".into(), 0.85)])),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: MarketplaceEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_name, "my-finetune");
        assert_eq!(back.tags.len(), 2);
        assert_eq!(back.eval_scores.unwrap()["mmlu"], 0.85);
    }

    #[test]
    fn marketplace_query_default() {
        let query = MarketplaceQuery::default();
        assert!(query.search.is_none());
        assert!(query.format.is_none());
        assert!(query.tags.is_none());
        assert!(query.max_size_bytes.is_none());
    }

    #[test]
    fn marketplace_query_serde() {
        let query = MarketplaceQuery {
            search: Some("llama".into()),
            format: Some(ModelFormat::Gguf),
            tags: Some(vec!["chat".into()]),
            max_size_bytes: Some(10_000_000_000),
        };
        let json = serde_json::to_string(&query).unwrap();
        let back: MarketplaceQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(back.search.unwrap(), "llama");
        assert_eq!(back.max_size_bytes, Some(10_000_000_000));
    }
}
