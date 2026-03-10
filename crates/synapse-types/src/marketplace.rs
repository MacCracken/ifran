//! Model marketplace types for sharing models between Synapse instances.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::model::{ModelFormat, QuantLevel};

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
