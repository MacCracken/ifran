//! Model publisher — creates marketplace entries from local models.

use synapse_types::error::Result;
use synapse_types::marketplace::MarketplaceEntry;
use synapse_types::model::ModelInfo;

/// Creates a marketplace entry from a local model.
///
/// The `instance_id` identifies this Synapse instance, and `base_url`
/// is the externally-reachable URL for this node's API server.
pub fn create_entry(
    model: &ModelInfo,
    instance_id: &str,
    base_url: &str,
) -> Result<MarketplaceEntry> {
    let download_url = format!(
        "{}/marketplace/download/{}",
        base_url.trim_end_matches('/'),
        urlencoded_name(&model.name)
    );

    Ok(MarketplaceEntry {
        model_name: model.name.clone(),
        description: model.architecture.as_ref().map(|a| format!("{a} model")),
        format: model.format,
        quant: model.quant,
        size_bytes: model.size_bytes,
        parameter_count: model.parameter_count,
        architecture: model.architecture.clone(),
        publisher_instance: instance_id.to_string(),
        download_url,
        sha256: model.sha256.clone(),
        tags: Vec::new(),
        published_at: chrono::Utc::now(),
        eval_scores: None,
    })
}

fn urlencoded_name(name: &str) -> String {
    name.replace('/', "__")
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_types::model::{ModelFormat, QuantLevel};

    #[test]
    fn create_entry_builds_download_url() {
        let model = ModelInfo {
            id: uuid::Uuid::new_v4(),
            name: "meta-llama/Llama-3.1-8B".into(),
            repo_id: None,
            format: ModelFormat::Gguf,
            quant: QuantLevel::Q4KM,
            size_bytes: 4_000_000_000,
            parameter_count: None,
            architecture: None,
            license: None,
            local_path: "/tmp/model.gguf".into(),
            sha256: None,
            pulled_at: chrono::Utc::now(),
        };
        let entry = create_entry(&model, "node-1", "http://localhost:8420").unwrap();
        assert_eq!(
            entry.download_url,
            "http://localhost:8420/marketplace/download/meta-llama__Llama-3.1-8B"
        );
        assert_eq!(entry.publisher_instance, "node-1");
    }
}
