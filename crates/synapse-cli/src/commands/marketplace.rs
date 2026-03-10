//! CLI commands for the model marketplace.

use synapse_core::config::SynapseConfig;
use synapse_core::marketplace::catalog::MarketplaceCatalog;
use synapse_core::marketplace::publisher;
use synapse_core::storage::db::ModelDatabase;
use synapse_types::marketplace::MarketplaceQuery;

/// Search the marketplace.
pub async fn search(query: Option<&str>) -> synapse_types::error::Result<()> {
    let config = SynapseConfig::discover();
    let catalog_path = config.storage.database.with_file_name("marketplace.db");
    let catalog = MarketplaceCatalog::open(&catalog_path)?;

    let mq = MarketplaceQuery {
        search: query.map(String::from),
        format: None,
        tags: None,
        max_size_bytes: None,
    };

    let entries = catalog.search(&mq)?;

    if entries.is_empty() {
        println!("No marketplace entries found.");
        return Ok(());
    }

    println!(
        "{:<40} {:<10} {:<12} {:<20}",
        "MODEL", "FORMAT", "SIZE", "PUBLISHER"
    );
    println!("{}", "-".repeat(82));
    for entry in &entries {
        let size = format_size(entry.size_bytes);
        let format = serde_json::to_string(&entry.format)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        println!(
            "{:<40} {:<10} {:<12} {:<20}",
            entry.model_name, format, size, entry.publisher_instance
        );
    }

    Ok(())
}

/// Publish a local model to the marketplace.
pub async fn publish(model_name: &str) -> synapse_types::error::Result<()> {
    let config = SynapseConfig::discover();
    let db = ModelDatabase::open(&config.storage.database)?;
    let model = db.get_by_name(model_name)?;

    let instance_id =
        std::env::var("SYNAPSE_INSTANCE_ID").unwrap_or_else(|_| config.server.bind.clone());
    let base_url = format!("http://{}", config.server.bind);

    let entry = publisher::create_entry(&model, &instance_id, &base_url)?;

    let catalog_path = config.storage.database.with_file_name("marketplace.db");
    let catalog = MarketplaceCatalog::open(&catalog_path)?;
    catalog.publish(&entry)?;

    println!("Published '{}' to marketplace", entry.model_name);
    println!("Download URL: {}", entry.download_url);

    Ok(())
}

/// Unpublish a model from the marketplace.
pub async fn unpublish(model_name: &str) -> synapse_types::error::Result<()> {
    let config = SynapseConfig::discover();
    let catalog_path = config.storage.database.with_file_name("marketplace.db");
    let catalog = MarketplaceCatalog::open(&catalog_path)?;
    catalog.unpublish(model_name)?;
    println!("Unpublished '{model_name}' from marketplace");
    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{} B", bytes)
    }
}
