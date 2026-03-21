//! CLI commands for the model marketplace.

use ifran_core::config::IfranConfig;
use ifran_core::marketplace::catalog::MarketplaceCatalog;
use ifran_core::marketplace::publisher;
use ifran_core::storage::db::ModelDatabase;
use ifran_types::marketplace::MarketplaceQuery;

/// Search the marketplace.
pub async fn search(query: Option<&str>) -> ifran_types::error::Result<()> {
    let config = IfranConfig::discover();
    let catalog_path = config.storage.database.with_file_name("marketplace.db");
    let catalog = MarketplaceCatalog::open(&catalog_path)?;

    let mq = MarketplaceQuery {
        search: query.map(String::from),
        format: None,
        tags: None,
        max_size_bytes: None,
    };

    let tenant = ifran_types::TenantId::default_tenant();
    let entries = catalog.search(&mq, &tenant)?;

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
pub async fn publish(model_name: &str) -> ifran_types::error::Result<()> {
    let config = IfranConfig::discover();
    let db = ModelDatabase::open(&config.storage.database)?;
    let tenant = ifran_types::TenantId::default_tenant();
    let model = db.get_by_name(model_name, &tenant)?;

    let instance_id =
        std::env::var("IFRAN_INSTANCE_ID").unwrap_or_else(|_| config.server.bind.clone());
    let base_url = format!("http://{}", config.server.bind);

    let entry = publisher::create_entry(&model, &instance_id, &base_url)?;

    let catalog_path = config.storage.database.with_file_name("marketplace.db");
    let catalog = MarketplaceCatalog::open(&catalog_path)?;
    catalog.publish(&entry, &tenant)?;

    println!("Published '{}' to marketplace", entry.model_name);
    println!("Download URL: {}", entry.download_url);

    Ok(())
}

/// Unpublish a model from the marketplace.
pub async fn unpublish(model_name: &str) -> ifran_types::error::Result<()> {
    let config = IfranConfig::discover();
    let catalog_path = config.storage.database.with_file_name("marketplace.db");
    let catalog = MarketplaceCatalog::open(&catalog_path)?;
    let tenant = ifran_types::TenantId::default_tenant();
    catalog.unpublish(model_name, &tenant)?;
    println!("Unpublished '{model_name}' from marketplace");
    Ok(())
}

/// Pull a model from a remote marketplace peer.
pub async fn pull(model_name: &str, peer_url: &str) -> ifran_types::error::Result<()> {
    use ifran_core::marketplace::resolver::MarketplaceResolver;
    use ifran_core::marketplace::trust::{TrustPolicy, verify_entry};

    let mut resolver = MarketplaceResolver::new();
    resolver.add_peer(peer_url.to_string());

    let query = MarketplaceQuery {
        search: Some(model_name.to_string()),
        ..Default::default()
    };

    let results = resolver.search_remote(&query).await?;
    let entry = results.iter().find(|e| e.model_name == model_name).ok_or(
        ifran_types::IfranError::MarketplaceError(format!(
            "Model '{model_name}' not found on peer {peer_url}"
        )),
    )?;

    // Verify against trust policy
    let policy = TrustPolicy::default();
    let trust_level = verify_entry(entry, &policy)?;
    println!("Trust level: {trust_level:?}");

    // Download the model
    let config = IfranConfig::discover();
    let safe_name = model_name.replace('/', "__");
    let dest = config.storage.models_dir.join(&safe_name);

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let client = ifran_core::pull::downloader::build_client()?;
    let progress = ifran_core::pull::progress::ProgressTracker::new(16);
    let download_req = ifran_core::pull::downloader::DownloadRequest {
        url: entry.download_url.clone(),
        dest: dest.clone(),
        model_name: model_name.to_string(),
        expected_sha256: entry.sha256.clone(),
    };

    println!("Downloading '{}' from {}...", model_name, peer_url);
    ifran_core::pull::downloader::download(&client, &download_req, &progress).await?;

    // Verify download integrity
    if entry.sha256.is_some() {
        let dl_trust = ifran_core::marketplace::trust::verify_download(&dest, entry)?;
        println!("Download verified: {dl_trust:?}");
    }

    println!("Pulled '{}' to {}", model_name, dest.display());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_size_gb() {
        assert_eq!(format_size(4_700_000_000), "4.7 GB");
        assert_eq!(format_size(1_000_000_000), "1.0 GB");
    }

    #[test]
    fn format_size_mb() {
        assert_eq!(format_size(500_000_000), "500.0 MB");
        assert_eq!(format_size(1_000_000), "1.0 MB");
    }

    #[test]
    fn format_size_bytes() {
        assert_eq!(format_size(999_999), "999999 B");
        assert_eq!(format_size(0), "0 B");
    }
}
