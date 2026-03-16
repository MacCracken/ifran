/// List all locally available models.
use synapse_core::config::SynapseConfig;
use synapse_core::storage::db::ModelDatabase;
use synapse_types::error::Result;

pub async fn execute() -> Result<()> {
    let config = SynapseConfig::discover();
    let db = ModelDatabase::open(&config.storage.database)?;
    let tenant = synapse_types::TenantId::default_tenant();
    let models = db.list(&tenant)?;

    if models.is_empty() {
        eprintln!("No models found. Use 'synapse pull <model>' to download one.");
        return Ok(());
    }

    // Header
    println!(
        "{:<40} {:<12} {:<8} {:<12} PULLED",
        "NAME", "FORMAT", "QUANT", "SIZE"
    );
    println!("{}", "-".repeat(90));

    for model in &models {
        let size = format_size(model.size_bytes);
        let format = format!("{:?}", model.format).to_lowercase();
        let quant = format!("{:?}", model.quant);
        let pulled = model.pulled_at.format("%Y-%m-%d").to_string();

        println!(
            "{:<40} {:<12} {:<8} {:<12} {}",
            truncate(&model.name, 39),
            format,
            quant,
            size,
            pulled,
        );
    }

    eprintln!("\n{} model(s)", models.len());
    Ok(())
}

fn format_size(bytes: u64) -> String {
    const GB: u64 = 1_000_000_000;
    const MB: u64 = 1_000_000;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_size_gigabytes() {
        assert_eq!(format_size(4_000_000_000), "4.0 GB");
        assert_eq!(format_size(1_500_000_000), "1.5 GB");
        assert_eq!(format_size(13_200_000_000), "13.2 GB");
    }

    #[test]
    fn format_size_megabytes() {
        assert_eq!(format_size(500_000_000), "500 MB");
        assert_eq!(format_size(100_000_000), "100 MB");
        assert_eq!(format_size(1_000_000), "1 MB");
    }

    #[test]
    fn format_size_boundary() {
        // Exactly 1 GB
        assert_eq!(format_size(1_000_000_000), "1.0 GB");
        // Just under 1 GB
        assert_eq!(format_size(999_999_999), "1000 MB");
    }

    #[test]
    fn truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_exact_length() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_long_string() {
        let result = truncate("this is a very long model name", 10);
        assert!(result.len() <= 12); // 9 chars + ellipsis (multi-byte)
        assert!(result.ends_with('…'));
    }

    #[test]
    fn format_size_zero() {
        assert_eq!(format_size(0), "0 MB");
    }
}
