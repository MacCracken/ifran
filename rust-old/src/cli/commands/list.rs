/// List all locally available models.
use crate::cli::output::{self, Table};
use crate::config::IfranConfig;
use crate::storage::db::ModelDatabase;
use crate::types::error::Result;

pub async fn execute() -> Result<()> {
    let config = IfranConfig::discover();
    let db = ModelDatabase::open(&config.storage.database)?;
    let tenant = crate::types::TenantId::default_tenant();
    let paged = db.list(&tenant, 1000, 0)?;

    if paged.items.is_empty() {
        output::warn("No models found. Use 'ifran pull <model>' to download one.");
        return Ok(());
    }

    let mut table = Table::new(vec!["NAME", "FORMAT", "QUANT", "SIZE", "PULLED"]);

    for model in &paged.items {
        let size = output::format_size(model.size_bytes);
        let format = serde_json::to_string(&model.format)
            .unwrap_or_else(|_| "unknown".into())
            .trim_matches('"')
            .to_string();
        let quant = serde_json::to_string(&model.quant)
            .unwrap_or_else(|_| "unknown".into())
            .trim_matches('"')
            .to_string();
        let pulled = model.pulled_at.format("%Y-%m-%d").to_string();

        table.add_row(vec![
            output::truncate(&model.name, 45),
            format,
            quant,
            size,
            pulled,
        ]);
    }

    table.print();
    output::info(&format!("\n{} model(s)", paged.items.len()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::cli::output;

    #[test]
    fn format_size_gigabytes() {
        assert_eq!(output::format_size(4_000_000_000), "4.0 GB");
        assert_eq!(output::format_size(1_500_000_000), "1.5 GB");
        assert_eq!(output::format_size(13_200_000_000), "13.2 GB");
    }

    #[test]
    fn format_size_megabytes() {
        assert_eq!(output::format_size(500_000_000), "500 MB");
        assert_eq!(output::format_size(100_000_000), "100 MB");
        assert_eq!(output::format_size(1_000_000), "1 MB");
    }

    #[test]
    fn format_size_boundary() {
        assert_eq!(output::format_size(1_000_000_000), "1.0 GB");
        assert_eq!(output::format_size(999_999_999), "1000 MB");
    }

    #[test]
    fn truncate_short_string() {
        assert_eq!(output::truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_exact_length() {
        assert_eq!(output::truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_long_string() {
        let result = output::truncate("this is a very long model name", 10);
        assert!(result.len() <= 12);
        assert!(result.ends_with('…'));
    }

    #[test]
    fn format_size_zero() {
        assert_eq!(output::format_size(0), "0 MB");
    }
}
