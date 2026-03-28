/// Remove a locally stored model.
use crate::config::IfranConfig;
use crate::storage::db::ModelDatabase;
use crate::storage::layout::StorageLayout;
use crate::types::IfranError;
use crate::types::error::Result;

pub async fn execute(model: &str, skip_confirm: bool) -> Result<()> {
    let config = IfranConfig::discover();
    let db = ModelDatabase::open(&config.storage.database)?;

    let tenant = crate::types::TenantId::default_tenant();

    // Try to find by name first, then by UUID
    let model_info = db.get_by_name(model, &tenant).or_else(|_| {
        uuid::Uuid::parse_str(model)
            .map_err(|_| IfranError::ModelNotFound(model.to_string()))
            .and_then(|id| db.get(id, &tenant))
    })?;

    if !skip_confirm {
        eprintln!(
            "Remove '{}' ({:.1} GB)?",
            model_info.name,
            model_info.size_bytes as f64 / 1_000_000_000.0
        );
        eprintln!("  Path: {}", model_info.local_path);
        eprint!("Confirm [y/N]: ");

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| IfranError::Other(e.to_string()))?;

        if !input.trim().eq_ignore_ascii_case("y") {
            eprintln!("Cancelled.");
            return Ok(());
        }
    }

    // Remove files from disk
    let local_path = std::path::Path::new(&model_info.local_path);
    if let Some(model_dir) = local_path.parent() {
        // Only remove the directory if it's inside our models dir
        let layout = StorageLayout::new(
            config
                .storage
                .models_dir
                .parent()
                .unwrap_or(&config.storage.models_dir),
        );
        if model_dir.starts_with(layout.models_dir()) {
            if model_dir.exists() {
                std::fs::remove_dir_all(model_dir)?;
            }
        } else if local_path.exists() {
            // Single file outside our tree — just remove the file
            std::fs::remove_file(local_path)?;
        }
    }

    // Remove from catalog
    db.delete(model_info.id, &tenant)?;

    eprintln!("Removed '{}'", model_info.name);
    Ok(())
}
