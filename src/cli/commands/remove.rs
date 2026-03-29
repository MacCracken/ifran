/// Remove a locally stored model.
use crate::config::IfranConfig;
use crate::storage::db::ModelDatabase;
use crate::storage::layout::StorageLayout;
use crate::types::IfranError;
use crate::types::error::Result;
use std::path::Path;

/// Check whether a model directory is safely inside the managed models directory.
///
/// Returns `true` if `model_dir` is a subdirectory of `models_root`, meaning it
/// is safe to remove the entire directory. Returns `false` if the path is outside
/// the managed tree (in which case only the individual file should be removed).
#[must_use]
fn is_safe_removal_path(model_dir: &Path, models_root: &Path) -> bool {
    model_dir.starts_with(models_root)
}

/// Format the confirmation prompt for model removal.
#[must_use]
fn format_removal_prompt(name: &str, size_bytes: u64) -> String {
    format!(
        "Remove '{}' ({:.1} GB)?",
        name,
        size_bytes as f64 / 1_000_000_000.0
    )
}

/// Check if user input confirms removal (case-insensitive "y").
#[must_use]
fn is_confirmed(input: &str) -> bool {
    input.trim().eq_ignore_ascii_case("y")
}

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
            "{}",
            format_removal_prompt(&model_info.name, model_info.size_bytes)
        );
        eprintln!("  Path: {}", model_info.local_path);
        eprint!("Confirm [y/N]: ");

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| IfranError::Other(e.to_string()))?;

        if !is_confirmed(&input) {
            eprintln!("Cancelled.");
            return Ok(());
        }
    }

    // Remove files from disk
    let local_path = Path::new(&model_info.local_path);
    if let Some(model_dir) = local_path.parent() {
        // Only remove the directory if it's inside our models dir
        let layout = StorageLayout::new(
            config
                .storage
                .models_dir
                .parent()
                .unwrap_or(&config.storage.models_dir),
        );
        if is_safe_removal_path(model_dir, &layout.models_dir()) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn safe_removal_path_inside_models_dir() {
        let models_root = PathBuf::from("/data/ifran/models");
        let model_dir = PathBuf::from("/data/ifran/models/llama-7b__q4km");
        assert!(is_safe_removal_path(&model_dir, &models_root));
    }

    #[test]
    fn safe_removal_path_nested_inside() {
        let models_root = PathBuf::from("/data/ifran/models");
        let model_dir = PathBuf::from("/data/ifran/models/org/model/v1");
        assert!(is_safe_removal_path(&model_dir, &models_root));
    }

    #[test]
    fn unsafe_removal_path_outside_models_dir() {
        let models_root = PathBuf::from("/data/ifran/models");
        let model_dir = PathBuf::from("/tmp/some-model");
        assert!(!is_safe_removal_path(&model_dir, &models_root));
    }

    #[test]
    fn unsafe_removal_path_sibling_dir() {
        let models_root = PathBuf::from("/data/ifran/models");
        let model_dir = PathBuf::from("/data/ifran/config");
        assert!(!is_safe_removal_path(&model_dir, &models_root));
    }

    #[test]
    fn unsafe_removal_path_parent_traversal() {
        let models_root = PathBuf::from("/data/ifran/models");
        let model_dir = PathBuf::from("/data/ifran");
        assert!(!is_safe_removal_path(&model_dir, &models_root));
    }

    #[test]
    fn safe_removal_path_exact_match() {
        let models_root = PathBuf::from("/data/ifran/models");
        let model_dir = PathBuf::from("/data/ifran/models");
        // starts_with returns true for exact match
        assert!(is_safe_removal_path(&model_dir, &models_root));
    }

    #[test]
    fn format_removal_prompt_display() {
        let prompt = format_removal_prompt("llama-7b", 7_000_000_000);
        assert_eq!(prompt, "Remove 'llama-7b' (7.0 GB)?");
    }

    #[test]
    fn format_removal_prompt_small_model() {
        let prompt = format_removal_prompt("tiny-model", 500_000_000);
        assert_eq!(prompt, "Remove 'tiny-model' (0.5 GB)?");
    }

    #[test]
    fn format_removal_prompt_zero_size() {
        let prompt = format_removal_prompt("empty", 0);
        assert_eq!(prompt, "Remove 'empty' (0.0 GB)?");
    }

    #[test]
    fn is_confirmed_yes() {
        assert!(is_confirmed("y"));
        assert!(is_confirmed("Y"));
        assert!(is_confirmed("y\n"));
        assert!(is_confirmed("  y  "));
        assert!(is_confirmed("Y\n"));
    }

    #[test]
    fn is_confirmed_no() {
        assert!(!is_confirmed("n"));
        assert!(!is_confirmed("N"));
        assert!(!is_confirmed(""));
        assert!(!is_confirmed("yes"));
        assert!(!is_confirmed("no"));
        assert!(!is_confirmed("\n"));
    }
}
