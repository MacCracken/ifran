//! Filesystem layout for model storage.
//!
//! Manages the directory structure under `~/.synapse/` (or a configured root).
//! Each model gets its own directory named by a slug derived from the model name.
//!
//! ```text
//! ~/.synapse/
//! ├── synapse.db          # SQLite catalog
//! ├── cache/              # Temporary download cache
//! └── models/
//!     ├── llama-3.1-8b-q4km/
//!     │   ├── model.gguf
//!     │   └── metadata.json
//!     └── mistral-7b-q5km/
//!         ├── model.gguf
//!         └── metadata.json
//! ```

use std::path::{Path, PathBuf};
use synapse_types::SynapseError;
use synapse_types::error::Result;

/// Manages the Synapse directory layout on the filesystem.
pub struct StorageLayout {
    root: PathBuf,
}

impl StorageLayout {
    /// Create a layout rooted at the given directory (e.g. `~/.synapse`).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Create a layout using the default `~/.synapse` path.
    pub fn default_location() -> Result<Self> {
        let home = std::env::var("HOME")
            .map_err(|_| SynapseError::StorageError("HOME environment variable not set".into()))?;
        Ok(Self::new(PathBuf::from(home).join(".synapse")))
    }

    /// Root directory (e.g. `~/.synapse`).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to the SQLite database file.
    pub fn database_path(&self) -> PathBuf {
        self.root.join("synapse.db")
    }

    /// Root models directory.
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    /// Directory for a specific model, derived from a slugified name.
    pub fn model_dir(&self, slug: &str) -> PathBuf {
        self.models_dir().join(slug)
    }

    /// Path to the primary model file within a model directory.
    pub fn model_file(&self, slug: &str, filename: &str) -> PathBuf {
        self.model_dir(slug).join(filename)
    }

    /// Path to the metadata sidecar JSON for a model.
    pub fn model_metadata(&self, slug: &str) -> PathBuf {
        self.model_dir(slug).join("metadata.json")
    }

    /// Temporary download cache directory.
    pub fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }

    /// Checkpoints directory for training artifacts.
    pub fn checkpoints_dir(&self) -> PathBuf {
        self.root.join("checkpoints")
    }

    /// Configuration file path.
    pub fn config_path(&self) -> PathBuf {
        self.root.join("synapse.toml")
    }

    /// Ensure all required directories exist.
    pub fn ensure_dirs(&self) -> Result<()> {
        let dirs = [
            self.root.clone(),
            self.models_dir(),
            self.cache_dir(),
            self.checkpoints_dir(),
        ];
        for dir in &dirs {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }

    /// Generate a filesystem-safe slug from a model name and quant level.
    ///
    /// Examples:
    /// - `("meta-llama/Llama-3.1-8B-Instruct", "q4_k_m")` → `"llama-3.1-8b-instruct-q4km"`
    /// - `("mistral-7b", "f16")` → `"mistral-7b-f16"`
    pub fn slugify(name: &str, quant: &str) -> String {
        let base = name.rsplit('/').next().unwrap_or(name);

        let mut slug = String::with_capacity(base.len() + quant.len() + 1);
        for ch in base.chars() {
            match ch {
                'A'..='Z' => slug.push(ch.to_ascii_lowercase()),
                'a'..='z' | '0'..='9' => slug.push(ch),
                '.' => slug.push('.'),
                '-' => slug.push('-'),
                '_' | ' ' => slug.push('-'),
                _ => {} // drop special chars
            }
        }
        // Collapse consecutive dashes while preserving dots
        let mut collapsed = String::with_capacity(slug.len());
        let mut prev_dash = false;
        for ch in slug.chars() {
            if ch == '-' {
                if !prev_dash && !collapsed.is_empty() {
                    collapsed.push('-');
                }
                prev_dash = true;
            } else {
                prev_dash = false;
                collapsed.push(ch);
            }
        }
        // Trim trailing dash
        let collapsed = collapsed.trim_end_matches('-');

        let quant_clean: String = quant
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect();

        if quant_clean.is_empty() || quant_clean == "none" {
            collapsed.to_string()
        } else {
            format!("{collapsed}-{quant_clean}")
        }
    }

    /// Remove a model's directory and all its contents.
    pub fn remove_model_dir(&self, slug: &str) -> Result<()> {
        let dir = self.model_dir(slug);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn slugify_basic() {
        assert_eq!(
            StorageLayout::slugify("meta-llama/Llama-3.1-8B-Instruct", "q4_k_m"),
            "llama-3.1-8b-instruct-q4km"
        );
    }

    #[test]
    fn slugify_no_quant() {
        assert_eq!(StorageLayout::slugify("mistral-7b", "none"), "mistral-7b");
    }

    #[test]
    fn slugify_simple() {
        assert_eq!(StorageLayout::slugify("my_model", "f16"), "my-model-f16");
    }

    #[test]
    fn paths_are_correct() {
        let layout = StorageLayout::new("/tmp/synapse-test-layout");
        assert_eq!(
            layout.database_path(),
            PathBuf::from("/tmp/synapse-test-layout/synapse.db")
        );
        assert_eq!(
            layout.models_dir(),
            PathBuf::from("/tmp/synapse-test-layout/models")
        );
        assert_eq!(
            layout.model_dir("llama-8b-q4km"),
            PathBuf::from("/tmp/synapse-test-layout/models/llama-8b-q4km")
        );
        assert_eq!(
            layout.model_metadata("llama-8b-q4km"),
            PathBuf::from("/tmp/synapse-test-layout/models/llama-8b-q4km/metadata.json")
        );
    }

    #[test]
    fn ensure_dirs_creates_structure() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = StorageLayout::new(tmp.path().join("synapse"));
        layout.ensure_dirs().unwrap();
        assert!(layout.models_dir().is_dir());
        assert!(layout.cache_dir().is_dir());
        assert!(layout.checkpoints_dir().is_dir());
    }

    #[test]
    fn remove_model_dir_works() {
        let tmp = tempfile::tempdir().unwrap();
        let layout = StorageLayout::new(tmp.path().join("synapse"));
        layout.ensure_dirs().unwrap();
        let model_dir = layout.model_dir("test-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.gguf"), b"fake").unwrap();
        assert!(model_dir.exists());
        layout.remove_model_dir("test-model").unwrap();
        assert!(!model_dir.exists());
    }
}
