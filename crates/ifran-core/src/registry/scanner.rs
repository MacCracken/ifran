//! Local filesystem model scanner.
//!
//! Scans a directory tree for model files (GGUF, SafeTensors, ONNX, etc.)
//! and returns metadata about each discovered model. Used to import
//! models that were manually placed on disk.

use std::path::{Path, PathBuf};
use ifran_types::error::Result;
use ifran_types::model::ModelFormat;

/// A model file discovered on the filesystem.
#[derive(Debug, Clone)]
pub struct ScannedModel {
    pub path: PathBuf,
    pub filename: String,
    pub format: ModelFormat,
    pub size_bytes: u64,
}

/// Scan a directory (non-recursively) for model files.
pub fn scan_dir(dir: &Path) -> Result<Vec<ScannedModel>> {
    let mut models = Vec::new();

    if !dir.exists() {
        return Ok(models);
    }

    for entry in std::fs::read_dir(dir)?.flatten() {
        let path = entry.path();

        // If it's a subdirectory, scan one level into it
        if path.is_dir() {
            for sub_entry in std::fs::read_dir(&path)?.flatten() {
                if let Some(model) = try_scan_file(&sub_entry.path()) {
                    models.push(model);
                }
            }
        } else if let Some(model) = try_scan_file(&path) {
            models.push(model);
        }
    }

    Ok(models)
}

/// Try to identify a single file as a model.
fn try_scan_file(path: &Path) -> Option<ScannedModel> {
    if !path.is_file() {
        return None;
    }

    let filename = path.file_name()?.to_string_lossy().to_string();
    let format = detect_format(&filename)?;
    let size_bytes = std::fs::metadata(path).ok()?.len();

    Some(ScannedModel {
        path: path.to_path_buf(),
        filename,
        format,
        size_bytes,
    })
}

/// Detect model format from file extension.
fn detect_format(filename: &str) -> Option<ModelFormat> {
    let lower = filename.to_lowercase();
    if lower.ends_with(".gguf") {
        Some(ModelFormat::Gguf)
    } else if lower.ends_with(".safetensors") {
        Some(ModelFormat::SafeTensors)
    } else if lower.ends_with(".onnx") {
        Some(ModelFormat::Onnx)
    } else if lower.ends_with(".engine") || lower.ends_with(".trt") {
        Some(ModelFormat::TensorRt)
    } else if lower.ends_with(".pt") || lower.ends_with(".pth") {
        Some(ModelFormat::PyTorch)
    } else if lower.ends_with(".bin") && !lower.contains("tokenizer") {
        Some(ModelFormat::Bin)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn detect_formats() {
        assert_eq!(detect_format("model.gguf"), Some(ModelFormat::Gguf));
        assert_eq!(
            detect_format("model.safetensors"),
            Some(ModelFormat::SafeTensors)
        );
        assert_eq!(detect_format("model.onnx"), Some(ModelFormat::Onnx));
        assert_eq!(detect_format("model.pt"), Some(ModelFormat::PyTorch));
        assert_eq!(detect_format("model.bin"), Some(ModelFormat::Bin));
        assert_eq!(detect_format("tokenizer.bin"), None);
        assert_eq!(detect_format("readme.md"), None);
    }

    #[test]
    fn scan_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let models = scan_dir(tmp.path()).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn scan_finds_models() {
        let tmp = tempfile::tempdir().unwrap();

        // Direct file
        fs::write(tmp.path().join("model.gguf"), b"fake gguf").unwrap();

        // File in subdirectory
        let sub = tmp.path().join("my-model");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("weights.safetensors"), b"fake safetensors").unwrap();

        // Non-model file (should be ignored)
        fs::write(tmp.path().join("readme.md"), b"hello").unwrap();

        let models = scan_dir(tmp.path()).unwrap();
        assert_eq!(models.len(), 2);

        let formats: Vec<_> = models.iter().map(|m| m.format).collect();
        assert!(formats.contains(&ModelFormat::Gguf));
        assert!(formats.contains(&ModelFormat::SafeTensors));
    }

    #[test]
    fn scan_nonexistent_dir() {
        let models = scan_dir(Path::new("/nonexistent/path")).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn detect_tensorrt_formats() {
        assert_eq!(detect_format("model.engine"), Some(ModelFormat::TensorRt));
        assert_eq!(detect_format("model.trt"), Some(ModelFormat::TensorRt));
    }

    #[test]
    fn detect_pytorch_pth() {
        assert_eq!(detect_format("model.pth"), Some(ModelFormat::PyTorch));
    }

    #[test]
    fn detect_case_insensitive() {
        assert_eq!(detect_format("MODEL.GGUF"), Some(ModelFormat::Gguf));
        assert_eq!(
            detect_format("Weights.SafeTensors"),
            Some(ModelFormat::SafeTensors)
        );
        assert_eq!(detect_format("MODEL.ONNX"), Some(ModelFormat::Onnx));
    }

    #[test]
    fn scan_reports_correct_size() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"0123456789"; // 10 bytes
        fs::write(tmp.path().join("model.gguf"), content).unwrap();

        let models = scan_dir(tmp.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].size_bytes, 10);
        assert_eq!(models[0].filename, "model.gguf");
        assert_eq!(models[0].format, ModelFormat::Gguf);
    }

    #[test]
    fn scan_multiple_files_in_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("my-model");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("model.gguf"), b"gguf").unwrap();
        fs::write(sub.join("model.safetensors"), b"st").unwrap();
        fs::write(sub.join("config.json"), b"{}").unwrap(); // ignored

        let models = scan_dir(tmp.path()).unwrap();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn tokenizer_bin_excluded() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("tokenizer.bin"), b"tok").unwrap();
        fs::write(tmp.path().join("model.bin"), b"mod").unwrap();

        let models = scan_dir(tmp.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].format, ModelFormat::Bin);
    }
}
