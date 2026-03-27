//! Dataset loader for reading training data from various sources and formats.

use ifran_types::IfranError;
use ifran_types::error::Result;
use ifran_types::training::{DatasetConfig, DatasetFormat};
use std::path::Path;

/// A loaded dataset ready for training.
#[derive(Debug)]
pub struct LoadedDataset {
    pub path: String,
    pub format: DatasetFormat,
    pub sample_count: usize,
}

/// Load and validate a dataset from the given config.
pub fn load(config: &DatasetConfig) -> Result<LoadedDataset> {
    let path = Path::new(&config.path);
    if !path.exists() {
        return Err(IfranError::TrainingError(format!(
            "Dataset not found: {}",
            config.path
        )));
    }

    let sample_count = count_samples(path, config.format)?;

    let effective_count = match config.max_samples {
        Some(max) => sample_count.min(max),
        None => sample_count,
    };

    Ok(LoadedDataset {
        path: config.path.clone(),
        format: config.format,
        sample_count: effective_count,
    })
}

/// Count samples in a dataset file.
fn count_samples(path: &Path, format: DatasetFormat) -> Result<usize> {
    match format {
        DatasetFormat::Jsonl => {
            let content = std::fs::read_to_string(path)?;
            Ok(content.lines().filter(|l| !l.trim().is_empty()).count())
        }
        DatasetFormat::Csv => {
            let content = std::fs::read_to_string(path)?;
            // Subtract header row
            Ok(content
                .lines()
                .filter(|l| !l.trim().is_empty())
                .count()
                .saturating_sub(1))
        }
        DatasetFormat::Parquet => {
            // Parquet needs a specialized reader — return file-based estimate
            let size = std::fs::metadata(path)?.len();
            Ok((size / 500) as usize) // rough estimate: ~500 bytes per row
        }
        DatasetFormat::HuggingFace => {
            // HuggingFace datasets are loaded by the training script
            Ok(0) // count determined at runtime by the script
        }
        _ => Ok(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_jsonl() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, r#"{{"text": "hello"}}"#).unwrap();
        writeln!(tmp, r#"{{"text": "world"}}"#).unwrap();
        tmp.flush().unwrap();

        let config = DatasetConfig {
            path: tmp.path().to_string_lossy().to_string(),
            format: DatasetFormat::Jsonl,
            split: None,
            max_samples: None,
        };
        let dataset = load(&config).unwrap();
        assert_eq!(dataset.sample_count, 2);
    }

    #[test]
    fn load_with_max_samples() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for i in 0..100 {
            writeln!(tmp, r#"{{"text": "sample {i}"}}"#).unwrap();
        }
        tmp.flush().unwrap();

        let config = DatasetConfig {
            path: tmp.path().to_string_lossy().to_string(),
            format: DatasetFormat::Jsonl,
            split: None,
            max_samples: Some(10),
        };
        let dataset = load(&config).unwrap();
        assert_eq!(dataset.sample_count, 10);
    }

    #[test]
    fn load_missing_file() {
        let config = DatasetConfig {
            path: "/nonexistent/data.jsonl".to_string(),
            format: DatasetFormat::Jsonl,
            split: None,
            max_samples: None,
        };
        assert!(load(&config).is_err());
    }

    #[test]
    fn load_csv() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "text,label").unwrap();
        writeln!(tmp, "hello,1").unwrap();
        writeln!(tmp, "world,0").unwrap();
        writeln!(tmp, "foo,1").unwrap();
        tmp.flush().unwrap();

        let config = DatasetConfig {
            path: tmp.path().to_string_lossy().to_string(),
            format: DatasetFormat::Csv,
            split: None,
            max_samples: None,
        };
        let dataset = load(&config).unwrap();
        assert_eq!(dataset.sample_count, 3); // 4 lines minus header
        assert_eq!(dataset.format, DatasetFormat::Csv);
    }

    #[test]
    fn load_parquet() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        // Write some dummy bytes to simulate a parquet file
        tmp.write_all(&[0u8; 5000]).unwrap();
        tmp.flush().unwrap();

        let config = DatasetConfig {
            path: tmp.path().to_string_lossy().to_string(),
            format: DatasetFormat::Parquet,
            split: None,
            max_samples: None,
        };
        let dataset = load(&config).unwrap();
        assert_eq!(dataset.sample_count, 10); // 5000 / 500
    }

    #[test]
    fn load_huggingface() {
        let tmp = tempfile::NamedTempFile::new().unwrap();

        let config = DatasetConfig {
            path: tmp.path().to_string_lossy().to_string(),
            format: DatasetFormat::HuggingFace,
            split: None,
            max_samples: None,
        };
        let dataset = load(&config).unwrap();
        assert_eq!(dataset.sample_count, 0); // HF returns 0
    }

    #[test]
    fn load_csv_with_max_samples() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "text,label").unwrap();
        for i in 0..50 {
            writeln!(tmp, "sample{i},1").unwrap();
        }
        tmp.flush().unwrap();

        let config = DatasetConfig {
            path: tmp.path().to_string_lossy().to_string(),
            format: DatasetFormat::Csv,
            split: None,
            max_samples: Some(5),
        };
        let dataset = load(&config).unwrap();
        assert_eq!(dataset.sample_count, 5);
    }
}
