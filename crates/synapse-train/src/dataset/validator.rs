//! Dataset validator for verifying data integrity and format compliance.

use std::path::Path;
use synapse_types::SynapseError;
use synapse_types::error::Result;
use synapse_types::training::DatasetFormat;

/// Validation result.
#[derive(Debug)]
pub struct ValidationResult {
    pub valid: bool,
    pub total_rows: usize,
    pub invalid_rows: usize,
    pub errors: Vec<String>,
}

/// Validate a dataset file for format compliance.
pub fn validate(path: &Path, format: DatasetFormat) -> Result<ValidationResult> {
    if !path.exists() {
        return Err(SynapseError::TrainingError(format!(
            "Dataset not found: {}",
            path.display()
        )));
    }

    match format {
        DatasetFormat::Jsonl => validate_jsonl(path),
        DatasetFormat::Csv => validate_csv(path),
        DatasetFormat::Parquet | DatasetFormat::HuggingFace => {
            // These formats are validated by their respective loaders
            Ok(ValidationResult {
                valid: true,
                total_rows: 0,
                invalid_rows: 0,
                errors: Vec::new(),
            })
        }
    }
}

fn validate_jsonl(path: &Path) -> Result<ValidationResult> {
    let content = std::fs::read_to_string(path)?;
    let mut total = 0;
    let mut invalid = 0;
    let mut errors = Vec::new();

    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        total += 1;

        if let Err(e) = serde_json::from_str::<serde_json::Value>(line) {
            invalid += 1;
            if errors.len() < 10 {
                errors.push(format!("Line {}: {e}", i + 1));
            }
        }
    }

    Ok(ValidationResult {
        valid: invalid == 0,
        total_rows: total,
        invalid_rows: invalid,
        errors,
    })
}

fn validate_csv(path: &Path) -> Result<ValidationResult> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();

    if lines.is_empty() {
        return Ok(ValidationResult {
            valid: false,
            total_rows: 0,
            invalid_rows: 0,
            errors: vec!["Empty CSV file".into()],
        });
    }

    let header_cols = lines[0].split(',').count();
    let mut invalid = 0;
    let mut errors = Vec::new();

    for (i, line) in lines.iter().skip(1).enumerate() {
        let cols = line.split(',').count();
        if cols != header_cols {
            invalid += 1;
            if errors.len() < 10 {
                errors.push(format!(
                    "Row {}: expected {header_cols} columns, got {cols}",
                    i + 2
                ));
            }
        }
    }

    Ok(ValidationResult {
        valid: invalid == 0,
        total_rows: lines.len() - 1,
        invalid_rows: invalid,
        errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn valid_jsonl() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, r#"{{"text": "hello"}}"#).unwrap();
        writeln!(tmp, r#"{{"text": "world"}}"#).unwrap();
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Jsonl).unwrap();
        assert!(result.valid);
        assert_eq!(result.total_rows, 2);
    }

    #[test]
    fn invalid_jsonl() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, r#"{{"text": "hello"}}"#).unwrap();
        writeln!(tmp, "not json").unwrap();
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Jsonl).unwrap();
        assert!(!result.valid);
        assert_eq!(result.invalid_rows, 1);
    }

    #[test]
    fn valid_csv() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "text,label").unwrap();
        writeln!(tmp, "hello,1").unwrap();
        writeln!(tmp, "world,0").unwrap();
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Csv).unwrap();
        assert!(result.valid);
        assert_eq!(result.total_rows, 2);
    }
}
