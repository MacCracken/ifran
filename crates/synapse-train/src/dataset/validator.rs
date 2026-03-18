//! Dataset validator for verifying data integrity and format compliance.

use std::path::Path;
use synapse_types::SynapseError;
use synapse_types::error::Result;
use synapse_types::training::DatasetFormat;

/// Validation result.
#[derive(Debug, serde::Serialize)]
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

/// Count CSV columns handling RFC 4180 quoted fields.
fn count_csv_columns(line: &str) -> usize {
    if line.is_empty() {
        return 0;
    }
    let mut cols = 1;
    let mut in_quotes = false;
    for c in line.chars() {
        match c {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => cols += 1,
            _ => {}
        }
    }
    cols
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

    let header_cols = count_csv_columns(lines[0]);
    let mut invalid = 0;
    let mut errors = Vec::new();

    for (i, line) in lines.iter().skip(1).enumerate() {
        let cols = count_csv_columns(line);
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

    #[test]
    fn invalid_csv_column_mismatch() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "text,label").unwrap();
        writeln!(tmp, "hello,1").unwrap();
        writeln!(tmp, "bad_row").unwrap(); // only 1 column instead of 2
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Csv).unwrap();
        assert!(!result.valid);
        assert_eq!(result.invalid_rows, 1);
        assert_eq!(result.total_rows, 2);
    }

    #[test]
    fn empty_csv() {
        let tmp = tempfile::NamedTempFile::new().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Csv).unwrap();
        assert!(!result.valid);
        assert_eq!(result.errors[0], "Empty CSV file");
    }

    #[test]
    fn parquet_passthrough() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let result = validate(tmp.path(), DatasetFormat::Parquet).unwrap();
        assert!(result.valid);
        assert_eq!(result.total_rows, 0);
    }

    #[test]
    fn huggingface_passthrough() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let result = validate(tmp.path(), DatasetFormat::HuggingFace).unwrap();
        assert!(result.valid);
        assert_eq!(result.total_rows, 0);
    }

    #[test]
    fn csv_with_quoted_commas() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "text,label").unwrap();
        writeln!(tmp, r#""hello, world",1"#).unwrap();
        writeln!(tmp, r#""foo",0"#).unwrap();
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Csv).unwrap();
        assert!(result.valid);
        assert_eq!(result.total_rows, 2);
    }

    #[test]
    fn count_csv_columns_basic() {
        assert_eq!(super::count_csv_columns("a,b,c"), 3);
        assert_eq!(super::count_csv_columns(r#""a,b",c"#), 2);
        assert_eq!(super::count_csv_columns(""), 0);
        assert_eq!(super::count_csv_columns("single"), 1);
    }

    #[test]
    fn missing_file() {
        let result = validate(Path::new("/nonexistent/file.jsonl"), DatasetFormat::Jsonl);
        assert!(result.is_err());
    }

    #[test]
    fn jsonl_with_blank_lines() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, r#"{{"text": "hello"}}"#).unwrap();
        writeln!(tmp).unwrap(); // blank line
        writeln!(tmp, r#"{{"text": "world"}}"#).unwrap();
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Jsonl).unwrap();
        assert!(result.valid);
        assert_eq!(result.total_rows, 2);
    }

    #[test]
    fn jsonl_all_invalid() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "not json 1").unwrap();
        writeln!(tmp, "not json 2").unwrap();
        writeln!(tmp, "not json 3").unwrap();
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Jsonl).unwrap();
        assert!(!result.valid);
        assert_eq!(result.total_rows, 3);
        assert_eq!(result.invalid_rows, 3);
        assert_eq!(result.errors.len(), 3);
    }

    #[test]
    fn jsonl_errors_capped_at_10() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for i in 0..15 {
            writeln!(tmp, "invalid line {i}").unwrap();
        }
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Jsonl).unwrap();
        assert!(!result.valid);
        assert_eq!(result.invalid_rows, 15);
        assert_eq!(result.errors.len(), 10); // capped
    }

    #[test]
    fn csv_header_only() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "col1,col2,col3").unwrap();
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Csv).unwrap();
        assert!(result.valid);
        assert_eq!(result.total_rows, 0);
    }

    #[test]
    fn csv_many_columns() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "a,b,c,d,e").unwrap();
        writeln!(tmp, "1,2,3,4,5").unwrap();
        writeln!(tmp, "6,7,8,9,10").unwrap();
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Csv).unwrap();
        assert!(result.valid);
        assert_eq!(result.total_rows, 2);
    }

    #[test]
    fn count_csv_columns_quoted_commas_complex() {
        // Deeply nested quoted field
        assert_eq!(count_csv_columns(r#""a,b,c",d,"e,f""#), 3);
    }

    #[test]
    fn jsonl_empty_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let result = validate(tmp.path(), DatasetFormat::Jsonl).unwrap();
        assert!(result.valid); // no rows = no invalid rows
        assert_eq!(result.total_rows, 0);
    }

    #[test]
    fn csv_column_mismatch_multiple() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "a,b").unwrap();
        writeln!(tmp, "1,2").unwrap();
        writeln!(tmp, "x").unwrap(); // 1 col
        writeln!(tmp, "y").unwrap(); // 1 col
        writeln!(tmp, "3,4").unwrap();
        tmp.flush().unwrap();

        let result = validate(tmp.path(), DatasetFormat::Csv).unwrap();
        assert!(!result.valid);
        assert_eq!(result.invalid_rows, 2);
        assert_eq!(result.total_rows, 4);
    }
}
