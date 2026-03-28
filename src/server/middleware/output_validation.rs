//! Inference output validation — JSON mode enforcement with auto-retry.
//!
//! Validates LLM output against expected formats (plain text, JSON, or JSON
//! with required keys). Returns a [`ValidationResult`] indicating whether the
//! output conforms to the expected [`OutputFormat`].

use std::fmt;

use serde_json::Value;

/// Expected output format for validation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum OutputFormat {
    /// No validation — accept any text.
    Text,
    /// Must be valid JSON.
    Json,
    /// Must be valid JSON object containing all `required_keys`.
    JsonSchema { required_keys: Vec<String> },
}

/// Result of output validation.
#[derive(Debug)]
pub enum ValidationResult {
    Valid,
    Invalid { reason: String },
}

impl ValidationResult {
    /// Returns `true` if the output passed validation.
    #[inline]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Valid => f.write_str("Valid"),
            Self::Invalid { reason } => write!(f, "Invalid: {reason}"),
        }
    }
}

/// Validate inference output against the expected format.
#[must_use]
pub fn validate_output(text: &str, format: &OutputFormat) -> ValidationResult {
    match format {
        OutputFormat::Text => ValidationResult::Valid,
        OutputFormat::Json => match serde_json::from_str::<Value>(text.trim()) {
            Ok(_) => ValidationResult::Valid,
            Err(e) => ValidationResult::Invalid {
                reason: format!("Invalid JSON: {e}"),
            },
        },
        OutputFormat::JsonSchema { required_keys } => {
            match serde_json::from_str::<Value>(text.trim()) {
                Ok(Value::Object(map)) => {
                    for key in required_keys {
                        if !map.contains_key(key) {
                            return ValidationResult::Invalid {
                                reason: format!("Missing required key: {key}"),
                            };
                        }
                    }
                    ValidationResult::Valid
                }
                Ok(_) => ValidationResult::Invalid {
                    reason: "Expected JSON object, got different type".into(),
                },
                Err(e) => ValidationResult::Invalid {
                    reason: format!("Invalid JSON: {e}"),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_mode_always_valid() {
        assert!(validate_output("anything at all!!", &OutputFormat::Text).is_valid());
        assert!(validate_output("", &OutputFormat::Text).is_valid());
        assert!(validate_output("{not json", &OutputFormat::Text).is_valid());
    }

    #[test]
    fn valid_json_passes() {
        let result = validate_output(r#"{"key": "value"}"#, &OutputFormat::Json);
        assert!(result.is_valid());
    }

    #[test]
    fn valid_json_array_passes() {
        let result = validate_output(r#"[1, 2, 3]"#, &OutputFormat::Json);
        assert!(result.is_valid());
    }

    #[test]
    fn invalid_json_fails() {
        let result = validate_output("{not json}", &OutputFormat::Json);
        assert!(!result.is_valid());
        if let ValidationResult::Invalid { reason } = result {
            assert!(reason.contains("Invalid JSON"));
        }
    }

    #[test]
    fn empty_string_fails_json_mode() {
        let result = validate_output("", &OutputFormat::Json);
        assert!(!result.is_valid());
    }

    #[test]
    fn json_schema_with_required_keys_validates() {
        let format = OutputFormat::JsonSchema {
            required_keys: vec!["name".into(), "age".into()],
        };
        let result = validate_output(r#"{"name": "Alice", "age": 30}"#, &format);
        assert!(result.is_valid());
    }

    #[test]
    fn missing_required_key_fails() {
        let format = OutputFormat::JsonSchema {
            required_keys: vec!["name".into(), "age".into()],
        };
        let result = validate_output(r#"{"name": "Alice"}"#, &format);
        assert!(!result.is_valid());
        if let ValidationResult::Invalid { reason } = result {
            assert!(reason.contains("Missing required key: age"));
        }
    }

    #[test]
    fn array_instead_of_object_fails_schema() {
        let format = OutputFormat::JsonSchema {
            required_keys: vec!["key".into()],
        };
        let result = validate_output(r#"[1, 2, 3]"#, &format);
        assert!(!result.is_valid());
        if let ValidationResult::Invalid { reason } = result {
            assert!(reason.contains("Expected JSON object"));
        }
    }

    #[test]
    fn json_with_extra_keys_still_passes() {
        let format = OutputFormat::JsonSchema {
            required_keys: vec!["id".into()],
        };
        let result = validate_output(r#"{"id": 1, "extra": "data", "more": true}"#, &format);
        assert!(result.is_valid());
    }

    #[test]
    fn json_schema_empty_required_keys_accepts_any_object() {
        let format = OutputFormat::JsonSchema {
            required_keys: vec![],
        };
        let result = validate_output(r#"{"anything": true}"#, &format);
        assert!(result.is_valid());
    }

    #[test]
    fn whitespace_trimmed_before_json_parse() {
        let result = validate_output("  \n{\"ok\": true}\n  ", &OutputFormat::Json);
        assert!(result.is_valid());
    }

    #[test]
    fn validation_result_display() {
        assert_eq!(ValidationResult::Valid.to_string(), "Valid");
        let invalid = ValidationResult::Invalid {
            reason: "bad".into(),
        };
        assert_eq!(invalid.to_string(), "Invalid: bad");
    }
}
