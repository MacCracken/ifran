//! Input validation helpers for API request data.
//!
//! These are called from handlers after deserialization to enforce
//! security constraints on user-provided strings.

use axum::http::StatusCode;

/// Reject prompts that exceed the configured maximum length.
pub fn validate_prompt_length(prompt: &str, max_len: usize) -> Result<(), (StatusCode, String)> {
    if prompt.len() > max_len {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "Prompt exceeds maximum length of {max_len} characters ({} provided)",
                prompt.len()
            ),
        ));
    }
    Ok(())
}

/// Validate a model name: alphanumeric, hyphens, underscores, slashes
/// (for org/name), dots, colons. No `..` or path traversal.
pub fn validate_model_name(name: &str) -> Result<(), (StatusCode, String)> {
    if name.is_empty() || name.len() > 256 {
        return Err((
            StatusCode::BAD_REQUEST,
            "Model name must be 1–256 characters".into(),
        ));
    }
    if name.contains("..") {
        return Err((
            StatusCode::BAD_REQUEST,
            "Model name must not contain '..'".into(),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || "-_./: ".contains(c))
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "Model name contains invalid characters".into(),
        ));
    }
    Ok(())
}

/// Validate a filename used for ingestion: no path separators, no `..`,
/// no hidden files.
pub fn validate_filename(name: &str) -> Result<(), (StatusCode, String)> {
    if name.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Filename must not be empty".into()));
    }
    if name.contains("..") || name.contains('/') || name.contains('\\') || name.starts_with('.') {
        return Err((
            StatusCode::BAD_REQUEST,
            "Filename contains invalid characters or path traversal".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- validate_prompt_length --

    #[test]
    fn prompt_within_limit() {
        assert!(validate_prompt_length("hello", 100).is_ok());
    }

    #[test]
    fn prompt_at_limit() {
        let s = "a".repeat(100);
        assert!(validate_prompt_length(&s, 100).is_ok());
    }

    #[test]
    fn prompt_over_limit() {
        let s = "a".repeat(101);
        let err = validate_prompt_length(&s, 100).unwrap_err();
        assert_eq!(err.0, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(err.1.contains("101"));
    }

    #[test]
    fn prompt_empty_ok() {
        assert!(validate_prompt_length("", 100).is_ok());
    }

    // -- validate_model_name --

    #[test]
    fn model_name_valid_simple() {
        assert!(validate_model_name("llama-7b").is_ok());
    }

    #[test]
    fn model_name_valid_org_slash() {
        assert!(validate_model_name("meta-llama/Llama-2-7B-GGUF").is_ok());
    }

    #[test]
    fn model_name_valid_with_dots() {
        assert!(validate_model_name("model.v2.gguf").is_ok());
    }

    #[test]
    fn model_name_path_traversal() {
        let err = validate_model_name("../etc/passwd").unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn model_name_empty() {
        assert!(validate_model_name("").is_err());
    }

    #[test]
    fn model_name_too_long() {
        let long = "a".repeat(257);
        assert!(validate_model_name(&long).is_err());
    }

    #[test]
    fn model_name_control_chars() {
        assert!(validate_model_name("model\x00name").is_err());
    }

    #[test]
    fn model_name_semicolons() {
        assert!(validate_model_name("model;drop table").is_err());
    }

    // -- validate_filename --

    #[test]
    fn filename_valid() {
        assert!(validate_filename("document.txt").is_ok());
    }

    #[test]
    fn filename_traversal_dotdot() {
        assert!(validate_filename("../secret").is_err());
    }

    #[test]
    fn filename_absolute_path() {
        assert!(validate_filename("/etc/passwd").is_err());
    }

    #[test]
    fn filename_hidden() {
        assert!(validate_filename(".hidden").is_err());
    }

    #[test]
    fn filename_backslash() {
        assert!(validate_filename("dir\\file").is_err());
    }

    #[test]
    fn filename_empty() {
        assert!(validate_filename("").is_err());
    }
}
