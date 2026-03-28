//! Input validation helpers for API request data.
//!
//! These are called from handlers after deserialization to enforce
//! security constraints on user-provided strings.

use axum::http::StatusCode;

/// Absolute hard cap on input length (50K characters). Config can only be stricter.
const MAX_INPUT_CHARS: usize = 50_000;

/// Reject prompts that exceed the configured maximum length.
///
/// A hard cap of [`MAX_INPUT_CHARS`] is always enforced regardless of what the
/// caller passes as `max_len`.
pub fn validate_prompt_length(prompt: &str, max_len: usize) -> Result<(), (StatusCode, String)> {
    let effective_limit = max_len.min(MAX_INPUT_CHARS);
    if prompt.len() > effective_limit {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "Prompt exceeds maximum length of {effective_limit} characters ({} provided)",
                prompt.len()
            ),
        ));
    }
    Ok(())
}

/// Wrap user content in boundary markers to prevent prompt confusion.
///
/// This adds delimiters that separate user content from system instructions,
/// making it harder for injected text to be interpreted as system-level directives.
#[must_use]
#[inline]
pub fn sanitize_prompt(user_content: &str) -> String {
    format!("<|user_input_start|>\n{user_content}\n<|user_input_end|>")
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

    #[test]
    fn prompt_hard_cap_enforced_when_config_higher() {
        // Config says 100_000 but hard cap is 50_000
        let s = "a".repeat(50_001);
        let err = validate_prompt_length(&s, 100_000).unwrap_err();
        assert_eq!(err.0, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(err.1.contains("50000"));
    }

    #[test]
    fn prompt_config_stricter_than_hard_cap() {
        // Config says 1000, which is stricter than 50K hard cap
        let s = "a".repeat(1001);
        let err = validate_prompt_length(&s, 1000).unwrap_err();
        assert_eq!(err.0, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(err.1.contains("1000"));
    }

    #[test]
    fn prompt_at_hard_cap_ok() {
        let s = "a".repeat(50_000);
        assert!(validate_prompt_length(&s, 100_000).is_ok());
    }

    // -- sanitize_prompt --

    #[test]
    fn sanitize_wraps_content() {
        let result = sanitize_prompt("hello world");
        assert_eq!(
            result,
            "<|user_input_start|>\nhello world\n<|user_input_end|>"
        );
    }

    #[test]
    fn sanitize_empty_prompt() {
        let result = sanitize_prompt("");
        assert_eq!(result, "<|user_input_start|>\n\n<|user_input_end|>");
    }

    #[test]
    fn sanitize_prompt_with_existing_markers() {
        // Content that already contains boundary markers gets double-wrapped — outer markers are authoritative
        let input = "<|user_input_start|>\nmalicious\n<|user_input_end|>";
        let result = sanitize_prompt(input);
        assert!(result.starts_with("<|user_input_start|>\n"));
        assert!(result.ends_with("\n<|user_input_end|>"));
        // The original markers are inside the outer wrapping
        assert!(result.contains("<|user_input_start|>\nmalicious\n<|user_input_end|>"));
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
