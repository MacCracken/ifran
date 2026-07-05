//! Output filtering — redacts secrets and PII from inference responses.
//!
//! Scans inference output for leaked API keys, PII (email, phone, SSN,
//! credit cards), and system-prompt leakage. All matches are replaced
//! with `[REDACTED_<CATEGORY>]` markers.
//!
//! No `regex` crate — all scanning is done with manual helpers.

use std::fmt::Write as _;

/// Result of filtering a piece of text.
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// The text with sensitive content replaced.
    pub text: String,
    /// Metadata about each redaction performed.
    pub redactions: Vec<Redaction>,
}

/// A single redaction event.
#[derive(Debug, Clone)]
pub struct Redaction {
    /// Human-readable category such as `"AWS_KEY"` or `"EMAIL"`.
    pub category: &'static str,
    /// Byte length of the original matched text.
    pub original_length: usize,
}

// ---------------------------------------------------------------------------
// Fixed-string system-prompt leak patterns
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT_PHRASES: &[&str] = &[
    "system prompt",
    "my instructions",
    "I was told to",
    "my initial prompt",
    "<|system|>",
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Scan and redact sensitive content from model output.
#[must_use]
pub fn filter_output(text: &str) -> FilterResult {
    let mut redactions: Vec<Redaction> = Vec::new();
    let mut buf = text.to_string();

    // 1. Fixed-string patterns (system prompt leaks)
    // Pre-compute lowercase once for all fixed-pattern scans
    let lower = buf.to_ascii_lowercase();
    for &phrase in SYSTEM_PROMPT_PHRASES {
        redact_fixed(&mut buf, &lower, phrase, "SYSTEM_PROMPT", &mut redactions);
    }

    // 2. Structured patterns — order matters: longer / more specific first.
    redact_pattern(&mut buf, find_aws_key, "AWS_KEY", &mut redactions);
    redact_pattern(&mut buf, find_github_token, "GITHUB_TOKEN", &mut redactions);
    redact_pattern(&mut buf, find_bearer_token, "BEARER_TOKEN", &mut redactions);
    redact_pattern(&mut buf, find_generic_api_key, "API_KEY", &mut redactions);
    redact_pattern(&mut buf, find_ssn, "SSN", &mut redactions);
    redact_pattern(&mut buf, find_credit_card, "CREDIT_CARD", &mut redactions);
    redact_pattern(&mut buf, find_email, "EMAIL", &mut redactions);
    redact_pattern(&mut buf, find_us_phone, "US_PHONE", &mut redactions);

    FilterResult {
        text: buf,
        redactions,
    }
}

// ---------------------------------------------------------------------------
// Redaction helpers
// ---------------------------------------------------------------------------

/// Replace all case-insensitive occurrences of `needle` in `haystack`.
///
/// `lower` is the pre-computed lowercase version of `haystack` (avoids
/// re-lowercasing on every call).
fn redact_fixed(
    haystack: &mut String,
    lower: &str,
    needle: &str,
    category: &'static str,
    redactions: &mut Vec<Redaction>,
) {
    let needle_lower = needle.to_ascii_lowercase();
    let mut replacement = String::new();
    let _ = write!(replacement, "[REDACTED_{category}]");

    let mut start = 0;
    let mut result = String::with_capacity(haystack.len());
    while let Some(pos) = lower[start..].find(&needle_lower) {
        let abs = start + pos;
        result.push_str(&haystack[start..abs]);
        result.push_str(&replacement);
        redactions.push(Redaction {
            category,
            original_length: needle.len(),
        });
        start = abs + needle.len();
    }
    if !redactions.is_empty() || start > 0 {
        result.push_str(&haystack[start..]);
        if result != *haystack {
            *haystack = result;
        }
    }
}

/// Generic scanner: repeatedly find & redact the first match returned by `finder`.
fn redact_pattern(
    haystack: &mut String,
    finder: fn(&str) -> Option<(usize, usize)>,
    category: &'static str,
    redactions: &mut Vec<Redaction>,
) {
    let mut replacement = String::new();
    let _ = write!(replacement, "[REDACTED_{category}]");

    loop {
        let Some((start, end)) = finder(haystack) else {
            break;
        };
        let original_length = end - start;
        let mut new = String::with_capacity(haystack.len());
        new.push_str(&haystack[..start]);
        new.push_str(&replacement);
        new.push_str(&haystack[end..]);
        *haystack = new;
        redactions.push(Redaction {
            category,
            original_length,
        });
    }
}

// ---------------------------------------------------------------------------
// Pattern finders — each returns Option<(start, end)> of the first match.
// ---------------------------------------------------------------------------

/// `(?:AKIA|ASIA)[A-Z0-9]{16}`
fn find_aws_key(text: &str) -> Option<(usize, usize)> {
    for prefix in &["AKIA", "ASIA"] {
        if let Some(pos) = text.find(prefix) {
            let rest = &text[pos + 4..];
            let count = rest
                .chars()
                .take(16)
                .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
                .count();
            if count == 16 {
                return Some((pos, pos + 4 + 16));
            }
        }
    }
    None
}

/// `gh[ps]_[A-Za-z0-9_]{36,}`
fn find_github_token(text: &str) -> Option<(usize, usize)> {
    for prefix in &["ghp_", "ghs_"] {
        let mut search_from = 0;
        while let Some(rel) = text[search_from..].find(prefix) {
            let pos = search_from + rel;
            let rest = &text[pos + 4..];
            let count = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .count();
            if count >= 36 {
                return Some((pos, pos + 4 + count));
            }
            search_from = pos + 4;
        }
    }
    None
}

/// `Bearer [A-Za-z0-9\-._~+/]+=*`
fn find_bearer_token(text: &str) -> Option<(usize, usize)> {
    let needle = "Bearer ";
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find(needle) {
        let pos = search_from + rel;
        let rest = &text[pos + needle.len()..];
        let count = rest
            .chars()
            .take_while(|c| {
                c.is_ascii_alphanumeric()
                    || *c == '-'
                    || *c == '.'
                    || *c == '_'
                    || *c == '~'
                    || *c == '+'
                    || *c == '/'
                    || *c == '='
            })
            .count();
        if count >= 1 {
            return Some((pos, pos + needle.len() + count));
        }
        search_from = pos + needle.len();
    }
    None
}

/// Generic API key: `(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[=:]\s*["']?[A-Za-z0-9\-._~+/]{20,}`
fn find_generic_api_key(text: &str) -> Option<(usize, usize)> {
    let lower = text.to_ascii_lowercase();
    let labels = [
        "api_key",
        "api-key",
        "apikey",
        "secret_key",
        "secret-key",
        "secretkey",
        "access_token",
        "access-token",
        "accesstoken",
    ];

    let mut best: Option<(usize, usize)> = None;

    for label in &labels {
        let mut search_from = 0;
        while let Some(rel) = lower[search_from..].find(label) {
            let abs = search_from + rel;
            let after_label = abs + label.len();
            // skip whitespace
            let rest = &text[after_label..];
            let ws1 = rest.chars().take_while(|c| c.is_ascii_whitespace()).count();
            let rest = &rest[ws1..];
            // expect = or :
            if rest.is_empty() || (rest.as_bytes()[0] != b'=' && rest.as_bytes()[0] != b':') {
                search_from = after_label;
                continue;
            }
            let rest = &rest[1..];
            let ws2 = rest.chars().take_while(|c| c.is_ascii_whitespace()).count();
            let rest = &rest[ws2..];
            // optional quote
            let quote_skip = if rest.starts_with('"') || rest.starts_with('\'') {
                1
            } else {
                0
            };
            let rest = &rest[quote_skip..];
            let value_len = rest
                .chars()
                .take_while(|c| {
                    c.is_ascii_alphanumeric()
                        || *c == '-'
                        || *c == '.'
                        || *c == '_'
                        || *c == '~'
                        || *c == '+'
                        || *c == '/'
                })
                .count();
            if value_len >= 20 {
                let end = after_label + ws1 + 1 + ws2 + quote_skip + value_len;
                let candidate = (abs, end);
                if best.is_none_or(|b| candidate.0 < b.0) {
                    best = Some(candidate);
                }
            }
            search_from = after_label;
        }
    }
    best
}

/// SSN: `\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b`
fn find_ssn(text: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    if bytes.len() < 11 {
        return None;
    }
    for i in 0..=bytes.len() - 11 {
        // word boundary before
        if i > 0 && bytes[i - 1].is_ascii_alphanumeric() {
            continue;
        }
        if bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3] == b'-'
            && bytes[i + 4].is_ascii_digit()
            && bytes[i + 5].is_ascii_digit()
            && bytes[i + 6] == b'-'
            && bytes[i + 7].is_ascii_digit()
            && bytes[i + 8].is_ascii_digit()
            && bytes[i + 9].is_ascii_digit()
            && bytes[i + 10].is_ascii_digit()
        {
            // word boundary after
            if i + 11 < bytes.len() && bytes[i + 11].is_ascii_alphanumeric() {
                continue;
            }
            return Some((i, i + 11));
        }
    }
    None
}

/// Credit card: `\b[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}\b`
fn find_credit_card(text: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        // word boundary before
        if i > 0 && bytes[i - 1].is_ascii_digit() {
            i += 1;
            continue;
        }
        if let Some(end) = try_parse_cc(bytes, i) {
            // word boundary after
            if end < len && bytes[end].is_ascii_digit() {
                i += 1;
                continue;
            }
            return Some((i, end));
        }
        i += 1;
    }
    None
}

fn try_parse_cc(bytes: &[u8], start: usize) -> Option<usize> {
    let mut pos = start;
    for group in 0..4 {
        for _ in 0..4 {
            if pos >= bytes.len() || !bytes[pos].is_ascii_digit() {
                return None;
            }
            pos += 1;
        }
        if group < 3 && pos < bytes.len() && (bytes[pos] == b'-' || bytes[pos] == b' ') {
            pos += 1;
        }
    }
    Some(pos)
}

/// Email: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`
fn find_email(text: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    for (idx, &b) in bytes.iter().enumerate() {
        if b != b'@' {
            continue;
        }
        // Walk backwards for local part
        let local_start = {
            let mut s = idx;
            while s > 0 {
                let c = bytes[s - 1];
                if c.is_ascii_alphanumeric()
                    || c == b'.'
                    || c == b'_'
                    || c == b'%'
                    || c == b'+'
                    || c == b'-'
                {
                    s -= 1;
                } else {
                    break;
                }
            }
            s
        };
        if local_start == idx {
            continue; // empty local part
        }
        // Walk forward for domain
        let after_at = idx + 1;
        let mut end = after_at;
        while end < bytes.len()
            && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'.' || bytes[end] == b'-')
        {
            end += 1;
        }
        // Must contain at least one dot in the domain with 2+ char TLD
        let domain = &text[after_at..end];
        if let Some(last_dot) = domain.rfind('.') {
            let tld = &domain[last_dot + 1..];
            if tld.len() >= 2 && tld.chars().all(|c| c.is_ascii_alphabetic()) {
                return Some((local_start, end));
            }
        }
    }
    None
}

/// US phone: `(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}`
fn find_us_phone(text: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if let Some((start, end)) = try_parse_phone(bytes, i) {
            return Some((start, end));
        }
        i += 1;
    }
    None
}

fn try_parse_phone(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let len = bytes.len();
    let mut pos = start;

    // Optional +1 prefix
    if pos < len && bytes[pos] == b'+' {
        if pos + 1 < len && bytes[pos + 1] == b'1' {
            pos += 2;
            if pos < len && is_phone_sep(bytes[pos]) {
                pos += 1;
            }
        } else {
            return None;
        }
    }

    // Optional (
    let has_paren = pos < len && bytes[pos] == b'(';
    if has_paren {
        pos += 1;
    }

    // 3 digits (area code)
    let area_start = pos;
    for _ in 0..3 {
        if pos >= len || !bytes[pos].is_ascii_digit() {
            return None;
        }
        pos += 1;
    }
    // avoid matching if there are more leading digits before area code
    if area_start > start
        && !has_paren
        && bytes[area_start - 1] != b'+'
        && bytes[area_start - 1] != b'1'
        && !is_phone_sep(bytes[area_start - 1])
    {
        // If preceding char is a digit, this is probably just a number
        if start > 0 && bytes[start - 1].is_ascii_digit() {
            return None;
        }
    }

    if has_paren {
        if pos >= len || bytes[pos] != b')' {
            return None;
        }
        pos += 1;
    }

    if pos < len && is_phone_sep(bytes[pos]) {
        pos += 1;
    }

    // 3 digits
    for _ in 0..3 {
        if pos >= len || !bytes[pos].is_ascii_digit() {
            return None;
        }
        pos += 1;
    }

    if pos < len && is_phone_sep(bytes[pos]) {
        pos += 1;
    }

    // 4 digits
    for _ in 0..4 {
        if pos >= len || !bytes[pos].is_ascii_digit() {
            return None;
        }
        pos += 1;
    }

    // Must not be followed by more digits
    if pos < len && bytes[pos].is_ascii_digit() {
        return None;
    }

    Some((start, pos))
}

#[inline]
fn is_phone_sep(b: u8) -> bool {
    b == b'-' || b == b'.' || b == b' '
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text_unchanged() {
        let result = filter_output("Hello, world! This is a normal response.");
        assert_eq!(result.text, "Hello, world! This is a normal response.");
        assert!(result.redactions.is_empty());
    }

    #[test]
    fn aws_key_redacted() {
        let input = "My key is AKIAIOSFODNN7EXAMPLE1";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_AWS_KEY]"));
        assert!(!result.text.contains("AKIAIOSFODNN7EXAMPLE1"));
        assert_eq!(result.redactions.len(), 1);
        assert_eq!(result.redactions[0].category, "AWS_KEY");
    }

    #[test]
    fn asia_key_redacted() {
        let input = "Temp creds: ASIAQWERTYUIOP1234AB";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_AWS_KEY]"));
        assert!(!result.text.contains("ASIAQWERTYUIOP1234AB"));
    }

    #[test]
    fn github_token_redacted() {
        let token = format!("ghp_{}", "a".repeat(40));
        let input = format!("Use this token: {token}");
        let result = filter_output(&input);
        assert!(result.text.contains("[REDACTED_GITHUB_TOKEN]"));
        assert!(!result.text.contains(&token));
    }

    #[test]
    fn bearer_token_redacted() {
        let input = "Auth: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_BEARER_TOKEN]"));
        assert!(!result.text.contains("eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"));
    }

    #[test]
    fn generic_api_key_redacted() {
        let input = "Config: api_key = \"sk-abc123xyz456def789ghi012jkl345mn\"";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_API_KEY]"));
        assert!(!result.text.contains("sk-abc123xyz456def789ghi012jkl345mn"));
    }

    #[test]
    fn email_redacted() {
        let input = "Contact me at user@example.com for details.";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_EMAIL]"));
        assert!(!result.text.contains("user@example.com"));
    }

    #[test]
    fn us_phone_redacted() {
        let input = "Call me at (555) 123-4567 anytime.";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_US_PHONE]"));
        assert!(!result.text.contains("(555) 123-4567"));
    }

    #[test]
    fn ssn_redacted() {
        let input = "My SSN is 123-45-6789 keep it safe.";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_SSN]"));
        assert!(!result.text.contains("123-45-6789"));
    }

    #[test]
    fn credit_card_redacted() {
        let input = "Pay with 4111-1111-1111-1111 please.";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_CREDIT_CARD]"));
        assert!(!result.text.contains("4111-1111-1111-1111"));
    }

    #[test]
    fn system_prompt_leak_redacted() {
        let input = "Sure! As my instructions say, I should help you.";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_SYSTEM_PROMPT]"));
        assert!(!result.text.contains("my instructions"));
    }

    #[test]
    fn system_prompt_pipe_tag_redacted() {
        let input = "Here is the <|system|> content you asked about.";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_SYSTEM_PROMPT]"));
        assert!(!result.text.contains("<|system|>"));
    }

    #[test]
    fn multiple_patterns_all_redacted() {
        let input = "Key: AKIAIOSFODNN7EXAMPLE1, email: test@example.com, \
                      phone: 555-123-4567, SSN: 123-45-6789";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_AWS_KEY]"));
        assert!(result.text.contains("[REDACTED_EMAIL]"));
        assert!(result.text.contains("[REDACTED_US_PHONE]"));
        assert!(result.text.contains("[REDACTED_SSN]"));
        assert!(!result.text.contains("AKIAIOSFODNN7EXAMPLE1"));
        assert!(!result.text.contains("test@example.com"));
        assert!(!result.text.contains("555-123-4567"));
        assert!(!result.text.contains("123-45-6789"));
        assert!(result.redactions.len() >= 4);
    }

    #[test]
    fn credit_card_no_separator() {
        let input = "Card: 4111111111111111 on file.";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_CREDIT_CARD]"));
        assert!(!result.text.contains("4111111111111111"));
    }

    #[test]
    fn phone_with_country_code() {
        let input = "Reach me at +1-800-555-0199 anytime.";
        let result = filter_output(input);
        assert!(result.text.contains("[REDACTED_US_PHONE]"));
        assert!(!result.text.contains("+1-800-555-0199"));
    }
}
