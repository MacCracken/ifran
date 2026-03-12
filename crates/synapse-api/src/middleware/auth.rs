//! Authentication middleware for API requests.
//!
//! When `SYNAPSE_API_KEY` is set (or configured), all requests except
//! `/health` must include `Authorization: Bearer <key>`. When no key is
//! configured, auth is disabled (open access).

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;

/// Extract the configured API key (from env or config).
///
/// Returns `None` if auth is disabled (no key set).
pub fn configured_api_key() -> Option<String> {
    std::env::var("SYNAPSE_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
}

/// Auth middleware — validates Bearer token if an API key is configured.
pub async fn require_auth(req: Request, next: Next) -> Result<Response, StatusCode> {
    let api_key = match configured_api_key() {
        Some(key) => key,
        None => return Ok(next.run(req).await), // No key configured — open access
    };

    // Skip auth for health endpoint
    if req.uri().path() == "/health" {
        return Ok(next.run(req).await);
    }

    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = header.strip_prefix("Bearer ").unwrap_or("");
            if token == api_key {
                Ok(next.run(req).await)
            } else {
                Err(StatusCode::UNAUTHORIZED)
            }
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn empty_string_is_none() {
        let key: Option<String> = Some(String::new()).filter(|k: &String| !k.is_empty());
        assert!(key.is_none());
    }

    #[test]
    fn non_empty_string_is_some() {
        let key: Option<String> = Some("secret".to_string()).filter(|k: &String| !k.is_empty());
        assert_eq!(key.unwrap(), "secret");
    }
}
