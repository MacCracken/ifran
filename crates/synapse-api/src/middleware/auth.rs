//! Authentication middleware for API requests.
//!
//! Supports two modes:
//! - **Single-tenant** (default): `SYNAPSE_API_KEY` env var for auth. When no key
//!   is configured, auth is disabled (open access). All requests get `TenantId::default_tenant()`.
//! - **Multi-tenant**: API keys are looked up in the `TenantStore`. Each key maps
//!   to a tenant. Disabled tenants are rejected.

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use synapse_types::TenantId;

use crate::state::AppState;

/// Extract the configured API key (from env or config).
///
/// Returns `None` if auth is disabled (no key set).
pub fn configured_api_key() -> Option<String> {
    std::env::var("SYNAPSE_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
}

/// Auth middleware — validates Bearer token and injects TenantId into request extensions.
///
/// In single-tenant mode, validates against `SYNAPSE_API_KEY` and injects `TenantId::default_tenant()`.
/// In multi-tenant mode, looks up the key in the TenantStore and injects the resolved tenant.
pub async fn require_auth(
    axum::extract::State(state): axum::extract::State<AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip auth for health endpoint
    if req.uri().path() == "/health" {
        req.extensions_mut().insert(TenantId::default_tenant());
        return Ok(next.run(req).await);
    }

    if state.config.security.multi_tenant {
        // Multi-tenant mode: resolve tenant from API key via TenantStore
        let tenant_store = state
            .tenant_store
            .as_ref()
            .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

        let token = extract_bearer_token(&req).ok_or(StatusCode::UNAUTHORIZED)?;

        let store = tenant_store.lock().await;
        let record = store
            .resolve_by_key(token)
            .map_err(|_| StatusCode::UNAUTHORIZED)?;

        if !record.enabled {
            return Err(StatusCode::FORBIDDEN);
        }

        req.extensions_mut().insert(record.id);
        Ok(next.run(req).await)
    } else {
        // Single-tenant mode: legacy SYNAPSE_API_KEY behavior
        let api_key = match configured_api_key() {
            Some(key) => key,
            None => {
                // No key configured — open access, default tenant
                req.extensions_mut().insert(TenantId::default_tenant());
                return Ok(next.run(req).await);
            }
        };

        let token = extract_bearer_token(&req).ok_or(StatusCode::UNAUTHORIZED)?;

        if token == api_key {
            req.extensions_mut().insert(TenantId::default_tenant());
            Ok(next.run(req).await)
        } else {
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

/// Extract the Bearer token from the Authorization header.
fn extract_bearer_token(req: &Request) -> Option<&str> {
    req.headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn extract_bearer_from_header() {
        let req = axum::http::Request::builder()
            .header("authorization", "Bearer my-token")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(extract_bearer_token(&req), Some("my-token"));
    }

    #[test]
    fn extract_bearer_missing_header() {
        let req = axum::http::Request::builder()
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(extract_bearer_token(&req), None);
    }

    #[test]
    fn extract_bearer_wrong_scheme() {
        let req = axum::http::Request::builder()
            .header("authorization", "Basic abc123")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(extract_bearer_token(&req), None);
    }
}
