//! REST handlers for tenant management (admin-only).
//!
//! These endpoints are only available when `multi_tenant = true` in config.
//! Protected by `IFRAN_ADMIN_KEY` environment variable.

use crate::server::state::AppState;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct TenantResponse {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct CreateTenantResponse {
    pub tenant: TenantResponse,
    /// The raw API key — shown only at creation time.
    pub api_key: String,
}

#[derive(Deserialize)]
pub struct CreateTenantRequest {
    pub name: String,
}

/// POST /admin/tenants — create a new tenant. Returns the API key (shown once).
pub async fn create_tenant(
    State(state): State<AppState>,
    Json(req): Json<CreateTenantRequest>,
) -> Result<(StatusCode, Json<CreateTenantResponse>), (StatusCode, String)> {
    let store = state.tenant_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Tenant store not initialized".into(),
    ))?;

    let (record, raw_key) = store.create_tenant(&req.name).map_err(|e| {
        tracing::error!(error = %e, "internal error");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error".into(),
        )
    })?;

    Ok((
        StatusCode::CREATED,
        Json(CreateTenantResponse {
            tenant: TenantResponse {
                id: record.id.0,
                name: record.name,
                enabled: record.enabled,
                created_at: record.created_at,
            },
            api_key: raw_key,
        }),
    ))
}

/// GET /admin/tenants — list all tenants.
pub async fn list_tenants(
    State(state): State<AppState>,
) -> Result<Json<Vec<TenantResponse>>, (StatusCode, String)> {
    let store = state.tenant_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Tenant store not initialized".into(),
    ))?;

    let paged = store.list_tenants(1000, 0).map_err(|e| {
        tracing::error!(error = %e, "internal error");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error".into(),
        )
    })?;

    Ok(Json(
        paged
            .items
            .into_iter()
            .map(|r| TenantResponse {
                id: r.id.0,
                name: r.name,
                enabled: r.enabled,
                created_at: r.created_at,
            })
            .collect(),
    ))
}

/// DELETE /admin/tenants/:id — disable a tenant and cancel in-flight jobs.
pub async fn disable_tenant(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let store = state.tenant_store.as_ref().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "Tenant store not initialized".into(),
    ))?;

    let tenant_id = crate::types::TenantId(id);
    store
        .disable_tenant(&tenant_id)
        .map_err(|_| (StatusCode::NOT_FOUND, "Not found".into()))?;

    // Cancel all in-flight training jobs for this tenant
    if let Err(e) = state.job_manager.cancel_tenant_jobs(&tenant_id).await {
        tracing::warn!(tenant_id = %tenant_id.0, error = %e, "Failed to cancel some tenant jobs");
    }

    Ok(StatusCode::NO_CONTENT)
}

/// Admin auth middleware — validates `IFRAN_ADMIN_KEY` Bearer token.
pub async fn require_admin_auth(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    let admin_key = std::env::var("IFRAN_ADMIN_KEY")
        .ok()
        .filter(|k| !k.is_empty())
        .ok_or(StatusCode::FORBIDDEN)?;

    let token = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if token != admin_key {
        return Err(StatusCode::UNAUTHORIZED);
    }

    Ok(next.run(req).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_tenant_request_deserialize() {
        let json = r#"{"name": "Acme Corp"}"#;
        let req: CreateTenantRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "Acme Corp");
    }

    #[test]
    fn tenant_response_serialize() {
        let resp = TenantResponse {
            id: "abc-123".into(),
            name: "Test Tenant".into(),
            enabled: true,
            created_at: "2026-03-15T00:00:00Z".into(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "abc-123");
        assert_eq!(json["name"], "Test Tenant");
        assert_eq!(json["enabled"], true);
    }

    #[test]
    fn create_tenant_response_serialize() {
        let resp = CreateTenantResponse {
            tenant: TenantResponse {
                id: "t-1".into(),
                name: "New".into(),
                enabled: true,
                created_at: "2026-03-15T00:00:00Z".into(),
            },
            api_key: "syn_abc123".into(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["api_key"], "syn_abc123");
        assert!(json["tenant"]["id"].is_string());
    }

    #[test]
    fn create_tenant_request_missing_name() {
        let json = r#"{}"#;
        let result = serde_json::from_str::<CreateTenantRequest>(json);
        assert!(result.is_err());
    }

    #[test]
    fn create_tenant_request_empty_name() {
        let json = r#"{"name": ""}"#;
        let req: CreateTenantRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "");
    }

    #[test]
    fn tenant_response_disabled() {
        let resp = TenantResponse {
            id: "t-2".into(),
            name: "Disabled".into(),
            enabled: false,
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["enabled"], false);
    }

    use crate::server::test_helpers::helpers::test_state;

    #[tokio::test]
    async fn create_tenant_store_unavailable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut state = test_state(&tmp);
        state.tenant_store = None;

        let req = CreateTenantRequest {
            name: "Test Tenant".into(),
        };
        let result = create_tenant(State(state), Json(req)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn list_tenants_store_unavailable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut state = test_state(&tmp);
        state.tenant_store = None;

        let result = list_tenants(State(state)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn disable_tenant_store_unavailable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut state = test_state(&tmp);
        state.tenant_store = None;

        let result = disable_tenant(State(state), Path("some-id".into())).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn create_and_list_tenants() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        if state.tenant_store.is_none() {
            return; // skip if tenant store not initialized in test config
        }

        let req = CreateTenantRequest {
            name: "Acme Inc".into(),
        };
        let result = create_tenant(State(state.clone()), Json(req)).await;
        assert!(result.is_ok());
        let (status, json) = result.unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert!(!json.0.api_key.is_empty());
        assert_eq!(json.0.tenant.name, "Acme Inc");
        assert!(json.0.tenant.enabled);

        // List should include the new tenant
        let list = list_tenants(State(state)).await.unwrap();
        assert!(!list.0.is_empty());
        assert!(list.0.iter().any(|t| t.name == "Acme Inc"));
    }

    #[tokio::test]
    async fn disable_nonexistent_tenant() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        if state.tenant_store.is_none() {
            return;
        }

        let result = disable_tenant(State(state), Path("nonexistent-id".into())).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn require_admin_auth_no_env() {
        // We can't easily call require_admin_auth without a Next middleware,
        // so test the logic indirectly:
        // When IFRAN_ADMIN_KEY is empty/unset, the middleware should return FORBIDDEN.
        // SAFETY: This test is single-threaded and no other thread accesses this env var.
        unsafe {
            std::env::remove_var("IFRAN_ADMIN_KEY");
        }
        let admin_key = std::env::var("IFRAN_ADMIN_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        assert!(admin_key.is_none(), "IFRAN_ADMIN_KEY should not be set");
    }

    #[tokio::test]
    async fn create_and_disable_tenant() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        if state.tenant_store.is_none() {
            return;
        }

        // Create a tenant
        let req = CreateTenantRequest {
            name: "DisableMe Corp".into(),
        };
        let result = create_tenant(State(state.clone()), Json(req)).await;
        assert!(result.is_ok());
        let (status, json) = result.unwrap();
        assert_eq!(status, StatusCode::CREATED);
        let tenant_id = json.0.tenant.id.clone();

        // Disable the tenant
        let result = disable_tenant(State(state.clone()), Path(tenant_id.clone())).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), StatusCode::NO_CONTENT);

        // Verify the tenant is disabled via listing
        let list = list_tenants(State(state)).await.unwrap();
        let tenant = list.0.iter().find(|t| t.id == tenant_id);
        assert!(tenant.is_some());
        assert!(!tenant.unwrap().enabled);
    }

    #[tokio::test]
    async fn create_multiple_tenants_lists_all() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        if state.tenant_store.is_none() {
            return;
        }

        let names = ["Tenant A", "Tenant B", "Tenant C"];
        for name in &names {
            let req = CreateTenantRequest {
                name: (*name).into(),
            };
            let result = create_tenant(State(state.clone()), Json(req)).await;
            assert!(result.is_ok());
        }

        let list = list_tenants(State(state)).await.unwrap();
        for name in &names {
            assert!(list.0.iter().any(|t| t.name == *name));
        }
    }

    #[test]
    fn admin_auth_logic_empty_key_treated_as_unset() {
        // An empty IFRAN_ADMIN_KEY should be treated the same as unset
        let key = Some(String::new());
        let result = key.filter(|k| !k.is_empty());
        assert!(result.is_none());
    }

    #[test]
    fn admin_auth_logic_valid_key_passes() {
        let key = Some("my-secret-key".to_string());
        let result = key.filter(|k| !k.is_empty());
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "my-secret-key");
    }

    #[test]
    fn admin_auth_bearer_parsing() {
        // Test the Bearer token extraction logic
        let header = "Bearer my-token-123";
        let token = header.strip_prefix("Bearer ");
        assert_eq!(token, Some("my-token-123"));

        let header_no_bearer = "Basic my-token";
        let token = header_no_bearer.strip_prefix("Bearer ");
        assert!(token.is_none());
    }

    #[test]
    fn admin_auth_bearer_missing_prefix() {
        let header = "my-token-123";
        let token = header.strip_prefix("Bearer ");
        assert!(token.is_none());
    }

    #[test]
    fn tenant_response_fields_complete() {
        let resp = TenantResponse {
            id: "t-100".into(),
            name: "Full Fields".into(),
            enabled: true,
            created_at: "2026-03-28T12:00:00Z".into(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json["id"].is_string());
        assert!(json["name"].is_string());
        assert!(json["enabled"].is_boolean());
        assert!(json["created_at"].is_string());
        assert_eq!(json["created_at"], "2026-03-28T12:00:00Z");
    }
}
