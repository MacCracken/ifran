//! REST handlers for tenant management (admin-only).
//!
//! These endpoints are only available when `multi_tenant = true` in config.
//! Protected by `IFRAN_ADMIN_KEY` environment variable.

use crate::server::state::AppState;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct TenantResponse {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub created_at: String,
}

#[derive(Serialize)]
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
}
