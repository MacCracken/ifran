//! Request ID middleware — injects a unique correlation ID per request.
//!
//! If the client sends `X-Request-ID`, it is reused. Otherwise a UUID v4 is generated.
//! The ID is:
//! - Inserted into request extensions for handler access
//! - Returned in the `X-Request-ID` response header

use axum::extract::Request;
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::response::Response;
use uuid::Uuid;

/// Newtype for the request ID, extractable from request extensions.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

/// Middleware that injects a correlation ID into every request/response.
///
/// Reads `X-Request-ID` from the incoming request header (if present and valid),
/// otherwise generates a new UUID v4. The ID is stored in request extensions
/// and echoed back in the response `X-Request-ID` header.
#[inline]
pub async fn inject_request_id(mut req: Request, next: Next) -> Response {
    let id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|s| {
            !s.is_empty()
                && s.len() <= 128
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
        })
        .map(String::from)
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    req.extensions_mut().insert(RequestId(id.clone()));

    let mut response = next.run(req).await;

    if let Ok(val) = HeaderValue::from_str(&id) {
        response.headers_mut().insert("x-request-id", val);
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::middleware as axum_mw;
    use axum::routing::get;
    use tower::ServiceExt;

    /// Helper: build a minimal router with the request_id middleware.
    fn app() -> Router {
        Router::new()
            .route("/ping", get(|| async { "pong" }))
            .layer(axum_mw::from_fn(inject_request_id))
    }

    #[tokio::test]
    async fn response_has_request_id_header() {
        let resp = app()
            .oneshot(
                HttpRequest::builder()
                    .uri("/ping")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().contains_key("x-request-id"));

        // Should be a valid UUID v4
        let val = resp.headers()["x-request-id"].to_str().unwrap();
        assert!(uuid::Uuid::parse_str(val).is_ok());
    }

    #[tokio::test]
    async fn provided_request_id_is_echoed() {
        let resp = app()
            .oneshot(
                HttpRequest::builder()
                    .uri("/ping")
                    .header("x-request-id", "my-custom-id-123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()["x-request-id"].to_str().unwrap(),
            "my-custom-id-123"
        );
    }

    #[tokio::test]
    async fn empty_header_generates_uuid() {
        let resp = app()
            .oneshot(
                HttpRequest::builder()
                    .uri("/ping")
                    .header("x-request-id", "")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let val = resp.headers()["x-request-id"].to_str().unwrap();
        assert!(uuid::Uuid::parse_str(val).is_ok());
    }

    #[tokio::test]
    async fn oversized_header_generates_uuid() {
        let long_id = "x".repeat(200);
        let resp = app()
            .oneshot(
                HttpRequest::builder()
                    .uri("/ping")
                    .header("x-request-id", &long_id)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let val = resp.headers()["x-request-id"].to_str().unwrap();
        assert_ne!(val, long_id);
        assert!(uuid::Uuid::parse_str(val).is_ok());
    }
}
