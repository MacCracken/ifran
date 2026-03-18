//! Structured API error responses for consistent client-side error handling.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::Serialize;

/// Structured error body returned to clients.
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

/// A complete error response with HTTP status and structured body.
pub struct ApiErrorResponse {
    pub status: StatusCode,
    pub body: ApiError,
}

impl IntoResponse for ApiErrorResponse {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

impl ApiErrorResponse {
    pub fn not_found(resource: &str, id: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: ApiError {
                code: "NOT_FOUND",
                message: format!("{resource} '{id}' not found"),
                hint: None,
            },
        }
    }

    pub fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ApiError {
                code,
                message: message.into(),
                hint: None,
            },
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ApiError {
                code: "INTERNAL_ERROR",
                message: message.into(),
                hint: None,
            },
        }
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.body.hint = Some(hint.into());
        self
    }
}

/// Convert a SynapseError into an ApiErrorResponse with appropriate status code and error code.
pub fn from_synapse_error(
    err: &synapse_types::SynapseError,
    status: StatusCode,
) -> (StatusCode, String) {
    // For backward compatibility, still return (StatusCode, String) tuples
    // The structured format is available via ApiErrorResponse for new endpoints
    (status, err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_error_serializes() {
        let err = ApiError {
            code: "MODEL_NOT_LOADED",
            message: "No model loaded for inference".into(),
            hint: Some("Load a model first with POST /models/load".into()),
        };
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["code"], "MODEL_NOT_LOADED");
        assert_eq!(json["message"], "No model loaded for inference");
        assert_eq!(json["hint"], "Load a model first with POST /models/load");
    }

    #[test]
    fn api_error_no_hint() {
        let err = ApiError {
            code: "BAD_REQUEST",
            message: "Invalid format".into(),
            hint: None,
        };
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["code"], "BAD_REQUEST");
        assert!(json.get("hint").is_none());
    }

    #[test]
    fn not_found_helper() {
        let resp = ApiErrorResponse::not_found("Training job", "abc-123");
        assert_eq!(resp.status, StatusCode::NOT_FOUND);
        assert_eq!(resp.body.code, "NOT_FOUND");
        assert!(resp.body.message.contains("abc-123"));
    }

    #[test]
    fn bad_request_helper() {
        let resp = ApiErrorResponse::bad_request("INVALID_CONFIG", "learning_rate must be > 0");
        assert_eq!(resp.status, StatusCode::BAD_REQUEST);
        assert_eq!(resp.body.code, "INVALID_CONFIG");
    }

    #[test]
    fn internal_helper() {
        let resp = ApiErrorResponse::internal("database unreachable");
        assert_eq!(resp.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(resp.body.code, "INTERNAL_ERROR");
    }

    #[test]
    fn with_hint_chaining() {
        let resp = ApiErrorResponse::bad_request("NO_MODEL", "No model loaded")
            .with_hint("Try: POST /models with a model name to load one");
        assert_eq!(
            resp.body.hint.unwrap(),
            "Try: POST /models with a model name to load one"
        );
    }
}
