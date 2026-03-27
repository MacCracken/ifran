//! REST handlers for RLHF annotation management.

use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use ifran_types::rlhf::{AnnotationPair, Preference};
use serde::Deserialize;
use uuid::Uuid;

use ifran_types::TenantId;

use super::pagination::{PaginatedResponse, PaginationQuery};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct CreateSessionRequest {
    pub name: String,
    pub model_name: String,
}

#[derive(Deserialize)]
pub struct AddPairsRequest {
    pub pairs: Vec<PairInput>,
}

#[derive(Deserialize)]
pub struct PairInput {
    pub prompt: String,
    pub response_a: String,
    pub response_b: String,
}

#[derive(Deserialize)]
pub struct AnnotateRequest {
    pub preference: Preference,
}

/// POST /rlhf/sessions
pub async fn create_session(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let name = req.name.trim();
    if name.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Session name must not be empty".into(),
        ));
    }
    if name.len() > 256 {
        return Err((
            StatusCode::BAD_REQUEST,
            "Session name must be 256 characters or fewer".into(),
        ));
    }
    if req.model_name.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Model name must not be empty".into(),
        ));
    }

    let store = state.annotation_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Annotation store not initialized".into(),
    ))?;

    let session = store
        .create_session(&req.name, &req.model_name, &tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "id": session.id.to_string(),
            "name": session.name,
            "model_name": session.model_name,
            "status": "active",
        })),
    ))
}

/// GET /rlhf/sessions
pub async fn list_sessions(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Query(page): Query<PaginationQuery>,
) -> Result<Json<PaginatedResponse<serde_json::Value>>, (StatusCode, String)> {
    let store = state.annotation_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Annotation store not initialized".into(),
    ))?;

    let safe_limit = page.safe_limit();
    let paged = store
        .list_sessions(&tenant_id, safe_limit, page.offset)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let data: Vec<serde_json::Value> = paged
        .items
        .iter()
        .map(|sess| {
            serde_json::json!({
                "id": sess.id.to_string(),
                "name": sess.name,
                "model_name": sess.model_name,
                "status": serde_json::to_value(sess.status).unwrap_or(serde_json::Value::Null),
                "created_at": sess.created_at.to_rfc3339(),
            })
        })
        .collect();

    Ok(Json(PaginatedResponse::pre_sliced(
        data,
        paged.total,
        safe_limit,
        page.offset,
    )))
}

/// GET /rlhf/sessions/{id}
pub async fn get_session(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.annotation_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Annotation store not initialized".into(),
    ))?;

    let session = store
        .get_session(id, &tenant_id)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let stats = store
        .get_stats(id, &tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "id": session.id.to_string(),
        "name": session.name,
        "model_name": session.model_name,
        "status": serde_json::to_value(session.status).unwrap_or(serde_json::Value::Null),
        "created_at": session.created_at.to_rfc3339(),
        "stats": {
            "total_pairs": stats.total_pairs,
            "annotated_count": stats.annotated_count,
            "remaining": stats.remaining,
        },
    })))
}

/// POST /rlhf/sessions/{id}/pairs
pub async fn add_pairs(
    State(state): State<AppState>,
    Extension(_tenant_id): Extension<TenantId>,
    Path(id): Path<Uuid>,
    Json(req): Json<AddPairsRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if req.pairs.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Pairs list must not be empty".into(),
        ));
    }
    if req.pairs.len() > 1000 {
        return Err((
            StatusCode::BAD_REQUEST,
            "Pairs list must not exceed 1000 entries".into(),
        ));
    }
    for (i, pair) in req.pairs.iter().enumerate() {
        if pair.prompt.trim().is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Pair {i}: prompt must not be empty"),
            ));
        }
        if pair.response_a.trim().is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Pair {i}: response_a must not be empty"),
            ));
        }
        if pair.response_b.trim().is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Pair {i}: response_b must not be empty"),
            ));
        }
    }

    let store = state.annotation_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Annotation store not initialized".into(),
    ))?;

    let pairs: Vec<AnnotationPair> = req
        .pairs
        .into_iter()
        .map(|p| AnnotationPair {
            id: Uuid::new_v4(),
            session_id: id,
            prompt: p.prompt,
            response_a: p.response_a,
            response_b: p.response_b,
            preference: None,
            annotated_at: None,
        })
        .collect();

    let count = pairs.len();
    store
        .add_pairs(&pairs)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "added": count })),
    ))
}

/// GET /rlhf/sessions/{id}/pairs
pub async fn get_pairs(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.annotation_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Annotation store not initialized".into(),
    ))?;

    let next = store
        .get_next_unannotated(id, &tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    match next {
        Some(pair) => Ok(Json(serde_json::json!({
            "id": pair.id.to_string(),
            "prompt": pair.prompt,
            "response_a": pair.response_a,
            "response_b": pair.response_b,
        }))),
        None => Ok(Json(serde_json::json!({ "done": true }))),
    }
}

/// POST /rlhf/pairs/{id}/annotate
pub async fn annotate(
    State(state): State<AppState>,
    Extension(_tenant_id): Extension<TenantId>,
    Path(id): Path<Uuid>,
    Json(req): Json<AnnotateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.annotation_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Annotation store not initialized".into(),
    ))?;

    store
        .annotate_pair(id, req.preference)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(
        serde_json::json!({ "id": id.to_string(), "status": "annotated" }),
    ))
}

/// POST /rlhf/sessions/{id}/export
pub async fn export_session(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.annotation_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Annotation store not initialized".into(),
    ))?;

    let pairs = store
        .export_session(id, &tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // JSONL format for DPO training
    let jsonl: Vec<serde_json::Value> = pairs
        .iter()
        .filter_map(|p| {
            p.preference.map(|pref| {
                let (chosen, rejected) = match pref {
                    Preference::ResponseA => (&p.response_a, &p.response_b),
                    Preference::ResponseB => (&p.response_b, &p.response_a),
                    Preference::Tie | Preference::BothBad => (&p.response_a, &p.response_b),
                    _ => (&p.response_a, &p.response_b),
                };
                serde_json::json!({
                    "prompt": p.prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "preference": serde_json::to_value(pref).unwrap_or(serde_json::Value::Null),
                })
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "session_id": id.to_string(),
        "format": "dpo_jsonl",
        "count": jsonl.len(),
        "data": jsonl,
    })))
}

/// GET /rlhf/sessions/{id}/stats
pub async fn get_stats(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.annotation_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Annotation store not initialized".into(),
    ))?;

    let stats = store
        .get_stats(id, &tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "session_id": id.to_string(),
        "total_pairs": stats.total_pairs,
        "annotated_count": stats.annotated_count,
        "remaining": stats.remaining,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_session_request_deserialize() {
        let json = r#"{"name": "test", "model_name": "llama-8b"}"#;
        let req: CreateSessionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "test");
        assert_eq!(req.model_name, "llama-8b");
    }

    #[test]
    fn add_pairs_request_deserialize() {
        let json = r#"{"pairs": [{"prompt": "q", "response_a": "a", "response_b": "b"}]}"#;
        let req: AddPairsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.pairs.len(), 1);
        assert_eq!(req.pairs[0].prompt, "q");
    }

    #[test]
    fn annotate_request_deserialize() {
        let json = r#"{"preference": "response_a"}"#;
        let req: AnnotateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.preference, Preference::ResponseA);
    }

    #[test]
    fn annotate_request_all_preferences() {
        for (s, expected) in [
            ("response_a", Preference::ResponseA),
            ("response_b", Preference::ResponseB),
            ("tie", Preference::Tie),
            ("both_bad", Preference::BothBad),
        ] {
            let json = format!(r#"{{"preference": "{s}"}}"#);
            let req: AnnotateRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(req.preference, expected);
        }
    }

    #[test]
    fn pair_input_deserialize() {
        let json =
            r#"{"prompt": "What is Rust?", "response_a": "a lang", "response_b": "systems lang"}"#;
        let input: PairInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.prompt, "What is Rust?");
    }
}
