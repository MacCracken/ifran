//! Tauri commands for RLHF annotation management.

use serde_json::Value;

const API_BASE: &str = "http://127.0.0.1:8420";

#[tauri::command]
pub async fn list_sessions() -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{API_BASE}/rlhf/sessions"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch sessions: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}

#[tauri::command]
pub async fn create_session(name: String, model_name: String) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{API_BASE}/rlhf/sessions"))
        .json(&serde_json::json!({ "name": name, "model_name": model_name }))
        .send()
        .await
        .map_err(|e| format!("Failed to create session: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}

#[tauri::command]
pub async fn get_next_pair(session_id: String) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{API_BASE}/rlhf/sessions/{session_id}/pairs"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch pair: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}

#[tauri::command]
pub async fn submit_annotation(pair_id: String, preference: String) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{API_BASE}/rlhf/pairs/{pair_id}/annotate"))
        .json(&serde_json::json!({ "preference": preference }))
        .send()
        .await
        .map_err(|e| format!("Failed to submit annotation: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}

#[tauri::command]
pub async fn get_session_stats(session_id: String) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{API_BASE}/rlhf/sessions/{session_id}/stats"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch stats: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}

#[tauri::command]
pub async fn export_session(session_id: String) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{API_BASE}/rlhf/sessions/{session_id}/export"))
        .send()
        .await
        .map_err(|e| format!("Failed to export session: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}
