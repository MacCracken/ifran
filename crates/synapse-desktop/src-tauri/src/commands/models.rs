//! Tauri commands for model management.

use serde_json::Value;

const API_BASE: &str = "http://127.0.0.1:8420";

#[tauri::command]
pub async fn list_models() -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{API_BASE}/models"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch models: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}

#[tauri::command]
pub async fn get_model(id: String) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{API_BASE}/models/{id}"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch model: {e}"))?;
    if resp.status() == 404 {
        return Err("Model not found".into());
    }
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}

#[tauri::command]
pub async fn delete_model(id: String) -> Result<(), String> {
    let client = reqwest::Client::new();
    let resp = client
        .delete(format!("{API_BASE}/models/{id}"))
        .send()
        .await
        .map_err(|e| format!("Failed to delete model: {e}"))?;
    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("Delete failed: {}", resp.status()))
    }
}

#[tauri::command]
pub async fn pull_model(repo_id: String, quant: Option<String>) -> Result<Value, String> {
    // Pull is handled by the CLI; the desktop app shows progress via the API.
    // For now, return a placeholder — real implementation would call the pull API.
    Ok(serde_json::json!({
        "status": "started",
        "repo_id": repo_id,
        "quant": quant,
    }))
}
