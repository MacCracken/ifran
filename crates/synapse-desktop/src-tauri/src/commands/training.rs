//! Tauri commands for training job management.

use serde_json::Value;

const API_BASE: &str = "http://127.0.0.1:8420";

#[tauri::command]
pub async fn list_jobs() -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{API_BASE}/training/jobs"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch jobs: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}

#[tauri::command]
pub async fn create_job(config: Value) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{API_BASE}/training/jobs"))
        .json(&config)
        .send()
        .await
        .map_err(|e| format!("Failed to create job: {e}"))?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Create job failed: {text}"));
    }

    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}

#[tauri::command]
pub async fn cancel_job(id: String) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{API_BASE}/training/jobs/{id}/cancel"))
        .send()
        .await
        .map_err(|e| format!("Failed to cancel job: {e}"))?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Cancel failed: {text}"));
    }

    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}
