//! Tauri commands for chat/inference.

use serde_json::Value;

const API_BASE: &str = "http://127.0.0.1:8420";

#[tauri::command]
pub async fn send_message(
    model: String,
    prompt: String,
    system_prompt: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<Value, String> {
    let client = reqwest::Client::new();

    let mut body = serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
        }],
        "max_tokens": max_tokens.unwrap_or(512),
        "temperature": temperature.unwrap_or(0.7),
        "stream": false,
    });

    if let Some(sys) = system_prompt {
        let messages = body["messages"].as_array_mut().unwrap();
        messages.insert(0, serde_json::json!({
            "role": "system",
            "content": sys,
        }));
    }

    let resp = client
        .post(format!("{API_BASE}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Inference failed: {e}"))?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Inference error: {text}"));
    }

    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}
