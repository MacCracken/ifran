//! Tauri commands for system status and hardware info.

use serde_json::Value;

const API_BASE: &str = "http://127.0.0.1:8420";

#[tauri::command]
pub async fn get_status() -> Result<Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{API_BASE}/system/status"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch status: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Invalid response: {e}"))
}

#[tauri::command]
pub async fn get_hardware() -> Result<Value, String> {
    // Use local hardware detection instead of API
    match ifran_core::hardware::detect::detect() {
        Ok(hw) => Ok(serde_json::json!({
            "cpu": {
                "model": hw.cpu.model_name,
                "cores": hw.cpu.physical_cores,
                "threads": hw.cpu.logical_cores,
                "memory_total_mb": hw.cpu.total_memory_mb,
                "memory_available_mb": hw.cpu.available_memory_mb,
            },
            "gpus": hw.gpus.iter().map(|g| serde_json::json!({
                "index": g.index,
                "name": g.name,
                "memory_total_mb": g.memory_total_mb,
                "memory_free_mb": g.memory_free_mb,
            })).collect::<Vec<_>>(),
            "has_gpu": hw.has_gpu(),
            "total_gpu_memory_mb": hw.total_gpu_memory_mb(),
        })),
        Err(e) => Err(format!("Hardware detection failed: {e}")),
    }
}
