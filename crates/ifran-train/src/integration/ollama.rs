//! Ollama adapter registration for post-training integration.

use ifran_types::IfranError;
use ifran_types::error::Result;

/// Register a LoRA adapter with a local Ollama instance.
///
/// Creates a Modelfile and calls `ollama create` to register the adapter.
pub async fn register_adapter(
    ollama_endpoint: &str,
    model_name: &str,
    base_model: &str,
    adapter_path: &str,
) -> Result<()> {
    tracing::info!(
        model_name,
        base_model,
        adapter_path,
        "Registering LoRA adapter with Ollama"
    );

    let modelfile = build_modelfile(base_model, adapter_path);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| IfranError::BackendError(format!("HTTP client error: {e}")))?;

    let resp = client
        .post(format!("{ollama_endpoint}/api/create"))
        .json(&serde_json::json!({
            "name": model_name,
            "modelfile": modelfile,
        }))
        .send()
        .await
        .map_err(|e| IfranError::BackendError(format!("Ollama create failed: {e}")))?;

    if resp.status().is_success() {
        tracing::info!(model_name, "LoRA adapter registered with Ollama");
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(IfranError::BackendError(format!(
            "Ollama create failed (HTTP {status}): {body}"
        )))
    }
}

/// Build an Ollama Modelfile for a LoRA adapter.
#[must_use]
pub fn build_modelfile(base_model: &str, adapter_path: &str) -> String {
    format!("FROM {base_model}\nADAPTER \"{adapter_path}\"\n")
}

/// Check if Ollama is available at the given endpoint.
#[must_use]
pub async fn check_ollama(endpoint: &str) -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok();

    match client {
        Some(c) => c
            .get(format!("{endpoint}/api/tags"))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success()),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modelfile_format() {
        let mf = build_modelfile("llama3.1", "/adapters/lora-v1");
        assert!(mf.contains("FROM llama3.1"));
        assert!(mf.contains("ADAPTER \"/adapters/lora-v1\""));
    }

    #[tokio::test]
    async fn check_ollama_unavailable() {
        let available = check_ollama("http://127.0.0.1:19999").await;
        assert!(!available);
    }

    #[tokio::test]
    async fn register_adapter_no_ollama() {
        let result =
            register_adapter("http://127.0.0.1:19999", "test", "llama3.1", "/adapters/v1").await;
        assert!(result.is_err());
    }

    #[test]
    fn modelfile_with_spaces_in_path() {
        let mf = build_modelfile("llama3.1", "/path/with spaces/adapter v2");
        assert!(mf.contains("FROM llama3.1"));
        // The adapter path is quoted in the modelfile.
        assert!(mf.contains("/path/with spaces/adapter v2"));
    }
}
