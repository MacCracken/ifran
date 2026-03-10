//! Ollama backend integration.
//!
//! HTTP client to a running Ollama server. Delegates model management and
//! inference to Ollama's REST API, making any Ollama-hosted model available
//! through Synapse's unified interface.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use synapse_types::backend::{AcceleratorType, BackendCapabilities, BackendId, DeviceConfig};
use synapse_types::error::Result;
use synapse_types::inference::{
    FinishReason, InferenceRequest, InferenceResponse, StreamChunk, TokenUsage,
};
use synapse_types::model::{ModelFormat, ModelManifest};
use synapse_types::SynapseError;
use tokio::sync::{mpsc, RwLock};
use tracing::info;

use crate::traits::{InferenceBackend, ModelHandle};

/// Ollama backend that proxies to a running Ollama server.
pub struct OllamaBackend {
    /// Base URL of the Ollama server.
    base_url: String,
    /// HTTP client.
    client: reqwest::Client,
    /// Loaded models (handle → model name).
    loaded: Arc<RwLock<HashMap<String, String>>>,
}

impl OllamaBackend {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            base_url: base_url.unwrap_or_else(|| "http://127.0.0.1:11434".into()),
            client: reqwest::Client::new(),
            loaded: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl InferenceBackend for OllamaBackend {
    fn id(&self) -> BackendId {
        BackendId("ollama".into())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            accelerators: vec![AcceleratorType::Cpu, AcceleratorType::Cuda, AcceleratorType::Rocm],
            max_context_length: Some(131072),
            supports_streaming: true,
            supports_embeddings: true,
            supports_vision: true,
        }
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        // Ollama manages its own model format internally (GGUF-based)
        &[ModelFormat::Gguf]
    }

    async fn load_model(
        &self,
        manifest: &ModelManifest,
        _device: &DeviceConfig,
    ) -> Result<ModelHandle> {
        let model_name = manifest
            .info
            .repo_id
            .as_deref()
            .unwrap_or(&manifest.info.name);

        // Tell Ollama to load the model by sending a generate request with keep_alive
        let url = format!("{}/api/generate", self.base_url);
        let body = serde_json::json!({
            "model": model_name,
            "prompt": "",
            "keep_alive": "10m",
        });

        self.client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| SynapseError::BackendError(format!("Failed to load model in Ollama: {e}")))?;

        let handle_id = format!("ollama-{}", model_name.replace('/', "-"));
        info!(handle = %handle_id, model = %model_name, "Loaded model in Ollama");

        self.loaded.write().await.insert(handle_id.clone(), model_name.to_string());
        Ok(ModelHandle(handle_id))
    }

    async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        let mut loaded = self.loaded.write().await;
        let model_name = loaded
            .remove(&handle.0)
            .ok_or_else(|| SynapseError::ModelNotFound(handle.0.clone()))?;

        // Unload by setting keep_alive to 0
        let url = format!("{}/api/generate", self.base_url);
        let body = serde_json::json!({
            "model": model_name,
            "prompt": "",
            "keep_alive": 0,
        });

        let _ = self.client.post(&url).json(&body).send().await;
        info!(handle = %handle.0, "Unloaded model from Ollama");
        Ok(())
    }

    async fn infer(
        &self,
        handle: &ModelHandle,
        req: &InferenceRequest,
    ) -> Result<InferenceResponse> {
        let loaded = self.loaded.read().await;
        let model_name = loaded
            .get(&handle.0)
            .ok_or_else(|| SynapseError::ModelNotFound(handle.0.clone()))?;

        let url = format!("{}/api/chat", self.base_url);
        let messages = build_ollama_messages(req);
        let body = serde_json::json!({
            "model": model_name,
            "messages": messages,
            "stream": false,
            "options": {
                "temperature": req.temperature.unwrap_or(0.7),
                "top_p": req.top_p.unwrap_or(0.9),
                "num_predict": req.max_tokens.unwrap_or(512),
            },
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| SynapseError::BackendError(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(SynapseError::BackendError(format!("Ollama error: {text}")));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| SynapseError::BackendError(e.to_string()))?;

        let text = json["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let prompt_tokens = json["prompt_eval_count"].as_u64().unwrap_or(0) as u32;
        let completion_tokens = json["eval_count"].as_u64().unwrap_or(0) as u32;

        Ok(InferenceResponse {
            text,
            usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            finish_reason: FinishReason::Stop,
        })
    }

    async fn infer_stream(
        &self,
        handle: &ModelHandle,
        req: InferenceRequest,
    ) -> Result<mpsc::Receiver<StreamChunk>> {
        let loaded = self.loaded.read().await;
        let model_name = loaded
            .get(&handle.0)
            .ok_or_else(|| SynapseError::ModelNotFound(handle.0.clone()))?
            .clone();

        let url = format!("{}/api/chat", self.base_url);
        let messages = build_ollama_messages(&req);
        let body = serde_json::json!({
            "model": model_name,
            "messages": messages,
            "stream": true,
            "options": {
                "temperature": req.temperature.unwrap_or(0.7),
                "top_p": req.top_p.unwrap_or(0.9),
                "num_predict": req.max_tokens.unwrap_or(512),
            },
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| SynapseError::BackendError(e.to_string()))?;

        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            use futures::StreamExt;
            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(_) => break,
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                        let done = json["done"].as_bool().unwrap_or(false);
                        let text = json["message"]["content"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();

                        if !text.is_empty() || done {
                            let _ = tx
                                .send(StreamChunk {
                                    text,
                                    done,
                                    usage: None,
                                })
                                .await;
                        }

                        if done {
                            return;
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

fn build_ollama_messages(req: &InferenceRequest) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();
    if let Some(ref system) = req.system_prompt {
        messages.push(serde_json::json!({
            "role": "system",
            "content": system,
        }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": &req.prompt,
    }));
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id() {
        let backend = OllamaBackend::new(None);
        assert_eq!(backend.id().0, "ollama");
    }

    #[test]
    fn default_url() {
        let backend = OllamaBackend::new(None);
        assert_eq!(backend.base_url, "http://127.0.0.1:11434");
    }

    #[test]
    fn custom_url() {
        let backend = OllamaBackend::new(Some("http://remote:11434".into()));
        assert_eq!(backend.base_url, "http://remote:11434");
    }
}
