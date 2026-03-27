//! Ollama backend integration.
//!
//! HTTP client to a running Ollama server. Delegates model management and
//! inference to Ollama's REST API, making any Ollama-hosted model available
//! through Ifran's unified interface.

use async_trait::async_trait;
use ifran_types::IfranError;
use ifran_types::backend::{
    AcceleratorType, BackendCapabilities, BackendId, BackendLocality, DeviceConfig,
};
use ifran_types::error::Result;
use ifran_types::inference::{
    FinishReason, InferenceRequest, InferenceResponse, StreamChunk, TokenUsage,
};
use ifran_types::model::{ModelFormat, ModelManifest};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn};

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
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
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
            accelerators: vec![
                AcceleratorType::Cpu,
                AcceleratorType::Cuda,
                AcceleratorType::Rocm,
            ],
            max_context_length: Some(131072),
            supports_streaming: true,
            supports_embeddings: true,
            supports_vision: true,
            locality: BackendLocality::Local,
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

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                IfranError::BackendError(format!("Failed to load model in Ollama: {e}"))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(IfranError::BackendError(format!(
                "Ollama failed to load model (HTTP {status}): {text}"
            )));
        }

        let handle_id = format!("ollama-{}", model_name.replace('/', "-"));
        info!(handle = %handle_id, model = %model_name, "Loaded model in Ollama");

        self.loaded
            .write()
            .await
            .insert(handle_id.clone(), model_name.to_string());
        Ok(ModelHandle(handle_id))
    }

    async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        let mut loaded = self.loaded.write().await;
        let model_name = loaded
            .remove(&handle.0)
            .ok_or_else(|| IfranError::ModelNotFound(handle.0.clone()))?;

        // Unload by setting keep_alive to 0
        let url = format!("{}/api/generate", self.base_url);
        let body = serde_json::json!({
            "model": model_name,
            "prompt": "",
            "keep_alive": 0,
        });

        if let Err(e) = self.client.post(&url).json(&body).send().await {
            warn!(handle = %handle.0, error = %e, "Failed to send unload request to Ollama");
        }
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
            .ok_or_else(|| IfranError::ModelNotFound(handle.0.clone()))?;

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
            .map_err(|e| IfranError::BackendError(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(IfranError::BackendError(format!("Ollama error: {text}")));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| IfranError::BackendError(e.to_string()))?;

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
            .ok_or_else(|| IfranError::ModelNotFound(handle.0.clone()))?
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
            .map_err(|e| IfranError::BackendError(e.to_string()))?;

        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            use futures::StreamExt;
            const MAX_BUFFER_SIZE: usize = 1024 * 1024; // 1 MB
            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = stream.next().await {
                if tx.is_closed() {
                    break;
                }
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("Ollama stream error: {e}");
                        break;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                if buffer.len() > MAX_BUFFER_SIZE {
                    tracing::warn!(
                        "Ollama stream buffer exceeded {MAX_BUFFER_SIZE} bytes, aborting"
                    );
                    break;
                }

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

    #[test]
    fn capabilities() {
        let backend = OllamaBackend::new(None);
        let caps = backend.capabilities();
        assert!(caps.supports_streaming);
        assert!(caps.supports_embeddings);
        assert!(caps.supports_vision);
        assert_eq!(caps.max_context_length, Some(131072));
        assert!(caps.accelerators.contains(&AcceleratorType::Cpu));
        assert!(caps.accelerators.contains(&AcceleratorType::Cuda));
        assert!(caps.accelerators.contains(&AcceleratorType::Rocm));
    }

    #[test]
    fn supported_formats_gguf() {
        let backend = OllamaBackend::new(None);
        assert_eq!(backend.supported_formats(), &[ModelFormat::Gguf]);
    }

    #[test]
    fn build_messages_user_only() {
        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };
        let msgs = build_ollama_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Hello");
    }

    #[test]
    fn build_messages_with_system() {
        let req = InferenceRequest {
            prompt: "Hi".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: Some("Be concise.".into()),
            sensitivity: None,
        };
        let msgs = build_ollama_messages(&req);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "Be concise.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[tokio::test]
    async fn load_model_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/generate")
            .with_status(200)
            .with_body("{}")
            .create_async()
            .await;

        let backend = OllamaBackend::new(Some(server.url()));
        let manifest = ModelManifest {
            info: ifran_types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test-model".into(),
                repo_id: Some("org/test-model".into()),
                format: ModelFormat::Gguf,
                quant: ifran_types::model::QuantLevel::Q4KM,
                size_bytes: 1000,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/model.gguf".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            },
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
        };
        let device = DeviceConfig {
            accelerator: AcceleratorType::Cpu,
            device_ids: vec![],
            memory_limit_mb: None,
        };

        let handle = backend.load_model(&manifest, &device).await.unwrap();
        assert_eq!(handle.0, "ollama-org-test-model");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn unload_model_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/generate")
            .with_status(200)
            .with_body("{}")
            .create_async()
            .await;

        let backend = OllamaBackend::new(Some(server.url()));
        backend
            .loaded
            .write()
            .await
            .insert("ollama-test".into(), "test".into());

        backend
            .unload_model(ModelHandle("ollama-test".into()))
            .await
            .unwrap();

        assert!(backend.loaded.read().await.is_empty());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn unload_model_not_found() {
        let backend = OllamaBackend::new(None);
        let result = backend
            .unload_model(ModelHandle("nonexistent".into()))
            .await;
        assert!(matches!(result, Err(IfranError::ModelNotFound(_))));
    }

    #[tokio::test]
    async fn infer_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "message": {"content": "Hello! How can I help?"},
                    "prompt_eval_count": 10,
                    "eval_count": 5
                }"#,
            )
            .create_async()
            .await;

        let backend = OllamaBackend::new(Some(server.url()));
        backend
            .loaded
            .write()
            .await
            .insert("ollama-test".into(), "test-model".into());

        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: Some(100),
            temperature: Some(0.5),
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };

        let resp = backend
            .infer(&ModelHandle("ollama-test".into()), &req)
            .await
            .unwrap();
        assert_eq!(resp.text, "Hello! How can I help?");
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
        assert_eq!(resp.usage.total_tokens, 15);
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn infer_model_not_loaded() {
        let backend = OllamaBackend::new(None);
        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };
        let result = backend
            .infer(&ModelHandle("nonexistent".into()), &req)
            .await;
        assert!(matches!(result, Err(IfranError::ModelNotFound(_))));
    }

    #[tokio::test]
    async fn infer_server_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .with_status(500)
            .with_body("Internal Server Error")
            .create_async()
            .await;

        let backend = OllamaBackend::new(Some(server.url()));
        backend
            .loaded
            .write()
            .await
            .insert("ollama-test".into(), "model".into());

        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };
        let result = backend
            .infer(&ModelHandle("ollama-test".into()), &req)
            .await;
        assert!(matches!(result, Err(IfranError::BackendError(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn health_check_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/api/tags")
            .with_status(200)
            .with_body(r#"{"models":[]}"#)
            .create_async()
            .await;

        let backend = OllamaBackend::new(Some(server.url()));
        let healthy = backend.health_check().await.unwrap();
        assert!(healthy);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn health_check_server_down() {
        let backend = OllamaBackend::new(Some("http://127.0.0.1:1".into()));
        let healthy = backend.health_check().await.unwrap();
        assert!(!healthy);
    }
}
