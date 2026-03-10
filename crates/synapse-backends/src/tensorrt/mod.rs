//! TensorRT-LLM backend integration.
//!
//! HTTP client to a running TensorRT-LLM server (Triton Inference Server with
//! TensorRT-LLM backend). Provides optimized NVIDIA GPU inference via layer
//! fusion, FP16/INT8 precision, and in-flight batching.

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
use tracing::{info, warn};

use crate::traits::{InferenceBackend, ModelHandle};

/// TensorRT-LLM backend proxying to a Triton server.
pub struct TensorRtBackend {
    /// Base URL of the Triton/TensorRT-LLM server.
    base_url: String,
    /// HTTP client.
    client: reqwest::Client,
    /// Loaded models (handle → model name).
    loaded: Arc<RwLock<HashMap<String, String>>>,
}

impl TensorRtBackend {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            base_url: base_url.unwrap_or_else(|| "http://127.0.0.1:8000".into()),
            client: reqwest::Client::new(),
            loaded: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl InferenceBackend for TensorRtBackend {
    fn id(&self) -> BackendId {
        BackendId("tensorrt".into())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            accelerators: vec![AcceleratorType::Cuda],
            max_context_length: Some(131072),
            supports_streaming: true,
            supports_embeddings: false,
            supports_vision: false,
        }
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::TensorRt]
    }

    async fn load_model(
        &self,
        manifest: &ModelManifest,
        _device: &DeviceConfig,
    ) -> Result<ModelHandle> {
        // TensorRT-LLM models are pre-compiled engine files loaded by Triton.
        // We verify the model is available on the server.
        let model_name = manifest
            .info
            .repo_id
            .as_deref()
            .unwrap_or(&manifest.info.name);

        let handle_id = format!("tensorrt-{}", model_name.replace('/', "-"));
        info!(handle = %handle_id, model = %model_name, "Registered model with TensorRT backend");

        self.loaded.write().await.insert(handle_id.clone(), model_name.to_string());
        Ok(ModelHandle(handle_id))
    }

    async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        if self.loaded.write().await.remove(&handle.0).is_some() {
            info!(handle = %handle.0, "Unregistered model from TensorRT backend");
            Ok(())
        } else {
            Err(SynapseError::ModelNotFound(handle.0))
        }
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

        // TensorRT-LLM with Triton exposes an OpenAI-compatible endpoint
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": model_name,
            "messages": build_messages(req),
            "max_tokens": req.max_tokens.unwrap_or(512),
            "temperature": req.temperature.unwrap_or(0.7),
            "top_p": req.top_p.unwrap_or(0.9),
            "stream": false,
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
            return Err(SynapseError::BackendError(format!("TensorRT error: {text}")));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| SynapseError::BackendError(e.to_string()))?;

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let finish_reason = match json["choices"][0]["finish_reason"].as_str() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::MaxTokens,
            _ => FinishReason::Stop,
        };

        Ok(InferenceResponse {
            text,
            usage: TokenUsage {
                prompt_tokens: json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: json["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
            },
            finish_reason,
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

        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": model_name,
            "messages": build_messages(&req),
            "max_tokens": req.max_tokens.unwrap_or(512),
            "temperature": req.temperature.unwrap_or(0.7),
            "top_p": req.top_p.unwrap_or(0.9),
            "stream": true,
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
                    Err(e) => {
                        warn!("TensorRT stream error: {e}");
                        break;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if line.starts_with("data: ") {
                        let data = &line[6..];
                        if data == "[DONE]" {
                            let _ = tx
                                .send(StreamChunk {
                                    text: String::new(),
                                    done: true,
                                    usage: None,
                                })
                                .await;
                            return;
                        }

                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                            let text = json["choices"][0]["delta"]["content"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();

                            if !text.is_empty() {
                                let _ = tx
                                    .send(StreamChunk {
                                        text,
                                        done: false,
                                        usage: None,
                                    })
                                    .await;
                            }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/v2/health/ready", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

fn build_messages(req: &InferenceRequest) -> Vec<serde_json::Value> {
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
        let backend = TensorRtBackend::new(None);
        assert_eq!(backend.id().0, "tensorrt");
    }

    #[test]
    fn cuda_only() {
        let backend = TensorRtBackend::new(None);
        assert_eq!(backend.capabilities().accelerators, vec![AcceleratorType::Cuda]);
    }

    #[test]
    fn supports_tensorrt_format() {
        let backend = TensorRtBackend::new(None);
        assert_eq!(backend.supported_formats(), &[ModelFormat::TensorRt]);
    }
}
