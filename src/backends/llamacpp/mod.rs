//! llama.cpp backend integration.
//!
//! Wraps the llama.cpp inference runtime via `llama-server` subprocess. This
//! approach avoids linking against C++ at compile time, supports any llama.cpp
//! build (CPU, CUDA, ROCm, Metal), and allows hot-swapping versions.
//!
//! The backend spawns a `llama-server` process per loaded model and
//! communicates via its HTTP API (OpenAI-compatible).

use crate::types::IfranError;
use crate::types::backend::{
    AcceleratorType, BackendCapabilities, BackendId, BackendLocality, DeviceConfig,
};
use crate::types::error::Result;
use crate::types::inference::{InferenceRequest, InferenceResponse, StreamChunk};
use crate::types::model::{ModelFormat, ModelManifest};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::process::{Child, Command};
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn};

use crate::backends::openai_compat::{
    build_openai_messages, parse_openai_response, stream_openai_sse,
};
use crate::backends::traits::{InferenceBackend, ModelHandle};

/// Starting port for llama-server instances.
const BASE_PORT: u16 = 8430;

/// Maximum port before wrapping back to [`BASE_PORT`].
const MAX_PORT: u16 = 65_000;

/// A running llama-server instance for one model.
struct ServerInstance {
    process: Child,
    port: u16,
    #[allow(dead_code)]
    model_path: String,
}

/// llama.cpp backend using `llama-server` subprocess.
pub struct LlamaCppBackend {
    /// Path to the llama-server binary.
    server_bin: String,
    /// Running instances keyed by model handle.
    instances: Arc<RwLock<HashMap<String, ServerInstance>>>,
    /// Next port to assign.
    next_port: Arc<RwLock<u16>>,
    /// HTTP client for communicating with instances.
    client: reqwest::Client,
}

impl LlamaCppBackend {
    /// Create a new llama.cpp backend.
    ///
    /// `server_bin` is the path to the `llama-server` binary. If None, looks
    /// in PATH.
    pub fn new(server_bin: Option<String>) -> Self {
        Self {
            server_bin: server_bin.unwrap_or_else(|| "llama-server".into()),
            instances: Arc::new(RwLock::new(HashMap::new())),
            next_port: Arc::new(RwLock::new(BASE_PORT)),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Allocate the next available port.
    ///
    /// Wraps back to [`BASE_PORT`] when the port exceeds [`MAX_PORT`] to avoid
    /// assigning privileged or invalid ports.
    async fn allocate_port(&self) -> u16 {
        let mut port = self.next_port.write().await;
        let p = *port;
        *port = if p >= MAX_PORT { BASE_PORT } else { p + 1 };
        p
    }

    /// Wait for a server instance to be ready (health endpoint).
    async fn wait_for_ready(&self, port: u16) -> Result<()> {
        let url = format!("http://127.0.0.1:{port}/health");
        for _ in 0..60 {
            if let Ok(resp) = self.client.get(&url).send().await {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
        Err(IfranError::BackendError(
            "llama-server failed to start within 60 seconds".into(),
        ))
    }
}

#[async_trait]
impl InferenceBackend for LlamaCppBackend {
    fn id(&self) -> BackendId {
        BackendId("llamacpp".into())
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
            supports_embeddings: false,
            supports_vision: false,
            locality: BackendLocality::Local,
        }
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Gguf]
    }

    async fn load_model(
        &self,
        manifest: &ModelManifest,
        device: &DeviceConfig,
    ) -> Result<ModelHandle> {
        let port = self.allocate_port().await;
        let model_path = &manifest.info.local_path;

        if !std::path::Path::new(model_path).exists() {
            return Err(IfranError::BackendError(format!(
                "Model file not found: {model_path}"
            )));
        }

        let mut cmd = Command::new(&self.server_bin);
        cmd.arg("--model")
            .arg(model_path)
            .arg("--port")
            .arg(port.to_string())
            .arg("--host")
            .arg("127.0.0.1");

        // GPU layers
        if device.accelerator != AcceleratorType::Cpu {
            let gpu_layers = manifest.gpu_layers.unwrap_or(999);
            cmd.arg("--n-gpu-layers").arg(gpu_layers.to_string());
        } else {
            cmd.arg("--n-gpu-layers").arg("0");
        }

        // Context length
        if let Some(ctx) = manifest.context_length {
            cmd.arg("--ctx-size").arg(ctx.to_string());
        }

        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        let process = cmd.spawn().map_err(|e| {
            IfranError::BackendError(format!(
                "Failed to start llama-server ({}): {e}",
                self.server_bin
            ))
        })?;

        let handle_id = format!("llamacpp-{port}");
        info!(handle = %handle_id, port, model = %model_path, "Starting llama-server");

        if let Err(e) = self.wait_for_ready(port).await {
            warn!(handle = %handle_id, "llama-server failed to become ready, killing process");
            let mut process = process;
            let _ = process.kill().await;
            let _ = process.wait().await;
            return Err(e);
        }

        let instance = ServerInstance {
            process,
            port,
            model_path: model_path.clone(),
        };

        self.instances
            .write()
            .await
            .insert(handle_id.clone(), instance);
        Ok(ModelHandle(handle_id))
    }

    async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        let mut instances = self.instances.write().await;
        if let Some(mut instance) = instances.remove(&handle.0) {
            info!(handle = %handle.0, "Stopping llama-server");
            let _ = instance.process.kill().await;
            // Reap the child process to prevent zombies
            let _ = instance.process.wait().await;
            Ok(())
        } else {
            Err(IfranError::ModelNotFound(handle.0))
        }
    }

    async fn infer(
        &self,
        handle: &ModelHandle,
        req: &InferenceRequest,
    ) -> Result<InferenceResponse> {
        let instances = self.instances.read().await;
        let instance = instances
            .get(&handle.0)
            .ok_or_else(|| IfranError::ModelNotFound(handle.0.clone()))?;

        let url = format!("http://127.0.0.1:{}/v1/chat/completions", instance.port);

        let messages = build_openai_messages(req);
        let body = serde_json::json!({
            "messages": messages,
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
            .map_err(|e| IfranError::BackendError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(IfranError::BackendError(format!(
                "llama-server returned HTTP {status}: {text}"
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| IfranError::BackendError(e.to_string()))?;

        parse_openai_response(&json)
    }

    async fn infer_stream(
        &self,
        handle: &ModelHandle,
        req: InferenceRequest,
    ) -> Result<mpsc::Receiver<StreamChunk>> {
        let instances = self.instances.read().await;
        let instance = instances
            .get(&handle.0)
            .ok_or_else(|| IfranError::ModelNotFound(handle.0.clone()))?;

        let url = format!("http://127.0.0.1:{}/v1/chat/completions", instance.port);
        let messages = build_openai_messages(&req);
        let body = serde_json::json!({
            "messages": messages,
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
            .map_err(|e| IfranError::BackendError(e.to_string()))?;

        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            let _ = stream_openai_sse(resp, tx).await;
        });

        Ok(rx)
    }

    async fn health_check(&self) -> Result<bool> {
        // Check if llama-server binary exists
        match Command::new(&self.server_bin)
            .arg("--version")
            .output()
            .await
        {
            Ok(output) => Ok(output.status.success()),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::traits::InferenceBackend;
    use crate::types::inference::FinishReason;

    #[test]
    fn new_default_server_bin() {
        let backend = LlamaCppBackend::new(None);
        assert_eq!(backend.server_bin, "llama-server");
    }

    #[test]
    fn new_custom_server_bin() {
        let backend = LlamaCppBackend::new(Some("/usr/local/bin/llama-server".into()));
        assert_eq!(backend.server_bin, "/usr/local/bin/llama-server");
    }

    #[test]
    fn backend_id() {
        let backend = LlamaCppBackend::new(None);
        assert_eq!(backend.id().0, "llamacpp");
    }

    #[test]
    fn backend_capabilities() {
        let backend = LlamaCppBackend::new(None);
        let caps = backend.capabilities();
        assert!(caps.supports_streaming);
        assert!(!caps.supports_embeddings);
        assert!(!caps.supports_vision);
        assert_eq!(caps.max_context_length, Some(131072));
        assert!(caps.accelerators.contains(&AcceleratorType::Cpu));
        assert!(caps.accelerators.contains(&AcceleratorType::Cuda));
    }

    #[test]
    fn supported_formats_is_gguf() {
        let backend = LlamaCppBackend::new(None);
        let formats = backend.supported_formats();
        assert_eq!(formats, &[ModelFormat::Gguf]);
    }

    #[tokio::test]
    async fn allocate_port_increments() {
        let backend = LlamaCppBackend::new(None);
        let p1 = backend.allocate_port().await;
        let p2 = backend.allocate_port().await;
        let p3 = backend.allocate_port().await;
        assert_eq!(p1, 8430);
        assert_eq!(p2, 8431);
        assert_eq!(p3, 8432);
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
        let msgs = build_openai_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Hello");
    }

    #[test]
    fn build_messages_with_system() {
        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: Some("Be helpful.".into()),
            sensitivity: None,
        };
        let msgs = build_openai_messages(&req);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "Be helpful.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn parse_completion_response_stop() {
        let json = serde_json::json!({
            "choices": [{
                "message": {"content": "Hello there!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8
            }
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "Hello there!");
        assert_eq!(resp.usage.prompt_tokens, 5);
        assert_eq!(resp.usage.completion_tokens, 3);
        assert_eq!(resp.usage.total_tokens, 8);
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn parse_completion_response_length() {
        let json = serde_json::json!({
            "choices": [{
                "message": {"content": "truncated"},
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 512,
                "total_tokens": 522
            }
        });
        let resp = parse_openai_response(&json).unwrap();
        assert!(matches!(resp.finish_reason, FinishReason::MaxTokens));
    }

    #[test]
    fn parse_completion_response_missing_fields() {
        let json = serde_json::json!({
            "choices": [{"message": {}}],
            "usage": {}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "");
        assert_eq!(resp.usage.prompt_tokens, 0);
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn parse_completion_response_empty_json() {
        let json = serde_json::json!({});
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "");
        assert_eq!(resp.usage.total_tokens, 0);
    }

    #[tokio::test]
    async fn unload_nonexistent_model_errors() {
        let backend = LlamaCppBackend::new(None);
        let result = backend
            .unload_model(ModelHandle("nonexistent".into()))
            .await;
        assert!(result.is_err());
    }

    /// Insert a fake server instance pointing at a mock server port.
    async fn insert_mock_instance(backend: &LlamaCppBackend, handle: &str, port: u16) {
        // Spawn a harmless process to satisfy the Child requirement
        let process = Command::new("sleep")
            .arg("60")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let instance = ServerInstance {
            process,
            port,
            model_path: "/tmp/test.gguf".into(),
        };
        backend
            .instances
            .write()
            .await
            .insert(handle.into(), instance);
    }

    #[tokio::test]
    async fn infer_success_with_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let port: u16 = server
            .url()
            .split(':')
            .next_back()
            .unwrap()
            .parse()
            .unwrap();

        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
                }"#,
            )
            .create_async()
            .await;

        let backend = LlamaCppBackend::new(None);
        insert_mock_instance(&backend, "llamacpp-test", port).await;

        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: Some(100),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };

        let resp = backend
            .infer(&ModelHandle("llamacpp-test".into()), &req)
            .await
            .unwrap();
        assert_eq!(resp.text, "Test response");
        assert_eq!(resp.usage.prompt_tokens, 5);
        assert_eq!(resp.usage.total_tokens, 8);
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn infer_model_not_loaded() {
        let backend = LlamaCppBackend::new(None);
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
        let port: u16 = server
            .url()
            .split(':')
            .next_back()
            .unwrap()
            .parse()
            .unwrap();

        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(500)
            .with_body("Server error")
            .create_async()
            .await;

        let backend = LlamaCppBackend::new(None);
        insert_mock_instance(&backend, "llamacpp-test", port).await;

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
            .infer(&ModelHandle("llamacpp-test".into()), &req)
            .await;
        assert!(matches!(result, Err(IfranError::BackendError(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn unload_kills_process() {
        let backend = LlamaCppBackend::new(None);
        insert_mock_instance(&backend, "llamacpp-kill-test", 9999).await;

        assert!(
            backend
                .instances
                .read()
                .await
                .contains_key("llamacpp-kill-test")
        );
        backend
            .unload_model(ModelHandle("llamacpp-kill-test".into()))
            .await
            .unwrap();
        assert!(
            !backend
                .instances
                .read()
                .await
                .contains_key("llamacpp-kill-test")
        );
    }
}
