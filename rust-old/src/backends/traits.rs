//! Core trait definitions for pluggable inference backends.

use crate::types::{
    backend::{BackendCapabilities, BackendId, DeviceConfig},
    inference::{InferenceRequest, InferenceResponse, StreamChunk},
    model::{ModelFormat, ModelManifest},
};
use async_trait::async_trait;
use tokio::sync::mpsc;

/// Handle to a loaded model within a backend.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ModelHandle(pub String);

/// The core trait that all inference backends must implement.
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    fn id(&self) -> BackendId;
    fn capabilities(&self) -> BackendCapabilities;
    fn supported_formats(&self) -> &[ModelFormat];
    async fn load_model(
        &self,
        manifest: &ModelManifest,
        device: &DeviceConfig,
    ) -> crate::types::error::Result<ModelHandle>;
    async fn unload_model(&self, handle: ModelHandle) -> crate::types::error::Result<()>;
    async fn infer(
        &self,
        handle: &ModelHandle,
        req: &InferenceRequest,
    ) -> crate::types::error::Result<InferenceResponse>;
    async fn infer_stream(
        &self,
        handle: &ModelHandle,
        req: InferenceRequest,
    ) -> crate::types::error::Result<mpsc::Receiver<StreamChunk>>;
    async fn health_check(&self) -> crate::types::error::Result<bool>;

    /// Generate an embedding vector for the given text.
    ///
    /// Backends that natively support embeddings (e.g. Ollama, vLLM) should
    /// override this with a direct embedding API call.  The default
    /// implementation falls back to running a short inference and hashing the
    /// response into a fixed-size vector — better than a static hash of the
    /// raw input because the model's output captures semantic content.
    async fn embed(
        &self,
        handle: &ModelHandle,
        text: &str,
        dims: usize,
    ) -> crate::types::error::Result<Vec<f32>> {
        // Ask the model to summarise / rephrase — the output is semantically
        // richer than the raw input, producing a better hash-based embedding.
        let req = InferenceRequest {
            prompt: format!(
                "Represent the following text for semantic search. Only output the representation, nothing else.\n\n{text}"
            ),
            max_tokens: Some(64),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };
        let resp = self.infer(handle, &req).await?;
        Ok(hash_text_to_embedding(&resp.text, dims))
    }
}

/// Deterministic hash of text into a normalised f32 vector.
///
/// Uses a simple but effective byte-mixing strategy. This is NOT a learned
/// embedding — it preserves no real semantic similarity — but it provides a
/// stable, deterministic vector that can be stored and compared with cosine
/// similarity. Backends with native embedding support should override
/// [`InferenceBackend::embed`] instead.
pub fn hash_text_to_embedding(text: &str, dims: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; dims];
    // Mix bytes with position-dependent rotation so that word order matters.
    for (i, byte) in text.bytes().enumerate() {
        let idx = i % dims;
        let secondary = (i.wrapping_mul(7) + byte as usize) % dims;
        embedding[idx] += (byte as f32) / 255.0;
        embedding[secondary] += ((byte as f32) / 255.0) * 0.5;
    }
    // L2 normalise
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut embedding {
            *v /= norm;
        }
    }
    embedding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_handle_equality() {
        let a = ModelHandle("llamacpp-8430".into());
        let b = ModelHandle("llamacpp-8430".into());
        let c = ModelHandle("llamacpp-8431".into());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn model_handle_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ModelHandle("a".into()));
        set.insert(ModelHandle("a".into()));
        assert_eq!(set.len(), 1);
        set.insert(ModelHandle("b".into()));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn model_handle_clone() {
        let h = ModelHandle("test".into());
        let h2 = h.clone();
        assert_eq!(h, h2);
    }

    #[test]
    fn model_handle_debug() {
        let h = ModelHandle("test-handle".into());
        let debug = format!("{h:?}");
        assert!(debug.contains("test-handle"));
    }
}
