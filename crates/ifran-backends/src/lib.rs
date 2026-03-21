//! Pluggable inference backend system for Ifran.
//!
//! This crate provides the [`InferenceBackend`] trait and a collection of
//! backend implementations that can be enabled via feature flags. A
//! [`BackendRouter`] handles smart backend selection at runtime.

pub mod cost;
pub mod router;
pub mod traits;

// -- Backend modules, gated behind feature flags --

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(feature = "vllm")]
pub mod vllm;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "gguf")]
pub mod gguf;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "tensorrt")]
pub mod tensorrt;

#[cfg(feature = "tpu")]
pub mod tpu;

#[cfg(feature = "candle-backend")]
pub mod candle;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "gaudi")]
pub mod gaudi;

#[cfg(feature = "inferentia")]
pub mod inferentia;

#[cfg(feature = "oneapi")]
pub mod oneapi;

#[cfg(feature = "qualcomm")]
pub mod qualcomm;

#[cfg(feature = "xdna")]
pub mod xdna;

// Re-exports for convenience.
pub use router::BackendRouter;
pub use traits::{InferenceBackend, ModelHandle, hash_text_to_embedding};
