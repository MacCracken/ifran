//! Ifran — Local LLM inference, training, and fleet management platform.
//!
//! # Architecture
//!
//! ```text
//! Clients (CLI, API, Bridge)
//!     │
//!     ▼
//! Server (REST + gRPC)
//!     │
//!     ├──▶ Backends (13 inference backends)
//!     ├──▶ Training (LoRA, QLoRA, Full, DPO, RLHF, Distillation)
//!     ├──▶ Evaluation (MMLU, perplexity, custom)
//!     └──▶ Fleet (multi-node, GPU scheduling)
//! ```
//!
//! # Feature flags
//!
//! Backend features (each enables one inference backend):
//! `llamacpp`, `candle-backend`, `gguf`, `onnx`, `tensorrt`, `tpu`, `vllm`,
//! `ollama`, `wasm`, `gaudi`, `inferentia`, `metal`, `vulkan`, `oneapi`,
//! `qualcomm`, `xdna`
//!
//! Other features:
//! - **`server`** — API server, gRPC bridge, fleet management, metrics, hoosh integration.
//! - **`hwaccel`** — Hardware accelerator detection via `ai-hwaccel`.
//! - **`otlp`** — OpenTelemetry OTLP trace export.
//! - **`full`** — All of the above.

// -- Shared types --
pub mod types;

// -- Audit trail --
pub mod audit;

// -- Core domain modules --
pub mod ab_test;
pub mod budget;
pub mod config;
pub mod dataset;
pub mod drift;
pub mod eval;
pub mod experiment;
pub mod fleet;
pub mod hardware;
pub mod lifecycle;
pub mod lineage;
pub mod marketplace;
pub mod preference;
pub mod pull;
pub mod rag;
pub mod registry;
pub mod rlhf;
pub mod scoring;
pub mod storage;
pub mod tenant;
pub mod training_events;
pub mod versioning;

// -- Inference backends --
pub mod backends;

// -- Training pipeline --
pub mod train;

// -- gRPC bridge (server feature) --
#[cfg(feature = "server")]
pub mod bridge;

// -- API server (server feature) --
#[cfg(feature = "server")]
pub mod server;

// -- CLI (requires sqlite for ModelDatabase) --
#[cfg(feature = "sqlite")]
pub mod cli;

// Re-export core types at crate root for convenience
pub use types::error::IfranError;
pub use types::model::{ModelFormat, ModelId, ModelInfo, QuantLevel};
pub use types::pagination::PagedResult;
pub use types::tenant::TenantId;
