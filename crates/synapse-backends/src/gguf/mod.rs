//! GGUF model format utilities.
//!
//! Provides helpers for loading, inspecting, and validating GGUF-formatted
//! model files. This module is not a runtime backend itself but supplies
//! shared functionality used by backends that consume GGUF models (e.g.
//! llama.cpp, candle).
