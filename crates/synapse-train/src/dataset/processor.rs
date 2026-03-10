//! Dataset processor for transforming and preparing training data.
//!
//! Currently a passthrough — the actual tokenization and formatting is
//! handled by the Python training scripts. This module provides a hook
//! for future Rust-side preprocessing (deduplication, filtering, etc.).
