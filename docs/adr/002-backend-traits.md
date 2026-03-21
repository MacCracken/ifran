# ADR-002: Trait-based Backend Pluggability

**Status**: Accepted
**Date**: 2026-03-09

## Context

Ifran must support multiple inference runtimes (llama.cpp, Candle, Ollama, vLLM, ONNX, TensorRT). Each has different capabilities, model format support, and hardware requirements.

## Decision

Define an `InferenceBackend` trait with dynamic dispatch (`Arc<dyn InferenceBackend>`). Backends are registered at runtime in a `BackendRegistry`. Feature flags gate heavy native dependencies so builds without CUDA/TensorRT still compile.

## Consequences

- New backends can be added by implementing one trait
- Feature flags keep compile times fast for development (only build what you need)
- A router module auto-selects the best backend based on model format and hardware
- Follows the pattern established in Agnosticos's `LlmProvider` trait
- Some runtime overhead from dynamic dispatch (negligible compared to inference cost)
