# Inference Backends

Synapse uses a trait-based pluggable backend system. Each backend implements the `InferenceBackend` trait and is registered at runtime in the `BackendRegistry`.

## Available Backends

| Backend | Format | Runtime | Feature Flag |
|---------|--------|---------|-------------|
| llama.cpp | GGUF | Native FFI | `llamacpp` |
| Candle | SafeTensors | Pure Rust | `candle-backend` |
| Ollama | Any (via API) | HTTP client | `ollama` |
| vLLM | Any (via API) | HTTP client | `vllm` |
| GGUF | GGUF | Via candle-gguf | (included in candle) |
| ONNX | ONNX | ort crate | `onnx` |
| TensorRT | TensorRT-LLM | FFI/subprocess | `tensorrt` |

## Backend Selection

The router (`synapse-backends/src/router.rs`) automatically selects the best backend based on:
1. Model format (GGUF → llama.cpp, SafeTensors → candle, ONNX → ort)
2. Available hardware (CUDA → prefer llama.cpp with GPU layers, CPU-only → candle)
3. User preference (override via config or API parameter)

## Adding a New Backend

1. Create a module in `crates/synapse-backends/src/<name>/mod.rs`
2. Implement the `InferenceBackend` trait
3. Add a feature flag in `crates/synapse-backends/Cargo.toml`
4. Register in `lib.rs` behind the feature gate
