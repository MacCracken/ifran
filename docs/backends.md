# Inference Backends

Synapse uses a trait-based pluggable backend system. Each backend implements the `InferenceBackend` trait and is registered at runtime in the `BackendRegistry`.

## Available Backends

| Backend | Format | Runtime | Feature Flag |
|---------|--------|---------|-------------|
| llama.cpp | GGUF | `llama-server` subprocess | `llamacpp` |
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

## llama.cpp Backend

The llama.cpp backend spawns a `llama-server` process per loaded model and communicates via its OpenAI-compatible HTTP API. This avoids linking against C++ at compile time and supports any llama.cpp build (CPU, CUDA, ROCm, Metal).

Requirements:
- `llama-server` binary in `PATH` (from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases))
- Each loaded model gets its own server on an auto-allocated port (starting at 8430)
- The backend waits up to 60 seconds for the server to be ready
- Supports streaming inference via SSE

## Adding a New Backend

1. Create a module in `crates/synapse-backends/src/<name>/mod.rs`
2. Implement the `InferenceBackend` trait
3. Add a feature flag in `crates/synapse-backends/Cargo.toml`
4. Register in `lib.rs` behind the feature gate
