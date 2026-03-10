# Inference Backends

Synapse uses a trait-based pluggable backend system. Each backend implements the `InferenceBackend` trait and is registered at runtime in the `BackendRouter`.

## Available Backends

| Backend | Format | Runtime | Feature Flag | Status |
|---------|--------|---------|-------------|--------|
| llama.cpp | GGUF | `llama-server` subprocess | `llamacpp` | Full |
| Candle | SafeTensors | Pure Rust (in-process) | `candle-backend` | Trait impl, inference pending |
| Ollama | GGUF (via API) | HTTP client | `ollama` | Full |
| vLLM | SafeTensors/PyTorch | HTTP client | `vllm` | Full |
| GGUF | GGUF | candle-gguf (in-process) | `gguf` | Trait impl, inference pending |
| ONNX | ONNX | ort crate (in-process) | `onnx` | Trait impl, inference pending |
| TensorRT | TensorRT engines | HTTP client (Triton) | `tensorrt` | Full |

All backends are enabled by default. Disable unneeded ones via feature flags in `Cargo.toml`.

## Backend Selection

The router (`synapse-backends/src/router.rs`) automatically selects the best backend based on:
1. **User preference** — explicit backend ID override via config or API parameter
2. **Configured default** — `default_backend` in config
3. **Model format** — GGUF → llama.cpp, SafeTensors → candle/vllm, ONNX → ort, TensorRT → tensorrt
4. **Hardware** — CUDA → prefer GPU-capable backends, CPU-only → candle or llama.cpp CPU mode

## llama.cpp Backend

Spawns a `llama-server` process per loaded model and communicates via its OpenAI-compatible HTTP API. Avoids C++ compile-time linking.

- Requires `llama-server` binary in `PATH` ([llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases))
- Each model gets its own server on an auto-allocated port (starting at 8430)
- Waits up to 60s for server readiness
- Supports streaming via SSE
- GPU layer offload controlled by `DeviceConfig`

## Ollama Backend

HTTP client to a running Ollama server (default: `http://127.0.0.1:11434`).

- Uses Ollama's `/api/chat` endpoint for inference
- Model load/unload via `keep_alive` parameter
- Streaming via Ollama's NDJSON response format
- Configurable server URL for remote Ollama instances

## vLLM Backend

HTTP client to a running vLLM server (default: `http://127.0.0.1:8000`).

- Uses vLLM's OpenAI-compatible `/v1/chat/completions` endpoint
- Models are loaded at vLLM server startup; Synapse registers and verifies availability
- Streaming via SSE
- Best for high-throughput GPU inference with PagedAttention and continuous batching

## TensorRT-LLM Backend

HTTP client to a Triton Inference Server with TensorRT-LLM backend (default: `http://127.0.0.1:8000`).

- Uses Triton's OpenAI-compatible API
- Pre-compiled TensorRT engine files loaded by Triton
- Streaming via SSE
- NVIDIA CUDA-only; optimal for maximum throughput on NVIDIA hardware

## Candle Backend

Pure-Rust inference via HuggingFace Candle. No Python or C++ dependencies.

- CPU and CUDA support
- SafeTensors model format
- Inference requires `candle`, `candle-nn`, `candle-transformers` crate dependencies (pending integration)

## GGUF Backend

Direct in-process GGUF loading via candle-gguf.

- CPU and CUDA support
- Lighter weight alternative to llama.cpp subprocess
- Inference requires `candle-gguf` crate dependency (pending integration)

## ONNX Backend

ONNX Runtime inference via the `ort` crate.

- CPU and CUDA execution providers
- ONNX model format
- Does not support streaming (run-to-completion only)
- Good for models exported from PyTorch/TensorFlow to ONNX
- Inference requires `ort` crate dependency (pending integration)

## Adding a New Backend

1. Create a module in `crates/synapse-backends/src/<name>/mod.rs`
2. Implement the `InferenceBackend` trait
3. Add a feature flag in `crates/synapse-backends/Cargo.toml`
4. Register in `lib.rs` behind the feature gate
5. Update the router's format matching if the backend handles a new format
