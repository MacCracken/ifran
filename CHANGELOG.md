# Changelog

All notable changes to Synapse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows CalVer: YYYY.M.D / YYYY.M.D-N for patches.

## [2026.3.10] — (in progress)

### Added

#### Core
- `synapse-types`: Core data structures — models, backends, inference, training, errors
- `synapse-core/config`: TOML config loading with defaults (`~/.synapse/synapse.toml`)
- `synapse-core/storage/db`: SQLite model catalog with full CRUD, schema migrations, and indexes
- `synapse-core/storage/layout`: Filesystem layout for `~/.synapse/models/` with slug generation
- `synapse-core/hardware/detect`: GPU detection (NVIDIA via nvidia-smi, AMD via sysfs, CPU from /proc)
- `synapse-core/registry/huggingface`: HuggingFace Hub API — model info, GGUF resolution by quant, search
- `synapse-core/registry/scanner`: Local filesystem scanner for GGUF, SafeTensors, ONNX, PyTorch, TensorRT files
- `synapse-core/pull/downloader`: Chunked HTTP download with resume via `.part` files and Range headers
- `synapse-core/pull/verifier`: SHA-256 and BLAKE3 integrity verification with auto-detection
- `synapse-core/pull/progress`: Broadcast-channel progress tracking for multi-consumer updates
- `synapse-core/lifecycle/manager`: Model load/unload orchestration with backend-agnostic handle tracking
- `synapse-core/lifecycle/memory`: VRAM/RAM budget estimation with GPU/CPU fallback
- CalVer versioning via `VERSION` file — all crates inherit from workspace
- Protobuf definitions for core, bridge, and training services

#### Backends
- `synapse-backends/traits`: `InferenceBackend` trait — load, unload, infer, stream, health check
- `synapse-backends/llamacpp`: llama.cpp via `llama-server` subprocess with auto port allocation
- `synapse-backends/ollama`: Ollama HTTP client — chat, streaming, model load/unload via keep_alive
- `synapse-backends/vllm`: vLLM HTTP client — OpenAI-compatible chat and streaming
- `synapse-backends/tensorrt`: TensorRT-LLM HTTP client to Triton server with streaming
- `synapse-backends/candle`: Candle (pure Rust) backend for SafeTensors — trait impl, inference pending candle crate dep
- `synapse-backends/gguf`: Direct GGUF loading backend — trait impl, inference pending candle-gguf dep
- `synapse-backends/onnx`: ONNX Runtime backend — trait impl, inference pending ort crate dep
- `synapse-backends/router`: Smart backend auto-selection by format, hardware, and user preference

#### API Server
- `synapse-api/rest/router`: Axum router with all route groups, CORS, telemetry, auth
- `synapse-api/rest/models`: `GET /models`, `GET /models/:id`, `DELETE /models/:id`
- `synapse-api/rest/inference`: `POST /inference`, `POST /inference/stream` (SSE)
- `synapse-api/rest/openai_compat`: `POST /v1/chat/completions` (streaming + non-streaming), `GET /v1/models`
- `synapse-api/rest/training`: `POST /training/jobs`, `GET /training/jobs`, `GET /training/jobs/:id`, `POST /training/jobs/:id/cancel`
- `synapse-api/rest/system`: `GET /health`, `GET /system/status`
- `synapse-api/middleware/telemetry`: Request tracing via tower-http
- `synapse-api/middleware/auth`: Optional Bearer token auth via `SYNAPSE_API_KEY`
- `synapse-api/state`: Shared application state with config, DB, backend router, model manager, job manager

#### Training
- `synapse-train/job/manager`: Job lifecycle (create, start, cancel) with concurrent job limits
- `synapse-train/job/scheduler`: FIFO priority queue
- `synapse-train/job/status`: Job state machine (Queued → Running → Completed/Failed/Cancelled)
- `synapse-train/executor/docker`: Docker container executor with GPU passthrough and method-specific script selection
- `synapse-train/executor/subprocess`: Python subprocess executor
- `synapse-train/dataset/loader`: JSONL, CSV, Parquet, HuggingFace dataset loading with sample counting
- `synapse-train/dataset/validator`: Schema validation for JSONL and CSV formats
- `synapse-train/methods`: LoRA/QLoRA, full fine-tune, DPO, RLHF, distillation configs and arg generation
- `synapse-train/checkpoint/store`: Checkpoint save/load/list/prune with metadata
- `synapse-train/checkpoint/merger`: LoRA adapter merging into base model via PEFT
- Python training scripts: `train_sft.py`, `train_full.py`, `train_dpo.py`, `train_rlhf.py`, `train_distill.py`

#### SY Bridge
- `synapse-bridge/server`: gRPC server with connection state machine, heartbeat, degraded mode
- `synapse-bridge/client`: gRPC client with reconnect (exponential backoff), capability announcement, GPU requests
- `synapse-bridge/protocol`: Connection states, heartbeat config, capability announcement types
- `synapse-bridge/discovery`: SY endpoint discovery (config → `SY_ENDPOINT` env → localhost:9420)

#### CLI
- `synapse pull`: Model pull with HuggingFace resolution, progress bar, integrity check, catalog registration
- `synapse list`: Table-formatted model listing
- `synapse rm`: Model removal with confirmation, disk cleanup, catalog deletion
- `synapse run`: Interactive inference with streaming output
- `synapse serve`: Full API server
- `synapse train`: Training job creation with `--base-model`, `--dataset`, `--method`
- `synapse status`: Hardware and catalog status

#### Desktop Application
- Tauri v2 + SvelteKit scaffold with dark theme UI
- Dashboard: model count, loaded models, hardware summary, version
- Models page: browse, delete, pull progress
- Chat page: model selection, message history, OpenAI-compatible inference
- Training page: job list with progress bars, step/epoch/loss, cancel
- Settings page: server status, hardware info, config guidance
- 10 Tauri commands bridging frontend to Synapse REST API

#### Infrastructure
- 8-crate Cargo workspace: types, core, backends, train, api, bridge, cli, desktop
- CI/CD (GitHub Actions): build, quality, security, test/coverage, benchmarks, docs, container, license
- Release pipeline: multi-arch binaries (amd64 + arm64), SBOM, GitHub Release
- Docker: server, dev, trainer, release containers with multi-arch support
- Dependency update automation (weekly cargo update PRs)
- Governance: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, SUPPORT
- 82 tests across all modules
