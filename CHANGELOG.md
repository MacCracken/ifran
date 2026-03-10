# Changelog

All notable changes to Synapse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows CalVer: YYYY.M.D / YYYY.M.D-N for patches.

## [2026.3.10] — (in progress)

### Added
- `synapse-bridge/server`: gRPC bridge server with connection state machine, heartbeat, and degraded mode support
- `synapse-bridge/client`: gRPC bridge client with connect, reconnect (exponential backoff), capability announcement, GPU requests, progress reporting
- `synapse-bridge/protocol`: Bridge protocol types — connection states, heartbeat config, capability announcement
- `synapse-bridge/discovery`: SY endpoint discovery chain (explicit config → `SY_ENDPOINT` env → localhost:9420)
- `synapse-train/job/manager`: Training job lifecycle manager with concurrent job limits and background task spawning
- `synapse-train/job/scheduler`: FIFO priority queue for training job scheduling
- `synapse-train/job/status`: Job state machine (Queued → Running → Completed/Failed/Cancelled) with progress tracking
- `synapse-train/executor/docker`: Docker container executor with GPU passthrough and container cancellation
- `synapse-train/executor/subprocess`: Python subprocess executor with method-specific training scripts
- `synapse-train/dataset/loader`: Dataset loading for JSONL, CSV, Parquet, and HuggingFace datasets with sample counting
- `synapse-train/dataset/validator`: Schema validation for JSONL and CSV training data formats
- `synapse-train/methods/lora`: LoRA/QLoRA default configs (rank 16, alpha 32) and CLI argument generation
- `synapse-train/methods/full`: Full fine-tuning defaults and argument generation
- `synapse-train/methods/dpo`: DPO defaults (lr 5e-6, beta 0.1) and argument generation
- `synapse-train/methods/rlhf`: RLHF argument generation
- `synapse-train/methods/distillation`: Knowledge distillation argument generation
- `synapse-train/checkpoint/store`: Checkpoint directory management — save, load, list, latest, prune
- `synapse-train/checkpoint/merger`: LoRA adapter merging into base model via PEFT
- `synapse-cli train`: Training command with `--base-model`, `--dataset`, `--method` flags
- `synapse-desktop`: Tauri v2 + SvelteKit desktop application scaffold
  - Model management page: browse, delete, pull progress
  - Chat interface: model selection, message history, OpenAI-compatible inference
  - Training dashboard: job list with progress bars, step/epoch/loss display, cancel
  - Settings page: server status, hardware info (CPU, GPU, VRAM), config guidance
  - System dashboard: model count, loaded models, hardware summary, version
  - Dark theme UI with sidebar navigation
  - Tauri commands: `list_models`, `get_model`, `delete_model`, `pull_model`, `send_message`, `get_status`, `get_hardware`, `list_jobs`, `create_job`, `cancel_job`
- `synapse-api/middleware/auth`: Bearer token authentication middleware — optional via `SYNAPSE_API_KEY` env, skips `/health`
- 14 API integration tests covering health, system status, model CRUD, OpenAI-compat, training jobs, and 404 handling
- Python training scripts: `train_sft.py` (LoRA/QLoRA SFT), `train_full.py` (full fine-tune), `train_dpo.py` (DPO), `train_rlhf.py` (PPO-based RLHF), `train_distill.py` (knowledge distillation)
- `docker/Dockerfile.trainer`: Updated to bundle all training scripts, method-specific script selection
- `synapse-api/rest/training`: Training REST endpoints — `POST /training/jobs`, `GET /training/jobs`, `GET /training/jobs/:id`, `POST /training/jobs/:id/cancel`
- `synapse-backends/ollama`: Ollama HTTP backend — chat, streaming, model load/unload via keep_alive, configurable server URL
- `synapse-backends/vllm`: vLLM HTTP backend — OpenAI-compatible chat and streaming, model registration via /v1/models
- `synapse-backends/candle`: Candle (pure Rust) backend for SafeTensors — full trait impl, inference pending candle crate dependency
- `synapse-backends/gguf`: Direct GGUF loading backend — full trait impl, inference pending candle-gguf dependency
- `synapse-backends/onnx`: ONNX Runtime backend — full trait impl, inference pending ort crate dependency
- `synapse-backends/tensorrt`: TensorRT-LLM backend — HTTP client to Triton server with streaming
- `synapse-backends/router`: Smart backend auto-selection based on model format, hardware, and user preference
- `synapse-api/rest/router`: Full Axum router with model, inference, system, and OpenAI-compatible routes + CORS + telemetry
- `synapse-api/rest/models`: Model CRUD endpoints — `GET /models`, `GET /models/:id`, `DELETE /models/:id`
- `synapse-api/rest/inference`: Inference endpoints — `POST /inference` (full) and `POST /inference/stream` (SSE)
- `synapse-api/rest/openai_compat`: OpenAI drop-in replacement — `POST /v1/chat/completions` (streaming + non-streaming) and `GET /v1/models`
- `synapse-api/rest/system`: System endpoints — `GET /health` and `GET /system/status` (hardware, loaded models, backends)
- `synapse-api/middleware/telemetry`: Request tracing middleware via `tower-http`
- `synapse-api/state`: Shared application state with config, DB, backend router, and model manager
- `synapse-cli serve`: Now uses the full API router instead of minimal stub
- `synapse-backends/llamacpp`: llama.cpp backend via `llama-server` subprocess — load/unload models, inference, streaming via OpenAI-compatible API
- `synapse-core/lifecycle/manager`: Model load/unload orchestration with backend-agnostic handle tracking
- `synapse-core/lifecycle/memory`: VRAM/RAM budget estimation and pre-load memory checks with GPU/CPU fallback
- `synapse-cli run`: Interactive inference session with streaming output
- `synapse-cli serve`: Minimal API server with `/health` and `/v1/models` endpoints
- `synapse-core/registry/huggingface`: HuggingFace Hub API client — model info, GGUF file resolution by quant level, search
- `synapse-core/pull/downloader`: Chunked HTTP downloader with resume support via `.part` files and Range headers
- `synapse-core/pull/verifier`: SHA-256 and BLAKE3 integrity verification with auto-detection
- `synapse-core/pull/progress`: Broadcast-channel progress tracking for multi-consumer updates (CLI, API, desktop)
- `synapse-core/registry/scanner`: Local filesystem scanner for model files (GGUF, SafeTensors, ONNX, PyTorch, TensorRT)
- `synapse-cli pull`: Full model pull command with HuggingFace resolution, progress bar, integrity check, and catalog registration
- `synapse-cli list`: Table-formatted model listing from local catalog
- `synapse-cli rm`: Model removal with confirmation prompt, disk cleanup, and catalog deletion
- `synapse-core/storage/db`: SQLite model catalog with full CRUD operations (insert, get, get_by_name, list, update, delete, count) and schema migrations
- `synapse-core/storage/layout`: Filesystem layout manager for `~/.synapse/` directory structure with model slug generation, directory creation, and cleanup
- `synapse-core/hardware/detect`: Hardware detection — NVIDIA GPUs via nvidia-smi, AMD ROCm GPUs via sysfs, CPU info from /proc; unified `SystemHardware` snapshot for backend selection
- CalVer versioning via workspace — all crates inherit from `VERSION` file
- 82 tests across storage, pull, registry, hardware, lifecycle, training, bridge, backend, and API integration
- Initial project scaffold with 8-crate Cargo workspace
- Protobuf definitions for core, bridge, and training services
- Pluggable inference backend trait system with 7 backend stubs (llama.cpp, Candle, Ollama, vLLM, GGUF, ONNX, TensorRT)
- Training orchestration framework with Docker/subprocess/native executors
- CLI structure with pull, list, run, serve, train, status, remove commands
- REST API (Axum) + gRPC (tonic) server structure
- SY<->Synapse bidirectional gRPC bridge
- Documentation: architecture, roadmap, 6 ADRs, 3 guides, API reference
- CI/CD pipeline (GitHub Actions): build, quality, security, test, benchmarks, docs, container, license
- Release pipeline with SBOM generation and multi-arch container publishing
- Dependency update automation (weekly cargo update PRs)
- Docker support: server, dev, trainer, release containers
- Agnosticos integration: systemd service, package spec
- Governance: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, SUPPORT
- Configuration: .audit.toml, osv-scanner.toml, .editorconfig, .gitattributes
