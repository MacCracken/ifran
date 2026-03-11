# Changelog

All notable changes to Synapse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows CalVer: YYYY.M.D / YYYY.M.D-N for patches.

## [2026.3.10] — (in progress)

### Added

#### Core
- `synapse-types`: Core data structures — models, backends, inference, training, eval, marketplace, distributed, errors
- `synapse-core/config`: TOML config loading with auto-discovery (`SYNAPSE_CONFIG` → `~/.synapse/` → `/etc/synapse/` → defaults)
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
- `synapse-core/eval/runner`: Eval runner with run lifecycle, custom benchmark execution via closure-based inference
- `synapse-core/eval/store`: SQLite eval results store with CRUD
- `synapse-core/eval/benchmarks`: JSONL sample loading, exact/contains match scoring
- `synapse-core/marketplace/catalog`: SQLite marketplace catalog — publish, search, list, unpublish
- `synapse-core/marketplace/publisher`: Create marketplace entries from local models
- `synapse-core/marketplace/resolver`: Peer management for remote marketplace search
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
- `synapse-api/rest/eval`: `POST /eval/runs`, `GET /eval/runs`, `GET /eval/runs/:id`
- `synapse-api/rest/marketplace`: `GET /marketplace/search`, `GET /marketplace/entries`, `POST /marketplace/publish`, `DELETE /marketplace/entries/:name`
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
- `synapse-train/distributed/coordinator`: Distributed job creation, worker assignment, lifecycle
- `synapse-train/distributed/worker`: Worker local state, distributed CLI arg generation, lifecycle
- `synapse-train/distributed/aggregator`: Checkpoint aggregation plans (average/weighted), command builder
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
- `synapse eval`: Model evaluation with benchmark selection (`--benchmark`, `--dataset`, `--sample-limit`)
- `synapse marketplace search/publish/unpublish`: Model marketplace management

#### Desktop Application
- Tauri v2 + SvelteKit scaffold with dark theme UI
- Dashboard: model count, loaded models, hardware summary, version
- Models page: browse, delete, pull progress
- Chat page: model selection, message history, OpenAI-compatible inference
- Training page: job list with progress bars, step/epoch/loss, cancel
- Settings page: server status, hardware info, config guidance
- 10 Tauri commands bridging frontend to Synapse REST API

#### Agnosticos Integration
- `deploy/synapse.service`: systemd unit with security hardening (ProtectSystem, PrivateTmp, NoNewPrivileges, GPU device access)
- `deploy/agnosticos/synapse.pkg.toml`: Package spec with user creation hooks, capability registration, systemd setup
- `deploy/synapse.toml.example`: System-level config template with all backends documented
- Config auto-discovery chain: `SYNAPSE_CONFIG` env → `~/.synapse/synapse.toml` → `/etc/synapse/synapse.toml` → defaults
- Agent-runtime capability provider registration with Agnosticos

#### Infrastructure
- 8-crate Cargo workspace: types, core, backends, train, api, bridge, cli, desktop
- CI/CD (GitHub Actions): build (x86_64 + aarch64), quality, security, per-package tests, coverage, docs, container, license
- Release pipeline: multi-arch binaries (amd64 + arm64), SBOM, GitHub Release
- Docker: server, dev, trainer, release containers with multi-arch support
- rustls-tls for cross-compilation without OpenSSL headers
- Dependency update automation (weekly cargo update PRs)
- Governance: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, SUPPORT
- 236 tests across all modules (~55% coverage)

#### Model Evaluation Benchmarks
- `synapse-core/eval/benchmarks`: MMLU, HellaSwag, HumanEval, perplexity prompt formatting and scoring
- `synapse-core/eval/runner`: `run_benchmark()` dispatcher with per-benchmark runners (MMLU, HellaSwag, HumanEval, perplexity)
- `synapse-api/rest/eval`: Background benchmark execution wired to inference backends via closure-based `infer_fn`
- `synapse-cli/eval`: CLI eval command wired to local API with all benchmark types

#### Model Marketplace — Remote & Trust
- `synapse-core/marketplace/resolver`: Remote peer search via `GET /marketplace/search` on each peer, deduplication
- `synapse-core/marketplace/trust`: Trust/verification layer — `TrustLevel` (Untrusted/ChecksumVerified/TrustedPublisher), `TrustPolicy`, `verify_entry()`, `verify_download()`
- `synapse-api/rest/marketplace`: Model download endpoint (`GET /marketplace/download/:name`), model pull endpoint (`POST /marketplace/pull`) with SHA-256 verification
- `synapse-cli/marketplace`: `synapse marketplace pull --peer <url>` command with trust verification

#### Distributed Training
- `synapse-api/rest/distributed`: Full REST API for distributed job management (create, list, get, assign workers, start, complete, fail, aggregate)
- `synapse-cli/train`: `--distributed`, `--world-size`, `--strategy` flags for distributed training
- `synapse-bridge/client`: `request_worker_assignment()` and `sync_checkpoint()` RPCs for cross-node coordination
- `synapse-train/distributed/coordinator`: `collect_checkpoint_paths()` for checkpoint synchronization
- `synapse-train/distributed/aggregator`: `FederatedConfig`, `build_federated_command()` for federated averaging

#### SY Bridge Integration
- `synapse-api/state`: Bridge client/server initialized in AppState when `bridge.enabled = true`, with SY endpoint discovery
- `synapse-api/rest/bridge`: REST endpoints — `GET /bridge/status`, `POST /bridge/connect`, `POST /bridge/heartbeat`
- `synapse-api/main`: Auto-connect to SY on startup, background heartbeat task with loaded models/GPU/active jobs
- `synapse-api/rest/system`: Bridge connection state included in `/system/status`
- `synapse-api/rest/training`: Training job start and cancel events reported to SY via bridge client
- `synapse-api/rest/distributed`: Worker assignments forwarded to SY via `RequestWorkerAssignment`, checkpoint sync via `SyncCheckpoint` on worker completion, job completion reported to SY
- 243 tests across all modules (~56% coverage)

### Fixed
- CI/CD container image build timeout: switched from compiling Rust inside Docker (30+ min under QEMU for arm64) to using pre-built binaries from the build-release job via `Dockerfile.release` with `TARGETARCH`
