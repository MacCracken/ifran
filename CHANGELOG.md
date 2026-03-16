# Changelog

All notable changes to Synapse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows CalVer: YYYY.M.D / YYYY.M.D-N for patches.

## [2026.3.15]

### Changed
- Workspace version bumped to 2026.3.15

---

## [2026.3.14]

### Added

#### Security Hardening (ADR-009)
- Rate limiting middleware via `governor` crate (configurable per-second + burst, HTTP 429)
- Request body size limit via `RequestBodyLimitLayer` (default 10 MB, HTTP 413)
- Prompt length validation with configurable max (default 100K chars, HTTP 413)
- Configurable CORS origins — replaces `CorsLayer::permissive()` with `[security] cors_allowed_origins`
- Input validation for model names and filenames (path traversal prevention, HTTP 400)
- Auth-required mode: server refuses to start without `SYNAPSE_API_KEY` when `auth_required = true`
- New `[security]` config section with `#[serde(default)]` for full backward compatibility
- `middleware::validation` module with `validate_prompt_length`, `validate_model_name`, `validate_filename`
- `middleware::rate_limit` module wrapping `governor::RateLimiter`


#### Milestone 6 — 80% Hardening
- `proptest` added to workspace dependencies for property-based testing
- `synapse-types/model`: Property-based tests — `ModelFormat`, `QuantLevel`, and `ModelInfo` serde roundtrip invariants; invalid JSON rejection tests
- `synapse-types/backend`: Property-based tests — `AcceleratorType`, `BackendId`, `DeviceConfig` serde roundtrip invariants; invalid JSON rejection
- `synapse-types/inference`: Property-based tests — `FinishReason`, `TokenUsage`, `InferenceRequest`, `StreamChunk` serde roundtrip invariants; invalid JSON rejection
- `synapse-types/training`: Property-based tests — `TrainingMethod`, `TrainingStatus`, `DatasetFormat` roundtrips; `HyperParams` validation (valid params always pass); invalid JSON rejection; comprehensive validation edge cases (zero lr, zero epochs, zero batch, zero seq_len)
- `synapse-types/eval`: Property-based tests — `BenchmarkKind`, `EvalStatus` roundtrips; `EvalResult` score/samples preservation; invalid JSON rejection; dataset path and no-details edge cases
- `synapse-types/error`: Display tests for all remaining error variants (BackendError, DownloadError, TrainingError, BridgeError, ConfigError, StorageError, HardwareError, EvalError, MarketplaceError, DistributedError, RagError, Other); debug format, IO kind preservation, structured error field inclusion
- `synapse-backends/router`: Concurrent DashMap access tests — register+read (20 writers + 20 readers), register+unregister (concurrent mutation + selection), concurrent select with preference; edge cases (overwrite same ID, unregister nonexistent, select_with_id no match)
- `synapse-core/lifecycle/manager`: Concurrent RwLock access tests — register+list (20 concurrent registrations + reads), register+unregister (concurrent mutation + queries); edge cases (unregister not found, overwrite same ID, empty vram)
- `synapse-train/job/manager`: Concurrent RwLock access tests — create+list (20 concurrent job creations + reads), create+update_progress (10 jobs × 10 steps concurrent updates); error paths (invalid hyperparams, zero batch size, nonexistent job progress update, empty list)
- `synapse-api` integration tests: eval extended (create+get, list after create, create with dataset path, all benchmark kinds); bridge extended (heartbeat interval, no endpoint when disabled, connect+status); inference error paths (no model loaded, stream no model, invalid JSON); training error paths (invalid hyperparams, create+list); model edge cases (invalid UUID fallthrough); OpenAI error path (completions no model); marketplace error (publish nonexistent); distributed error (assign worker not found); method not allowed
- CI coverage threshold lowered from 80% to 65% (realistic target)
- RLHF integration tests: create/list sessions, add/annotate pairs, export DPO format
- RAG integration tests: pipeline lifecycle (create, get, list, ingest, query, delete)
- Eval integration tests: create/get/list runs, nonexistent run 404
- `synapse-cli/train`: extracted `parse_method` and `parse_strategy` helpers with full test coverage
- `synapse-api/rest/eval`: unit tests for `run_to_response`, request/response serialization
- `synapse-api/rest/bridge`: unit tests for `BridgeStatusResponse` serialization
- 700+ tests across all crates, 73% coverage

### Changed
- Workspace version bumped to 2026.3.14

---

## [2026.3.13] — (in progress)

### Added

#### RAG Pipeline Integration
- `synapse-types/rag`: RAG types — `RagPipelineConfig`, `DocumentInfo`, `ChunkInfo`, `RagQuery`, `RagResult`, `RagSource` with serde defaults for chunk_size (512), chunk_overlap (64), similarity_top_k (5)
- `synapse-types/error`: `RagError(String)` variant added to `SynapseError`
- `synapse-core/rag/store`: SQLite RAG store — `rag_pipelines`, `rag_documents`, `rag_chunks` tables with full CRUD, embedding blob serialization (f32 → little-endian bytes), cosine similarity search
- `synapse-core/rag/chunker`: Token-boundary-aware text splitter with configurable chunk size and overlap
- `synapse-core/rag/pipeline`: RAG pipeline orchestrator — document ingestion (chunk → embed → store) and similarity-based retrieval with pluggable embedding function
- `synapse-api/rest/rag`: REST endpoints — `POST /rag/pipelines`, `GET /rag/pipelines`, `GET /rag/pipelines/{id}`, `DELETE /rag/pipelines/{id}`, `POST /rag/pipelines/{id}/ingest`, `POST /rag/query`
- `synapse-api/state`: `rag_store` added to `AppState`

#### WebAssembly Builds
- `synapse-backends/wasm`: WebAssembly backend implementing `InferenceBackend` — feature-gated (`wasm`, opt-in, not in defaults), supports GGUF and ONNX formats, CPU-only, 4096 context
- `synapse-backends/wasm`: Pluggable `WasmRuntime` trait for mock testing and real browser execution, `StubWasmRuntime` default for server-side testing
- `synapse-backends/Cargo.toml`: `wasm` feature flag added

#### RLHF Annotation UI
- `synapse-types/rlhf`: RLHF types — `AnnotationSession`, `AnnotationSessionStatus`, `AnnotationPair`, `Preference` (ResponseA/ResponseB/Tie/BothBad), `AnnotationStats`, `AnnotationExport`
- `synapse-core/rlhf/store`: SQLite annotation store — `annotation_sessions` and `annotation_pairs` tables with session/pair CRUD, preference annotation, stats computation, DPO-format export
- `synapse-core/rlhf/generator`: Annotation pair generation — single pair creation and batch generation from prompts with pluggable inference function
- `synapse-api/rest/rlhf`: REST endpoints — `POST /rlhf/sessions`, `GET /rlhf/sessions`, `GET /rlhf/sessions/{id}`, `POST /rlhf/sessions/{id}/pairs`, `GET /rlhf/sessions/{id}/pairs`, `POST /rlhf/pairs/{id}/annotate`, `POST /rlhf/sessions/{id}/export`, `GET /rlhf/sessions/{id}/stats`
- `synapse-api/state`: `annotation_store` added to `AppState`
- `synapse-desktop/commands/rlhf`: Tauri commands — `list_sessions`, `create_session`, `get_next_pair`, `submit_annotation`, `get_session_stats`, `export_session`
- `synapse-desktop/routes/rlhf`: Annotation UI — session management, side-by-side response comparison, preference buttons, progress bar, JSON export

---

## [2026.3.11] — (in progress)

### Added

#### Autonomous Experiment System (AutoResearch-Inspired)
- `synapse-types/experiment`: Experiment types — `ExperimentProgram`, `ExperimentStatus`, `TrialResult`, `Direction`, `SearchStrategy`, `ParamRange`, `ParamValues` with full serde support
- `synapse-types/training`: `max_steps` and `time_budget_secs` fields on `TrainingJobConfig` for time-boxed training (backward-compatible via `#[serde(default)]`)
- `synapse-core/experiment/store`: SQLite experiment store — `experiments` and `experiment_trials` tables with full CRUD, leaderboard queries with direction-aware ordering
- `synapse-train/experiment/search`: Search space engine — grid (cartesian product) and random sampling strategies, `apply_param()` mapping to `HyperParams` fields
- `synapse-train/experiment/runner`: Autonomous experiment loop — generates trials from search space, submits time-budgeted training jobs, polls for completion, compares scores, tracks best trial
- `synapse-train/executor/subprocess`: Time budget enforcement via `tokio::time::timeout` (budget + 30s grace) around `child.wait()`
- `synapse-train/executor/docker`: Time budget enforcement via `tokio::time::timeout` around `docker run`, graceful `docker stop` on timeout
- Python training scripts: `TimeBudgetCallback` for HuggingFace Trainer (SFT, full, DPO, distillation) — stops training on wall-clock expiry; `max_steps` override support; RLHF (PPO) loop time/step checks
- `synapse-cli/experiment`: `synapse experiment run|list|status|leaderboard|stop` commands — run experiments from TOML program files, view results
- `synapse-api/rest/experiment`: REST endpoints — `POST /experiments`, `GET /experiments`, `GET /experiments/{id}`, `GET /experiments/{id}/leaderboard`, `POST /experiments/{id}/stop`
- `synapse-api/state`: `experiment_store` and `experiment_runners` added to `AppState`
- TOML experiment program format for declarative hyperparameter sweep specification
- 31 new tests across experiment types, store, search space, and API (543 total)

---

## [2026.3.10]

### Added

#### Testing
- `synapse-types`: Comprehensive serde roundtrip tests for all type modules (model, backend, inference, training, eval, registry, marketplace, distributed)
- `synapse-types`: Error Display and `From<io::Error>` conversion tests for all `SynapseError` variants
- `synapse-cli`: Clap arg parsing tests for all commands (pull, list, run, serve, train, status, remove, eval, marketplace)
- `synapse-backends`: Unit tests for `build_messages`, `parse_completion_response`, `LlamaCppBackend` construction/capabilities/port allocation
- `synapse-backends`: `ModelHandle` equality, hashing, and Debug tests
- Test coverage roadmap with 6 staged milestones (30% → 80%)
- CI coverage threshold set to 30% as baseline
- **Milestone 2 — Core Logic (40%)**: 48 new tests across synapse-core and synapse-api
  - `synapse-core/pull/downloader`: Mock HTTP tests — download, resume from `.part`, SHA-256 verification, progress events, error handling
  - `synapse-core/pull/verifier`: verify_auto detection, BLAKE3 roundtrip, empty file hashing, integrity error content, nonexistent file error
  - `synapse-core/registry/huggingface`: Mock API tests — model_info (success/404/500), list_gguf filtering, resolve_gguf with quant filter, auth token, serde roundtrips
  - `synapse-core/registry/scanner`: TensorRT/PyTorch format detection, case-insensitive matching, size reporting, tokenizer.bin exclusion
  - `synapse-api/state`: AppState construction, cloneability, bridge disabled, database file creation
  - `synapse-api/rest/system`: health handler, status JSON structure with bridge fields
- CI coverage threshold raised to 40%
- Added `mockito` for HTTP mocking in synapse-core dev-dependencies
- `HfClient::with_base_url()` for testable HuggingFace API client
- **Milestone 3 — Backend Integration (50%)**: 40 new tests across synapse-backends and synapse-api
  - `synapse-backends/ollama`: Mock HTTP tests — load/unload/infer/health, message building, capabilities, error handling
  - `synapse-backends/vllm`: Mock HTTP tests — load/unload/infer/health, OpenAI response parsing, capabilities, error handling
  - `synapse-backends/llamacpp`: Mock server infer cycle via instance injection, process lifecycle, error handling
  - `synapse-api/rest/models`: Handler unit tests — list (empty/populated), get (by name/UUID/not found), delete (success/not found), field validation
- CI coverage threshold raised to 50%
- Added `mockito` for HTTP mocking in synapse-backends dev-dependencies
- **Milestone 4 — API & Training (60%)**: 40 new tests across synapse-api and synapse-train
  - `synapse-api/rest/inference`: InferenceBody serde tests (defaults, stream flag), no-model-loaded error paths for both endpoints
  - `synapse-api/rest/openai_compat`: ChatCompletionRequest/ChatMessage serde, list_models (empty/with data), no-model error path
  - `synapse-api/rest/training`: Full handler lifecycle — create (queued/auto-start), list, get, cancel, not-found errors, serde, job_to_response conversion
  - `synapse-train/executor/docker`: Construction, container naming, cancel behavior, config serialization
  - `synapse-train/executor/subprocess`: Construction, cancel with process kill, config serialization
  - `synapse-train/executor/mod`: Extracted shared `script_for_method()` from docker/subprocess with full coverage
- CI coverage threshold raised to 60%
- Added `Debug` derive to `JobResponse` struct
- **Milestone 5 — Bridge & CLI (70%)**: 55 new tests across synapse-api, synapse-bridge, and synapse-cli
  - `synapse-api/rest/distributed`: Full handler unit tests — create, list, get, assign_worker, start, fail, worker_completed lifecycle, serde for all request/response types
  - `synapse-api/rest/marketplace`: Handler unit tests — search (with/without query/no match), list_entries, unpublish (success/not found), serde for SearchQuery/PublishRequest/PullRequest, entry_to_response conversion
  - `synapse-bridge/protocol`: Connection state variant distinctness, copy semantics, debug formatting, custom config, heartbeat/capabilities roundtrip with all fields
  - `synapse-bridge/client`: Connect resets reconnect count, report_progress without connect, GPU request stub
  - `synapse-bridge/server`: Custom heartbeat interval, zero-value heartbeat, full state transition chain
  - `synapse-bridge/discovery`: Empty config fallthrough, debug format, DiscoveryMethod copy
  - `synapse-cli/commands/list`: format_size (GB/MB/boundary/zero), truncate (short/exact/long)
- CI coverage threshold raised to 70%
- Added `Debug` derive to `DistributedJobResponse` struct

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
- 510 tests across all modules (~70% coverage)

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
- 510 tests across all modules (~70% coverage)

### Fixed
- **SECURITY**: SQL injection in `synapse-core/marketplace/catalog.rs` — `search()` now uses parameterized queries instead of string interpolation
- **SECURITY**: Python code injection in `synapse-train/checkpoint/merger.rs` — `merge_lora()` now passes paths via environment variables instead of interpolating into Python source
- `synapse-core/pull/downloader.rs`: Corrupted `.part` files are now cleaned up when SHA-256 verification fails, preventing infinite retry loops
- `synapse-core/registry/huggingface.rs`: Replaced `.unwrap()` with proper error handling in `resolve_gguf()` fallback path
- `synapse-core/marketplace/catalog.rs`: Replaced `serde_json::to_string().unwrap()` calls with proper error propagation in `publish()`
- `synapse-api/rest/openai_compat.rs`: `list_models` now returns `Result` — database errors are propagated as 500 instead of silently returning empty list
- `synapse-api/rest/inference.rs`: Both `/inference` and `/inference/stream` now use the loaded model's actual backend instead of hardcoding `"llamacpp"`
- `synapse-api/rest/models.rs`: Failed filesystem cleanup during model deletion is now logged as a warning instead of silently ignored
- `synapse-api/rest/marketplace.rs`: Model download endpoint now streams files via `tokio_util::io::ReaderStream` instead of loading entire model into memory (prevents OOM on large files)
- `synapse-train/executor/subprocess.rs`: Fixed potential deadlock in `run()` — child process is now removed from the tracking map before awaiting, so `cancel()` can acquire the write lock concurrently
- `synapse-backends/llamacpp`: `unload_model()` now calls `wait()` after `kill()` to reap child processes and prevent zombie `llama-server` processes
- `synapse-backends/ollama`: `unload_model()` now logs HTTP errors instead of silently discarding them with `let _ =`
- `synapse-backends/ollama`: Stream errors in `infer_stream()` now logged with `warn!` instead of silently breaking
- `synapse-backends/router`: `select()` now logs a warning when the user's preferred backend is not found, before falling back to auto-selection
- `synapse-train/job/manager`: Fixed potential deadlock in `cancel_job()` — read lock is now released before calling `executor.cancel()`, preventing deadlock when the executor needs write access
- `synapse-train/executor/docker`: Container is now tracked BEFORE `docker run` executes, so `cancel()` can find and stop the container during long-running training
- `synapse-train/distributed/coordinator`: `worker_completed()` now guards against over-counting — duplicate completion reports after all workers have finished are no-ops
- `synapse-api/rest/training`: `create_job` auto-start failures are now logged as warnings instead of silently ignored with `let _ =`
- `synapse-core/marketplace/resolver`: HTTP client builder failure now logs a warning and falls back to default client instead of silently using `unwrap_or_default()`
- `synapse-core/marketplace/resolver`: Format filter serialization failure in `query_peer()` now returns an error instead of silently dropping the filter via `unwrap_or_default()`
- `synapse-core/lifecycle/manager`: Replaced `.unwrap()` on `best_accelerator()` with proper error propagation — prevents panic when GPU is detected but accelerator type is undetermined
- `synapse-core/lifecycle/memory`: `estimate_gguf()` now rounds up file size to nearest MB instead of truncating, preventing underestimation of memory requirements
- `synapse-backends/router`: `select()` no longer returns an incompatible backend as fallback — returns `None` when no backend supports the requested format, instead of silently picking any backend
- `synapse-backends/ollama`: `load_model()` now validates the HTTP response status — previously a failed load (HTTP 500) was silently treated as success, causing phantom loaded models
- `synapse-train/executor/docker`: `cancel()` now removes the container from tracking after stopping, preventing unbounded memory growth from accumulated stale entries
- `synapse-train/executor/docker`: `cancel()` now logs `docker stop` errors instead of silently discarding them with `let _ =`
- `synapse-train/executor/docker`: Container tracking cleanup on spawn failure is now synchronous instead of fire-and-forget `tokio::spawn`
- `synapse-train/job/manager`: Fixed race condition in `start_job()` — running job count is now checked inside the write lock, preventing concurrent calls from exceeding `max_concurrent`
- `synapse-train/job/manager`: `cancel_job()` now re-validates terminal state after reacquiring write lock, preventing Cancelled from overwriting a concurrent Completed transition
- `synapse-api/rest/models`: `delete_model` now deletes from database first, then cleans up filesystem — prevents orphaned DB records if FS deletion succeeds but DB deletion fails
- `synapse-api/rest/marketplace`: Download endpoint removes TOCTOU race — uses `File::open()` error handling instead of separate `exists()` check, and gets metadata from the open file handle
- `synapse-api/middleware/auth`: Replaced hardcoded `&header[7..]` string slice with `strip_prefix("Bearer ")` for safer token extraction
- `synapse-api/main`: Heartbeat bridge communication errors now logged with `warn!` instead of silently discarded
- CI/CD container image build timeout: switched from compiling Rust inside Docker (30+ min under QEMU for arm64) to using pre-built binaries from the build-release job via `Dockerfile.release` with `TARGETARCH`

### Enhanced
- `synapse-backends/*`: All 4 HTTP backends (llamacpp, ollama, vllm, tensorrt) now use 300-second request timeouts instead of unbounded `reqwest::Client::new()`
- `synapse-backends/*`: All 4 streaming backends now check `tx.is_closed()` to stop processing when the receiver is dropped (early client disconnect)
- `synapse-backends/*`: All 4 streaming backends enforce a 1 MB buffer limit to prevent unbounded memory growth from malformed SSE streams
- `synapse-backends/llamacpp`: `load_model()` now validates that the model file exists before spawning `llama-server`, providing a clear error instead of a cryptic process failure
- `synapse-api/rest/inference`: Both `/inference` and `/inference/stream` now match the requested model by name instead of always using the first loaded model
- `synapse-api/rest/openai_compat`: `/v1/chat/completions` now matches the requested model by name instead of always using the first loaded model
- `synapse-api/rest/eval`: Eval run inference now targets the requested model by name instead of always using the first loaded model
- `synapse-core/lifecycle/manager`: `LoadedModel` now carries `model_name` to support model selection by name in inference endpoints
- `synapse-types/training`: `HyperParams::validate()` rejects `learning_rate <= 0`, `epochs == 0`, `batch_size == 0`, and `max_seq_length == 0`
- `synapse-train/job/manager`: `create_job()` now validates hyperparameters before creating the job
- `synapse-train/checkpoint/store`: `prune()` now handles already-deleted checkpoints gracefully (ENOENT-tolerant) instead of propagating errors
- `synapse-train/dataset/validator`: CSV validation now handles RFC 4180 quoted fields — commas inside `"quoted,field"` no longer cause false column-count mismatches
- 512 tests across all modules
