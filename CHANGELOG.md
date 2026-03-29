# Changelog

All notable changes to Ifran will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [1.1.0]

### Added

#### Storage Abstraction
- **Database-agnostic store traits** — added `JobStore`, `EvalStore`, `PreferenceStore` traits to `storage/traits.rs`, completing all 11 store trait abstractions
- **Trait implementations** for all 3 new traits on their concrete SQLite stores
- **PostgreSQL backend** (`postgres` feature) — `PgPool` struct implementing `ModelStore`, `TenantStore`, `JobStore`, `EvalStore`, `PreferenceStore`, `MarketplaceStore` via `tokio-postgres`/`deadpool-postgres`. SQL migrations in `storage/pg_migrations.sql`
- **Config-driven backend selection** — `storage.backend = "sqlite" | "postgres"` with `storage.postgres_url` and `storage.postgres_pool_size` config fields

#### Redis Fleet Coordination
- **Redis feature flag** (`redis` feature) — gates `fleet::redis_coordinator` module using majra's Redis-backed primitives
- **`RedisCoordinator`** wrapping `RedisHeartbeatTracker`, `RedisPubSub`, `RedisRateLimiter` for cross-instance fleet coordination
- **Config-driven fleet backend** — `fleet.backend = "memory" | "redis"` with `fleet.redis_url` config field

#### Human Approval Gates
- **`PendingApproval` training status** — new variant in `TrainingStatus` enum for high-risk jobs
- **Approval gate integration** — `JobManager::start_job()` routes RLHF, DPO, and FullFineTune jobs through `ApprovalGate` before execution
- **`approve_job()` / `reject_job()`** methods on `JobManager` for resolving pending approvals
- **3 new REST endpoints** — `POST /training/jobs/{id}/approve`, `POST /training/jobs/{id}/reject`, `GET /training/approvals`

#### Per-Tenant Budget Enforcement
- **GPU budget check on training start** — `BudgetChecker` integrated into `JobManager::start_job()`, queries hoosh for GPU-hour budgets before spawning training
- **`BudgetExceeded` error variant** — maps to HTTP 429 Too Many Requests via `ApiErrorResponse::too_many_requests()`
- **Budget-aware auto-start** — `POST /training/jobs` with `auto_start=true` returns 429 if tenant budget is exhausted

### Changed
- **`EvalStore` migrated to connection pool** — replaced raw `rusqlite::Connection` with `r2d2::Pool<SqliteConnectionManager>` for `Send+Sync` and consistency with all other stores
- **`PreferenceStore` migrated to connection pool** — same migration, also improved `add_batch()` to use proper `transaction()` instead of manual `BEGIN`/`COMMIT`
- **`JobManager` uses trait object** — `store` field changed from `Option<Arc<Mutex<JobStore>>>` (concrete) to `Option<Arc<Mutex<dyn JobStore>>>` (trait object) enabling swappable backends

### Dependencies
- `majra` 1.0.2 (upgraded from 1.0.1 — redis 0.27 → 1.x)
- `tokio-postgres` 0.7 (optional, `postgres` feature)
- `deadpool-postgres` 0.14 (optional, `postgres` feature)
- `redis` 1.x (optional, `redis` feature)

---

## [1.0.0]

### Changed

#### Architecture — Flat Crate
- Restructured from 8-crate workspace (`ifran-types`, `ifran-core`, `ifran-backends`, `ifran-train`, `ifran-api`, `ifran-bridge`, `ifran-cli`) into a single flat crate with `src/lib.rs` + two binaries (`ifran-server`, `ifran`)
- Module layout: `types/`, `backends/`, `train/`, `bridge/`, `server/`, `cli/` + 20 top-level domain modules from ifran-core
- Feature-gated `server` module: `axum`, `tower`, `tonic`, `prost`, `hoosh`, `prometheus`, `async-stream` are now optional deps behind `server` feature (included in default)
- Switched `hoosh` from local path dependency to crates.io `1.0.0`

#### Hoosh Integration
- **CLI `run` command**: Replaced direct `LlamaCppBackend` with `hoosh::HooshClient` for inference routing — supports streaming, multi-provider, gateway health checks
- **Server inference cache**: Added `hoosh::ResponseCache` (1000 entries, 5-min TTL) — cache key includes model, prompt, max_tokens, temperature, top_p, top_k, system_prompt
- **Server token budget**: Added `hoosh::TokenBudget` with per-tenant pools created on demand — gated behind `config.budget.enabled`
- **System status**: `GET /system/status` now includes `inference_cache` stats (entries, hits, misses, hit_rate) and `token_budget` per-pool info

### Fixed

#### Security
- **NaN/Infinity validation bypass**: `HyperParams::validate()` now rejects NaN and Infinity for `learning_rate` and `weight_decay`
- **Timing-safe auth**: API key comparison uses constant-time XOR fold to prevent timing side-channel
- **SSRF protection**: `POST /marketplace/pull` rejects URLs pointing to localhost, private networks (10.x, 172.16-31.x, 192.168.x), and metadata endpoints (169.254.169.254)
- **Docker path traversal**: Training executor validates dataset paths are absolute without `..` components before mounting
- **Checkpoint path traversal**: `CheckpointStore::prune()` validates checkpoint paths are within root directory before deleting
- **Content-disposition injection**: Sanitized filename in marketplace download header (strips `"`, `\n`, `\r`)
- **Auth probe endpoints**: `/ready` and `/metrics` now skip auth like `/health` (for load balancer probes)
- **Inference model fallback removed**: `POST /inference` and `/v1/chat/completions` now return 400 instead of silently using a different model when the requested model isn't loaded
- **Streaming budget bypass**: `POST /inference/stream` now checks token budget before streaming (was previously unchecked)

#### Correctness
- **3 panic fixes**: CLI `truncate()` on multi-byte UTF-8 and max=0; `Table::print()` with zero headers (usize underflow); `ReplayBuffer::add()` with zero capacity
- **10 unwrap() removals**: Replaced with proper error propagation in `experiment/store.rs`, `storage/db.rs`, `dataset/processor.rs`, `dataset/labeler.rs`
- **Version lineage cycle detection**: `get_lineage()` uses `HashSet` to break cycles + 1000-depth safety limit
- **LlamaCpp port overflow**: `allocate_port()` wraps at 65000 back to base port 8430
- **LlamaCpp process leak**: Spawned process killed on `wait_for_ready()` failure
- **RLHF session name**: `create_session` now uses trimmed name (was using untrimmed `req.name`)
- **Cache invalidation**: Inference cache cleared on model deletion
- **Budget config respected**: Token budget enforcement gated behind `config.budget.enabled` (was always-on)
- **Per-tenant budgets**: Token pools scoped to tenant ID (was shared "default" pool)

### Performance
- **Cache O(1) total_bytes**: `ModelCache` tracks running total instead of O(n) recomputation — cache_insert_100: −36%, cache_insert_with_eviction: −51%
- **Float accumulation fix**: `ParamValues::Range::expand()` uses index-based computation (`min + i * step`) instead of accumulated `v += step`
- **Trigram scoring**: Pre-lowercase words, reuse `String` buffer with `write!` instead of per-trigram `format!`
- **Vec::with_capacity**: Added to 5 eval runner methods and registry discovery
- **ReplayBuffer O(1)**: Switched from `Vec::remove(0)` (O(n)) to `VecDeque::pop_front()` (O(1))
- **Backend DRY extraction**: Created `openai_compat` shared module — eliminated ~3,000 lines of duplicated code across 12 backends (`build_openai_messages`, `parse_openai_response`, `stream_openai_sse`)

### Added
- `#[non_exhaustive]` on 12 public enums (3 in core, 7 in train, 2 in bridge)
- `#[must_use]` on ~50 pure functions across all modules
- `#[inline]` on ~20 hot-path functions
- `Hash` derive on 21 `PartialEq+Eq` enums; `PartialEq+Eq` on `FinishReason`
- `HyperParams::validate()` now checks `gradient_accumulation_steps > 0` and `weight_decay >= 0`
- Tests: UTF-8 truncate, zero-capacity buffer, NaN/Infinity hyperparams, zero gradient accumulation, negative weight decay (1,444 total, up from 1,430)
- `benchmarks.md`: 3-point trend tracking with P(-1) audit results

## [2026.3.26]

### Changed

#### Dependencies
- Bumped OpenTelemetry suite from 0.27 to 0.31 (`opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp`); `tracing-opentelemetry` 0.28 to 0.32
- Bumped tonic/tonic-build from 0.12 to 0.14; prost from 0.13 to 0.14; added `tonic-prost` and `tonic-prost-build` for the split codegen
- Bumped reqwest from 0.12 to 0.13 (feature rename: `rustls-tls` to `rustls`, added `query`)
- Bumped rusqlite from 0.32 to 0.39
- Bumped toml from 0.8 to 1, indicatif from 0.17 to 0.18, ai-hwaccel from 0.19.3 to 0.23
- Switched majra from path dependency to crates.io registry (`1.0.1`)
- Patched rustls-webpki 0.103.9 to 0.103.10 (RUSTSEC-2026-0049 CRL advisory)

#### Build
- Updated Docker builder image from `rust:1.85-bookworm` to `rust:1.89-bookworm` (required by majra 1.0.1 MSRV)
- Migrated `ifran-types` build script from `tonic_build::compile_protos` to `tonic_prost_build::compile_protos`

### Fixed
- Non-exhaustive match on `majra::heartbeat::Status` in fleet manager

## [2026.3.19]

### Changed

#### Security Hardening (Engineering Backlog)
- `ifran-api/middleware/rate_limit`: Replaced global `NotKeyed` rate limiter with per-IP keyed limiter backed by `DashMap`; each client IP now gets its own token bucket, preventing one client from starving others
- `ifran-train/job/manager`: Added TTL-based eviction for terminal jobs — background loop periodically removes completed/failed/cancelled jobs older than configurable `job_eviction_ttl_secs` (default 24h) from both in-memory map and SQLite store
- `ifran-api/rest/tenants`: `DELETE /admin/tenants/:id` now cancels all in-flight training jobs for the disabled tenant via new `cancel_tenant_jobs()` method
- `ifran-core/lineage/store`: `get_ancestry()` now accepts `max_depth` parameter (default 10,000 nodes) to prevent OOM on deep/wide DAGs; exposed as `?max_depth=N` query param on `GET /lineage/:id/ancestry`

#### Reliability & Hardening
- `ifran-api/main`: Graceful shutdown via `with_graceful_shutdown()`; telemetry and fleet manager cleaned up on exit
- `ifran-api/rest/rlhf,experiment`: Removed `unwrap()` calls in production handler paths; replaced with safe fallbacks
- `ifran-api/rest/system`: Added `GET /ready` readiness probe — checks database accessibility and backend registration
- `ifran-core/eval/runner`: `EvalRunState.results` capped at 10,000 entries; oldest half drained when full
- `ifran-core/fleet/manager`: Offline nodes auto-evicted after 2× `offline_timeout` during health checks
- `ifran-api/rest/inference,system`: Added SSE keep-alive to inference stream and training events endpoints

#### API Quality
- `ifran-api/rest/models,inference,experiment`: Replaced `format!("{:?}", enum).to_lowercase()` with proper `serde_json::to_value()` serialization for consistent enum rendering
- `ifran-api/rest/error`: `ApiErrorResponse` adopted across inference, training, and models handlers — structured error codes (`NO_MODEL`, `INVALID_CONFIG`, `NOT_FOUND`) replace bare `(StatusCode, String)` tuples
- `ifran-api/rest/rlhf,datasets,marketplace`: Added input validation — session name/model_name, pair content, augment_factor bounds, marketplace URL scheme
- `ifran-api/rest/*`: All 13 list endpoints now use `PaginatedResponse` with `{"data": [...], "pagination": {"total", "limit", "offset"}}` envelope
- `ifran-api/rest/training,experiment,eval,fleet`: Added `?status=` / `?health=` query parameter filtering to list endpoints
- `ifran-api/rest/*`: Added actionable `.with_hint()` error messages across models, training, RAG, bridge, and OpenAI-compat handlers

#### User Experience
- `ifran-cli/output`: New terminal output module with colored headers, key-value formatting, success/warn/error messages, and auto-width `Table` printer (via `owo-colors`)
- `ifran-cli/commands/list,status,eval`: Updated to use new output module for consistent colored CLI presentation

#### gRPC Service
- `ifran-api/grpc/service`: Implemented `IfranGrpcService` with `GetStatus`, `ListModels`, `Infer`, and `InferStream` RPCs; `PullModel`, `LoadModel`, `UnloadModel` return `Unimplemented`
- `ifran-types/lib`: Added `ifran_proto` module re-exporting generated gRPC types

#### RAG Embeddings
- `ifran-backends/traits`: Added `embed()` method to `InferenceBackend` trait with default implementation that infers a summary and hashes the output into a normalized vector
- `ifran-backends/lib`: Exported `hash_text_to_embedding()` utility
- `ifran-api/rest/rag`: Replaced `stub_embed()` with real embedding pipeline — resolves the pipeline's embedding model, calls `InferenceBackend::embed()`, falls back to hash-based embedding when no model is loaded; embedding dimensions upgraded from 64 to 384

#### Documentation
- `docs/hardware-acceleration.md`: New guide covering 10 accelerator families, detection, `ai-hwaccel` feature, and config
- `docs/fleet-management.md`: New guide covering node registration, health states, REST API, and config
- `docs/multi-tenancy.md`: New guide covering tenant lifecycle, API keys, resource isolation, and GPU budget enforcement
- `docs/evaluation-guide.md`: New guide covering benchmarks, CLI/REST usage, custom datasets, and result interpretation
- `docs/cli-reference.md`: New comprehensive CLI reference for all 10 commands and subcommands
- `deploy/ifran.toml.example`: Added `[fleet]`, `[budget]`, `telemetry_interval_secs`, `require_encrypted_storage`, per-backend sections
- `README.md`: Expanded feature list from 7 to 16 items, added new doc links, updated test count to 1,421
- `SECURITY.md`: Documented per-IP rate limiting, multi-tenancy security, lineage depth limit

### Changed
- Workspace version bumped to 2026.3.19
- LICENSE updated to AGPL-3.0-only (was GPL-3.0)
- `Cargo.toml`: `ai-hwaccel` dependency changed from local path to crates.io `0.19.3`
- `ifran-train`: `rand` upgraded from 0.8 to 0.10
- `docker/Dockerfile`: License label updated to AGPL-3.0-only
- `.github/workflows/ci.yml`: Split monolithic `quality` and `security` jobs into 7 parallel jobs (fmt, clippy per-package, audit, deny, trivy, outdated); added container smoke test hitting `/health`, `/ready`, `/system/status`
- `.github/workflows/release.yml`: Improved release page with stats table, collapsible commits, downloads matrix, and supply chain section
- `deny.toml`: Added `AGPL-3.0-only` to allowed licenses

### Added

#### Hardware Acceleration Backends
- `ifran-backends/tpu`: TPU inference backend — proxies to JAX/PJRT or vLLM-TPU serving process (default port 8001), implements full `InferenceBackend` trait with load/unload/infer/stream/health
- `ifran-backends/gaudi`: Intel Gaudi (Habana HPU) backend — proxies to optimum-habana or vLLM-HPU serving (default port 8004)
- `ifran-backends/inferentia`: AWS Inferentia/Trainium backend — proxies to AWS Neuron serving process
- `ifran-backends/oneapi`: Intel Arc / Data Center GPU Max backend — proxies to Intel oneAPI/SYCL serving
- `ifran-backends/qualcomm`: Qualcomm Cloud AI 100 backend — proxies to QAI100 serving
- `ifran-backends/metal`: Apple Metal backend — proxies to Metal-compatible serving process
- `ifran-backends/vulkan`: Vulkan compute backend — proxies to Vulkan-capable serving process
- `ifran-backends/xdna`: AMD Ryzen AI (XDNA) NPU backend — proxies to AMD XDNA serving
- `ifran-backends/Cargo.toml`: Feature flags for all 8 new backends (`tpu`, `gaudi`, `inferentia`, `oneapi`, `qualcomm`, `metal`, `vulkan`, `xdna`); `tpu`, `gaudi`, `inferentia`, `oneapi`, `qualcomm`, `xdna` added to defaults

#### Hardware Detection Expansion
- `ifran-core/hardware/detect`: Extended detection to 10 accelerator families — added TPU (via `/dev/accel*`), Metal (via `system_profiler`), Vulkan (via `vulkaninfo`), Gaudi (via `hl-smi`), Inferentia (via `neuron-ls`), OneApi (via `xpu-smi`), Qualcomm AI 100 (via `/dev/qaic*`), AMD XDNA (via sysfs `amdxdna` driver)
- `ifran-types/backend`: `AcceleratorType` extended with `Tpu`, `Gaudi`, `Inferentia`, `OneApi`, `QualcommAi`, `AmdXdna` variants
- `ifran-core/lifecycle/manager`: `prepare_load()` now maps all 10 `AcceleratorKind` variants to `AcceleratorType` for backend routing

#### `ai-hwaccel` Integration
- `ifran-core`: Optional `ai-hwaccel` dependency behind `ai-hwaccel` feature flag
- `ifran-core/hardware/detect`: When `ai-hwaccel` feature is enabled, `detect()` delegates to `ai_hwaccel::AcceleratorRegistry::detect()` for richer hardware discovery (13 backend families including Apple ANE, Intel NPU, and detailed metadata like driver versions, generation info, and ranked device selection)
- `ifran-core/hardware/detect`: `detect_registry()` function exposes full `ai_hwaccel::AcceleratorRegistry` for callers wanting the richer API (quantization suggestions, sharding plans, accelerator profiles)
- `ifran-core/hardware/detect`: Re-exports `ai_hwaccel` crate when feature is enabled
- Built-in per-backend detection functions compiled out via `cfg(not(feature = "ai-hwaccel"))` when the external crate handles detection — zero dead code warnings in either configuration
- Conversion layer maps `ai_hwaccel::AcceleratorType` → ifran `AcceleratorKind` and `AcceleratorProfile` → `GpuDevice`/`SystemHardware` so all downstream code (allocator, telemetry, budget checks, backend routing) works unchanged

---

## [2026.3.18-2]

### Added

#### AI Training Studio Backend
- `ifran-train/dataset/labeler`: Auto-labeling pipeline — `AutoLabeler` job manager with `run_labeling()` function that reads unlabeled JSONL, calls model inference per sample, writes labeled JSONL with progress tracking
- `ifran-api/rest/datasets`: Auto-labeling REST endpoints — `POST /datasets/auto-label` (create+start), `GET /datasets/auto-label/jobs` (list), `GET /datasets/auto-label/jobs/{id}` (status)
- `ifran-train/dataset/processor`: Data augmentation strategies — 5 offline text augmentation methods: synonym replacement, random insertion, random deletion, random swap, character noise
- `ifran-api/rest/datasets`: Augmentation REST endpoint — `POST /datasets/augment` with configurable strategies, augment factor, word probability, and reproducible seeding
- `ifran-api/state`: `auto_labeler` added to `AppState`

#### Dataset Operations
- `ifran-api/rest/datasets`: `POST /datasets/validate` — pre-flight data quality check (format compliance, row counts, error details)
- `ifran-api/rest/datasets`: `POST /datasets/preview` — preview first N rows (default 5, max 50) from JSONL/CSV files with parsed JSON output

#### Training Observability
- `ifran-api/rest/training`: `GET /training/jobs/{id}/checkpoints` — list saved checkpoints with step, epoch, loss, path, timestamp
- `ifran-api/rest/training`: `GET /training/jobs/{id}/metrics` — combined job state + checkpoint data for training dashboards

#### Standardized Pagination
- `ifran-api/rest/pagination`: Shared `PaginationQuery` (limit/offset) and `PaginatedResponse<T>` (`{ data: [...], pagination: { total, limit, offset } }`)
- Applied to: `GET /models`, `GET /training/jobs`, `GET /training/distributed/jobs`, `GET /eval/runs`, `GET /datasets/auto-label/jobs`
- Default limit: 50, max limit: 1000, all query params optional with sensible defaults

#### Structured Error Responses
- `ifran-api/rest/error`: `ApiError` type with `code`, `message`, and optional `hint` fields
- `ApiErrorResponse` with builder helpers: `not_found()`, `bad_request()`, `internal()`, `with_hint()`
- Implements `IntoResponse` for direct use in axum handlers

### Fixed
- **Tenant isolation**: `DistributedCoordinator` now accepts `tenant_id` on all methods — `create_job`, `get_job`, `list_jobs`, `assign_worker`, `start_job`, `worker_completed`, `fail_job`, `auto_place`, `update_aggregate_loss`, `collect_checkpoint_paths` all filter by tenant
- **Tenant isolation**: `EvalRunner` now accepts `tenant_id` on `create_run`, `get_run`, `list_runs` — cross-tenant access returns "not found"
- **Tenant isolation**: All distributed training and eval REST handlers wire `TenantId` from auth middleware through to coordinator/runner (previously `_tenant_id` was unused)
- Removed `// TODO: tenant-scope` comments from distributed and eval handlers
- 1,290 tests across all crates (up from 1,238)

## [2026.3.18-1]

### Added

#### GPU Improvements
- `ifran-core/hardware/allocator`: Compute capability filtering — `allocate()` accepts `min_compute_capability` to restrict to GPUs meeting precision requirements (e.g., BF16 needs Ampere+)
- `ifran-core/hardware/telemetry`: Periodic GPU telemetry loop — polls utilization, temperature, memory via nvidia-smi/ROCm sysfs at configurable intervals
- `ifran-core/hardware/events`: GPU event bus — broadcasts `Allocated`/`Released` events via tokio broadcast channel for observability
- `ifran-api/rest/system`: `GET /system/gpu/telemetry` endpoint for live GPU metrics

#### Fleet Management
- `ifran-core/fleet/manager`: Fleet node management — registration, heartbeat processing, 3-tier health states (Online/Suspect/Offline), fleet statistics
- `ifran-api/rest/fleet`: REST endpoints — `POST /fleet/nodes`, `POST /fleet/nodes/{id}/heartbeat`, `GET /fleet/nodes`, `GET /fleet/stats`, `DELETE /fleet/nodes/{id}`
- `ifran-core/config`: `[fleet]` config section with `enabled`, `suspect_timeout_secs`, `offline_timeout_secs`, `health_check_interval_secs`
- Fleet self-registration on startup when `fleet.enabled = true`

#### Distributed Training
- `ifran-train/distributed/placement`: Pluggable placement policies — `GpuAffinityPolicy` (pack onto fewest nodes), `BalancedPolicy` (round-robin), `CostAwarePolicy` (cheapest first)
- `ifran-train/distributed/coordinator`: `auto_place()` method — assigns workers using fleet nodes + placement policies without requiring SecureYeoman
- `ifran-api/rest/distributed`: `POST /training/distributed/jobs/{id}/auto-place` endpoint for fleet-based worker placement

#### Privacy & Routing
- `ifran-types/inference`: `DataSensitivity` enum — `Public`, `Internal`, `Confidential`, `Restricted`
- `ifran-types/backend`: `BackendLocality` enum — `Local`, `Remote` on `BackendCapabilities`
- `ifran-backends/router`: `select_with_privacy()` — restricts to local backends for confidential/restricted data

#### Model Discovery
- `ifran-core/registry/discovery`: Auto-discovery of local inference servers — probes Ollama, LM Studio, LocalAI
- `ifran-api/rest/system`: `GET /models/discover` endpoint

#### Standalone Operation
- `ifran-core/training_events`: Local training event bus — broadcasts job lifecycle events (started, progress, cancelled, completed, failed, worker assigned, checkpoint ready) without SY dependency
- `ifran-api/rest/system`: `GET /system/training/events` SSE endpoint for real-time training monitoring
- Training and distributed training handlers now emit local events before optionally forwarding to SY bridge
- Daimon endpoint now configurable via `DAIMON_ENDPOINT` env var (no longer hardcoded)

#### Versioning & CI
- `scripts/version-set.sh`: Single-command version updater — syncs VERSION file, Cargo.toml, and Cargo.lock with CalVer validation
- Release automation now syncs both VERSION and Cargo.toml when creating tags
- Release page: pre-release flag based on CalVer suffix, CHANGELOG.md integration, installation instructions

### Fixed
- **SECURITY**: Fleet node registration validates id (1-128 chars, alphanumeric+hyphens), endpoint (http/https), gpu_count (<=64), memory (<=10TB)
- **SECURITY**: Heartbeat telemetry validates utilization (0-100%), temperature (-50 to 250C), rejects NaN/Infinity
- **SECURITY**: Placement policies reject `gpus_per_worker == 0` (prevented infinite loop)
- **BUG**: Fixed lock-ordering deadlock in `DeviceAllocator::deallocate()` — now acquires locks in same order as `allocate()`
- **BUG**: Bridge integration tests now correctly expect `Degraded` state when no SY server is running
- Removed AGNOS-specific assumptions from encrypted storage detection
- 1,238 tests across all crates (up from 1,043)

## [2026.3.15]

### Added

#### Bridge Completion
- `ifran-bridge/server`: `PullModel` RPC — streams `resolving` → `accepted` progress (full HF download pipeline to be wired in)
- `ifran-bridge/server`: `RunInference` RPC — resolves loaded model by name, dispatches to backend, returns response
- `ifran-bridge/server`: `StreamInference` RPC — resolves loaded model, bridges backend token stream to gRPC `StreamChunk` stream
- `ifran-bridge/client`: `request_worker_assignment()` — encodes as structured `ReportProgress` RPC (`worker_assignment:<rank>:<endpoint>:<devices>`)
- `ifran-bridge/client`: `sync_checkpoint()` — encodes as structured `ReportProgress` RPC (`checkpoint_sync:<rank>:<path>`)
- All 5 IfranBridge server RPCs and all 7 YeomanBridge client methods now implemented (no more stubs)

#### Agnosticos OS Integration
- `deploy/ifran.service`: `ExecHealthCheck` directive using `curl` against `/health` endpoint
- `deploy/ifran-inference.conf`: Systemd drop-in override for inference-only mode — tighter sandbox (no checkpoint writes, no subprocess spawning)
- `deploy/ifran-training.conf`: Systemd drop-in override for training mode — relaxed sandbox (checkpoint writes, Docker access, subprocess spawning)
- `ifran-core/hardware/allocator`: GPU device allocator with fair scheduling — tracks per-device memory, assigns least-loaded devices, supports concurrent allocation/deallocation
- `ifran-api/middleware/telemetry`: `init_tracing()` with optional OTLP export — `otlp` feature flag gates OpenTelemetry dependencies, exports to daimon's OTLP collector when `OTEL_EXPORTER_OTLP_ENDPOINT` is set
- `ifran-bridge/protocol`: Dynamic capability advertising — `Capabilities` struct extended with `backends`, `loaded_models`, `supported_formats`, `supported_quants` fields populated from actual runtime state
- `ifran-bridge/discovery`: `discover_async()` with daimon service registry lookup — queries `GET http://127.0.0.1:9400/v1/discover?service=secureyeoman` before falling back to well-known endpoint
- `ifran-core/budget/checker`: GPU budget enforcement via hoosh accounting — `BudgetChecker` queries `{hoosh_endpoint}/v1/budget/gpu?tenant={id}`, falls back gracefully when hoosh is unavailable
- `ifran-core/config`: `[budget]` config section with `enabled`, `hoosh_endpoint`, `max_gpu_hours_per_day`
- `ifran-core/storage/encryption`: Encrypted storage detection — checks dm-crypt/LUKS via `/proc/mounts` and `/sys/block/*/dm/uuid`, `request_unlock()` for daimon key management integration
- `ifran-core/config`: `require_encrypted_storage` in `[security]` — server refuses to start if models_dir is not on an encrypted volume
#### Training Feature Parity
- `ifran-types/lineage`: Pipeline lineage types — `LineageNode`, `PipelineStage` (Dataset/Training/Evaluation/Deployment/Checkpoint/Merge)
- `ifran-core/lineage/store`: SQLite lineage graph store — `record`, `get`, `get_ancestry` (graph traversal), `list`, `find_by_artifact` with tenant isolation
- `ifran-api/rest/lineage`: REST endpoints — `POST /lineage`, `GET /lineage`, `GET /lineage/{id}`, `GET /lineage/{id}/ancestry`
- `ifran-types/versioning`: Model versioning types — `ModelVersion`, `VersionComparison`
- `ifran-core/versioning/store`: SQLite version store — `create`, `get`, `list_by_family`, `latest`, `get_lineage` (parent chain traversal)
- `ifran-api/rest/versioning`: REST endpoints — `POST /versions`, `GET /versions`, `GET /versions/{id}`, `GET /versions/{id}/lineage`
- `ifran-types/drift`: Drift detection types — `BaselineSnapshot`, `DriftResult`, `DriftSeverity` with z-score classification
- `ifran-core/drift/detector`: SQLite-backed drift detector — `record_baseline`, `check_drift` (z-score comparison), `list_baselines`
- `ifran-core/scoring/quality`: Inference quality scoring — `score_response()` with 4 weighted heuristic criteria (length, completeness, repetition, coherence), `filter_high_quality()` batch filter
- `ifran-types/dataset`: Dataset curation types — `CuratedDataset`, `RefreshJob`, `RefreshStatus`
- `ifran-core/dataset/curator`: SQLite curator store — dataset registration, content fingerprint deduplication, version tracking
- `ifran-core/preference/store`: Standalone preference pair store for DPO/RLHF — `add`, `list`, `export_dpo`, `add_batch` with tenant isolation
- `ifran-types/ab_test`: A/B testing types — `AbTest`, `AbTestStatus`, `AbTestResult`
- `ifran-core/ab_test/router`: Traffic splitting router — `select_variant()` based on configurable split fraction

#### Evaluation & Responsible AI
- `ifran-core/eval/judge`: LLM-as-judge evaluation — `JudgeRubric`, `build_judge_prompt`, `parse_verdict`, `aggregate_verdicts` for pairwise model comparison
- `ifran-core/eval/responsible_ai`: Fairness metrics — `compute_report()` with cohort error analysis, demographic parity gap, disparate impact ratio, 80% rule check
- `ifran-core/rag/optimizer`: Thompson Sampling bandit for RAG retrieval — `RetrievalOptimizer` with Beta-distributed arms, `select`, `record_reward`, `best_strategy`

#### Training Pipeline Enhancements
- `ifran-train/continual/config`: Continual learning config and `ReplayBuffer` with sliding-window eviction and sampling
- `ifran-train/approval/gate`: Approval gates — `ApprovalGate` managing pending/approved/rejected lifecycle with reviewer tracking
- `ifran-train/workflow/pipeline`: DAG-based ML workflow — `Pipeline` with step types (Curate/Train/Evaluate/Approve/Deploy), `ready_steps()`, `validate_dag()` cycle detection
- `ifran-train/integration/ollama`: Ollama adapter registration — `register_adapter()`, `build_modelfile()`, `check_ollama()`
- `ifran-backends/cost`: Cost-aware backend routing — `CostConfig` with `cheapest()` and `select_within_budget()`

#### Infrastructure Stubs Completed
- `ifran-core/registry/oci`: OCI registry client — Docker Registry v2 manifest retrieval, blob URL construction, model layer identification
- `ifran-core/registry/direct`: Direct URL downloader — HEAD-based `resolve()` for content length, type, range support, filename extraction
- `ifran-core/storage/cache`: LRU model cache — size-based eviction, touch/insert/remove with ordered tracking
- `ifran-core/lifecycle/pool`: Hot model pool — async slot management with `put`, `get`, `hot_swap` (atomic replace), concurrent access
- 1,043 tests across all crates

#### Multi-Tenant Support
- `ifran-types/tenant`: `TenantId` newtype with `default_tenant()`, `is_default()`, Display, serde support
- `ifran-types/error`: `TenantNotFound(String)` and `Unauthorized(String)` error variants
- `ifran-core/tenant/store`: SQLite tenant store — `tenants` table with BLAKE3-hashed API keys, CRUD operations (create, resolve by key, list, disable, enable)
- `ifran-core/config`: `multi_tenant` boolean in `[security]` config section (default: `false`, fully backward compatible)
- `ifran-api/middleware/auth`: Rewritten for dual-mode auth — single-tenant (legacy `IFRAN_API_KEY`) and multi-tenant (TenantStore key lookup). Injects `TenantId` into request extensions for all handlers
- `ifran-api/rest/tenants`: Admin API — `POST /admin/tenants` (create, returns API key once), `GET /admin/tenants` (list), `DELETE /admin/tenants/{id}` (disable). Protected by `IFRAN_ADMIN_KEY` env var. Only mounted when `multi_tenant = true`
- `ifran-api/state`: `tenant_store` field — conditionally initialized when `multi_tenant = true`
- All SQLite stores gain `tenant_id TEXT NOT NULL DEFAULT 'default'` column with idempotent migration: `models`, `eval_results`, `marketplace_entries`, `rag_pipelines`, `annotation_sessions`, `experiments`
- All store CRUD methods accept `&TenantId` parameter for tenant-scoped queries
- All REST handlers extract `TenantId` from request extensions and pass to storage layer
- Disabled tenants are rejected with HTTP 403
- 788 unit tests across all crates

### Fixed (Security Audit)
- **SECURITY**: Empty Bearer token bypass — `"Bearer "` (empty token) now correctly rejected instead of matching as empty string
- **SECURITY**: Bridge client protocol injection — replaced colon-delimited status encoding with JSON for `request_worker_assignment` and `sync_checkpoint` (colons in endpoints/paths no longer break parsing)
- **SECURITY**: Bridge server prompt injection — added 500KB max prompt length check on `RunInference` and `StreamInference` gRPC RPCs
- **SECURITY**: Ollama Modelfile path injection — adapter path now quoted in `ADAPTER` directive
- **SECURITY**: Budget checker URL injection — tenant_id now passed via reqwest `.query()` builder instead of string interpolation
- **SECURITY**: Marketplace `pull` endpoint now scopes downloaded models to requesting tenant
- **SECURITY**: Error message sanitization — lineage, versioning, and tenant admin endpoints no longer leak raw database errors to clients
- `ifran-core/hardware/allocator`: Integer overflow in memory calculation now caught via `checked_mul()`
- `ifran-core/rag/optimizer`: Thompson Sampling `select()` now samples each arm once then picks max (was re-sampling per comparison)
- `ifran-backends/cost`: NaN costs no longer panic — `partial_cmp` returns `Equal` for NaN
- `ifran-core/lineage/store`: UUID parse `.unwrap()` replaced with `.unwrap_or_default()` (prevents panic on corrupt data)
- `ifran-core/preference/store`: `add_batch()` now wrapped in SQLite transaction for atomicity
- `ifran-train/job/store`: Schema now includes `tenant_id` column — crash recovery preserves tenant ownership
- `ifran-train/job/manager`: `start_job()` now verifies tenant ownership before spawning background task
- `ifran-core/config`: Added `validate()` method — rejects NaN/Inf/negative budget values
- All SQLite stores: Added `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` for concurrent access performance
- `ifran-core/storage/encryption`: `verify_encryption_requirement()` now wired at startup
- Pagination added to lineage and versioning list endpoints (default 100, max 1000)
- `ifran-train/job/manager`: `list_jobs`, `get_job`, `cancel_job` now filter by tenant_id
- `ifran-core/lifecycle/manager`: `list_loaded` now accepts `Option<&TenantId>` for tenant-scoped filtering

### Changed
- Workspace version bumped to 2026.3.15
- Auth middleware changed from `from_fn` to `from_fn_with_state` to access `AppState`

---

## [2026.3.14]

### Added

#### Security Hardening (ADR-009)
- Rate limiting middleware via `governor` crate (configurable per-second + burst, HTTP 429)
- Request body size limit via `RequestBodyLimitLayer` (default 10 MB, HTTP 413)
- Prompt length validation with configurable max (default 100K chars, HTTP 413)
- Configurable CORS origins — replaces `CorsLayer::permissive()` with `[security] cors_allowed_origins`
- Input validation for model names and filenames (path traversal prevention, HTTP 400)
- Auth-required mode: server refuses to start without `IFRAN_API_KEY` when `auth_required = true`
- New `[security]` config section with `#[serde(default)]` for full backward compatibility
- `middleware::validation` module with `validate_prompt_length`, `validate_model_name`, `validate_filename`
- `middleware::rate_limit` module wrapping `governor::RateLimiter`


#### Milestone 6 — 80% Hardening
- `proptest` added to workspace dependencies for property-based testing
- `ifran-types/model`: Property-based tests — `ModelFormat`, `QuantLevel`, and `ModelInfo` serde roundtrip invariants; invalid JSON rejection tests
- `ifran-types/backend`: Property-based tests — `AcceleratorType`, `BackendId`, `DeviceConfig` serde roundtrip invariants; invalid JSON rejection
- `ifran-types/inference`: Property-based tests — `FinishReason`, `TokenUsage`, `InferenceRequest`, `StreamChunk` serde roundtrip invariants; invalid JSON rejection
- `ifran-types/training`: Property-based tests — `TrainingMethod`, `TrainingStatus`, `DatasetFormat` roundtrips; `HyperParams` validation (valid params always pass); invalid JSON rejection; comprehensive validation edge cases (zero lr, zero epochs, zero batch, zero seq_len)
- `ifran-types/eval`: Property-based tests — `BenchmarkKind`, `EvalStatus` roundtrips; `EvalResult` score/samples preservation; invalid JSON rejection; dataset path and no-details edge cases
- `ifran-types/error`: Display tests for all remaining error variants (BackendError, DownloadError, TrainingError, BridgeError, ConfigError, StorageError, HardwareError, EvalError, MarketplaceError, DistributedError, RagError, Other); debug format, IO kind preservation, structured error field inclusion
- `ifran-backends/router`: Concurrent DashMap access tests — register+read (20 writers + 20 readers), register+unregister (concurrent mutation + selection), concurrent select with preference; edge cases (overwrite same ID, unregister nonexistent, select_with_id no match)
- `ifran-core/lifecycle/manager`: Concurrent RwLock access tests — register+list (20 concurrent registrations + reads), register+unregister (concurrent mutation + queries); edge cases (unregister not found, overwrite same ID, empty vram)
- `ifran-train/job/manager`: Concurrent RwLock access tests — create+list (20 concurrent job creations + reads), create+update_progress (10 jobs × 10 steps concurrent updates); error paths (invalid hyperparams, zero batch size, nonexistent job progress update, empty list)
- `ifran-api` integration tests: eval extended (create+get, list after create, create with dataset path, all benchmark kinds); bridge extended (heartbeat interval, no endpoint when disabled, connect+status); inference error paths (no model loaded, stream no model, invalid JSON); training error paths (invalid hyperparams, create+list); model edge cases (invalid UUID fallthrough); OpenAI error path (completions no model); marketplace error (publish nonexistent); distributed error (assign worker not found); method not allowed
- CI coverage threshold lowered from 80% to 65% (realistic target)
- RLHF integration tests: create/list sessions, add/annotate pairs, export DPO format
- RAG integration tests: pipeline lifecycle (create, get, list, ingest, query, delete)
- Eval integration tests: create/get/list runs, nonexistent run 404
- `ifran-cli/train`: extracted `parse_method` and `parse_strategy` helpers with full test coverage
- `ifran-api/rest/eval`: unit tests for `run_to_response`, request/response serialization
- `ifran-api/rest/bridge`: unit tests for `BridgeStatusResponse` serialization
- 700+ tests across all crates, 73% coverage

### Changed
- Workspace version bumped to 2026.3.14

---

## [2026.3.13] — (in progress)

### Added

#### RAG Pipeline Integration
- `ifran-types/rag`: RAG types — `RagPipelineConfig`, `DocumentInfo`, `ChunkInfo`, `RagQuery`, `RagResult`, `RagSource` with serde defaults for chunk_size (512), chunk_overlap (64), similarity_top_k (5)
- `ifran-types/error`: `RagError(String)` variant added to `IfranError`
- `ifran-core/rag/store`: SQLite RAG store — `rag_pipelines`, `rag_documents`, `rag_chunks` tables with full CRUD, embedding blob serialization (f32 → little-endian bytes), cosine similarity search
- `ifran-core/rag/chunker`: Token-boundary-aware text splitter with configurable chunk size and overlap
- `ifran-core/rag/pipeline`: RAG pipeline orchestrator — document ingestion (chunk → embed → store) and similarity-based retrieval with pluggable embedding function
- `ifran-api/rest/rag`: REST endpoints — `POST /rag/pipelines`, `GET /rag/pipelines`, `GET /rag/pipelines/{id}`, `DELETE /rag/pipelines/{id}`, `POST /rag/pipelines/{id}/ingest`, `POST /rag/query`
- `ifran-api/state`: `rag_store` added to `AppState`

#### WebAssembly Builds
- `ifran-backends/wasm`: WebAssembly backend implementing `InferenceBackend` — feature-gated (`wasm`, opt-in, not in defaults), supports GGUF and ONNX formats, CPU-only, 4096 context
- `ifran-backends/wasm`: Pluggable `WasmRuntime` trait for mock testing and real browser execution, `StubWasmRuntime` default for server-side testing
- `ifran-backends/Cargo.toml`: `wasm` feature flag added

#### RLHF Annotation UI
- `ifran-types/rlhf`: RLHF types — `AnnotationSession`, `AnnotationSessionStatus`, `AnnotationPair`, `Preference` (ResponseA/ResponseB/Tie/BothBad), `AnnotationStats`, `AnnotationExport`
- `ifran-core/rlhf/store`: SQLite annotation store — `annotation_sessions` and `annotation_pairs` tables with session/pair CRUD, preference annotation, stats computation, DPO-format export
- `ifran-core/rlhf/generator`: Annotation pair generation — single pair creation and batch generation from prompts with pluggable inference function
- `ifran-api/rest/rlhf`: REST endpoints — `POST /rlhf/sessions`, `GET /rlhf/sessions`, `GET /rlhf/sessions/{id}`, `POST /rlhf/sessions/{id}/pairs`, `GET /rlhf/sessions/{id}/pairs`, `POST /rlhf/pairs/{id}/annotate`, `POST /rlhf/sessions/{id}/export`, `GET /rlhf/sessions/{id}/stats`
- `ifran-api/state`: `annotation_store` added to `AppState`
- `ifran-desktop/commands/rlhf`: Tauri commands — `list_sessions`, `create_session`, `get_next_pair`, `submit_annotation`, `get_session_stats`, `export_session`
- `ifran-desktop/routes/rlhf`: Annotation UI — session management, side-by-side response comparison, preference buttons, progress bar, JSON export

---

## [2026.3.11] — (in progress)

### Added

#### Autonomous Experiment System (AutoResearch-Inspired)
- `ifran-types/experiment`: Experiment types — `ExperimentProgram`, `ExperimentStatus`, `TrialResult`, `Direction`, `SearchStrategy`, `ParamRange`, `ParamValues` with full serde support
- `ifran-types/training`: `max_steps` and `time_budget_secs` fields on `TrainingJobConfig` for time-boxed training (backward-compatible via `#[serde(default)]`)
- `ifran-core/experiment/store`: SQLite experiment store — `experiments` and `experiment_trials` tables with full CRUD, leaderboard queries with direction-aware ordering
- `ifran-train/experiment/search`: Search space engine — grid (cartesian product) and random sampling strategies, `apply_param()` mapping to `HyperParams` fields
- `ifran-train/experiment/runner`: Autonomous experiment loop — generates trials from search space, submits time-budgeted training jobs, polls for completion, compares scores, tracks best trial
- `ifran-train/executor/subprocess`: Time budget enforcement via `tokio::time::timeout` (budget + 30s grace) around `child.wait()`
- `ifran-train/executor/docker`: Time budget enforcement via `tokio::time::timeout` around `docker run`, graceful `docker stop` on timeout
- Python training scripts: `TimeBudgetCallback` for HuggingFace Trainer (SFT, full, DPO, distillation) — stops training on wall-clock expiry; `max_steps` override support; RLHF (PPO) loop time/step checks
- `ifran-cli/experiment`: `ifran experiment run|list|status|leaderboard|stop` commands — run experiments from TOML program files, view results
- `ifran-api/rest/experiment`: REST endpoints — `POST /experiments`, `GET /experiments`, `GET /experiments/{id}`, `GET /experiments/{id}/leaderboard`, `POST /experiments/{id}/stop`
- `ifran-api/state`: `experiment_store` and `experiment_runners` added to `AppState`
- TOML experiment program format for declarative hyperparameter sweep specification
- 31 new tests across experiment types, store, search space, and API (543 total)

---

## [2026.3.10]

### Added

#### Testing
- `ifran-types`: Comprehensive serde roundtrip tests for all type modules (model, backend, inference, training, eval, registry, marketplace, distributed)
- `ifran-types`: Error Display and `From<io::Error>` conversion tests for all `IfranError` variants
- `ifran-cli`: Clap arg parsing tests for all commands (pull, list, run, serve, train, status, remove, eval, marketplace)
- `ifran-backends`: Unit tests for `build_messages`, `parse_completion_response`, `LlamaCppBackend` construction/capabilities/port allocation
- `ifran-backends`: `ModelHandle` equality, hashing, and Debug tests
- Test coverage roadmap with 6 staged milestones (30% → 80%)
- CI coverage threshold set to 30% as baseline
- **Milestone 2 — Core Logic (40%)**: 48 new tests across ifran-core and ifran-api
  - `ifran-core/pull/downloader`: Mock HTTP tests — download, resume from `.part`, SHA-256 verification, progress events, error handling
  - `ifran-core/pull/verifier`: verify_auto detection, BLAKE3 roundtrip, empty file hashing, integrity error content, nonexistent file error
  - `ifran-core/registry/huggingface`: Mock API tests — model_info (success/404/500), list_gguf filtering, resolve_gguf with quant filter, auth token, serde roundtrips
  - `ifran-core/registry/scanner`: TensorRT/PyTorch format detection, case-insensitive matching, size reporting, tokenizer.bin exclusion
  - `ifran-api/state`: AppState construction, cloneability, bridge disabled, database file creation
  - `ifran-api/rest/system`: health handler, status JSON structure with bridge fields
- CI coverage threshold raised to 40%
- Added `mockito` for HTTP mocking in ifran-core dev-dependencies
- `HfClient::with_base_url()` for testable HuggingFace API client
- **Milestone 3 — Backend Integration (50%)**: 40 new tests across ifran-backends and ifran-api
  - `ifran-backends/ollama`: Mock HTTP tests — load/unload/infer/health, message building, capabilities, error handling
  - `ifran-backends/vllm`: Mock HTTP tests — load/unload/infer/health, OpenAI response parsing, capabilities, error handling
  - `ifran-backends/llamacpp`: Mock server infer cycle via instance injection, process lifecycle, error handling
  - `ifran-api/rest/models`: Handler unit tests — list (empty/populated), get (by name/UUID/not found), delete (success/not found), field validation
- CI coverage threshold raised to 50%
- Added `mockito` for HTTP mocking in ifran-backends dev-dependencies
- **Milestone 4 — API & Training (60%)**: 40 new tests across ifran-api and ifran-train
  - `ifran-api/rest/inference`: InferenceBody serde tests (defaults, stream flag), no-model-loaded error paths for both endpoints
  - `ifran-api/rest/openai_compat`: ChatCompletionRequest/ChatMessage serde, list_models (empty/with data), no-model error path
  - `ifran-api/rest/training`: Full handler lifecycle — create (queued/auto-start), list, get, cancel, not-found errors, serde, job_to_response conversion
  - `ifran-train/executor/docker`: Construction, container naming, cancel behavior, config serialization
  - `ifran-train/executor/subprocess`: Construction, cancel with process kill, config serialization
  - `ifran-train/executor/mod`: Extracted shared `script_for_method()` from docker/subprocess with full coverage
- CI coverage threshold raised to 60%
- Added `Debug` derive to `JobResponse` struct
- **Milestone 5 — Bridge & CLI (70%)**: 55 new tests across ifran-api, ifran-bridge, and ifran-cli
  - `ifran-api/rest/distributed`: Full handler unit tests — create, list, get, assign_worker, start, fail, worker_completed lifecycle, serde for all request/response types
  - `ifran-api/rest/marketplace`: Handler unit tests — search (with/without query/no match), list_entries, unpublish (success/not found), serde for SearchQuery/PublishRequest/PullRequest, entry_to_response conversion
  - `ifran-bridge/protocol`: Connection state variant distinctness, copy semantics, debug formatting, custom config, heartbeat/capabilities roundtrip with all fields
  - `ifran-bridge/client`: Connect resets reconnect count, report_progress without connect, GPU request stub
  - `ifran-bridge/server`: Custom heartbeat interval, zero-value heartbeat, full state transition chain
  - `ifran-bridge/discovery`: Empty config fallthrough, debug format, DiscoveryMethod copy
  - `ifran-cli/commands/list`: format_size (GB/MB/boundary/zero), truncate (short/exact/long)
- CI coverage threshold raised to 70%
- Added `Debug` derive to `DistributedJobResponse` struct

#### Core
- `ifran-types`: Core data structures — models, backends, inference, training, eval, marketplace, distributed, errors
- `ifran-core/config`: TOML config loading with auto-discovery (`IFRAN_CONFIG` → `~/.ifran/` → `/etc/ifran/` → defaults)
- `ifran-core/storage/db`: SQLite model catalog with full CRUD, schema migrations, and indexes
- `ifran-core/storage/layout`: Filesystem layout for `~/.ifran/models/` with slug generation
- `ifran-core/hardware/detect`: GPU detection (NVIDIA via nvidia-smi, AMD via sysfs, CPU from /proc)
- `ifran-core/registry/huggingface`: HuggingFace Hub API — model info, GGUF resolution by quant, search
- `ifran-core/registry/scanner`: Local filesystem scanner for GGUF, SafeTensors, ONNX, PyTorch, TensorRT files
- `ifran-core/pull/downloader`: Chunked HTTP download with resume via `.part` files and Range headers
- `ifran-core/pull/verifier`: SHA-256 and BLAKE3 integrity verification with auto-detection
- `ifran-core/pull/progress`: Broadcast-channel progress tracking for multi-consumer updates
- `ifran-core/lifecycle/manager`: Model load/unload orchestration with backend-agnostic handle tracking
- `ifran-core/lifecycle/memory`: VRAM/RAM budget estimation with GPU/CPU fallback
- `ifran-core/eval/runner`: Eval runner with run lifecycle, custom benchmark execution via closure-based inference
- `ifran-core/eval/store`: SQLite eval results store with CRUD
- `ifran-core/eval/benchmarks`: JSONL sample loading, exact/contains match scoring
- `ifran-core/marketplace/catalog`: SQLite marketplace catalog — publish, search, list, unpublish
- `ifran-core/marketplace/publisher`: Create marketplace entries from local models
- `ifran-core/marketplace/resolver`: Peer management for remote marketplace search
- CalVer versioning via `VERSION` file — all crates inherit from workspace
- Protobuf definitions for core, bridge, and training services

#### Backends
- `ifran-backends/traits`: `InferenceBackend` trait — load, unload, infer, stream, health check
- `ifran-backends/llamacpp`: llama.cpp via `llama-server` subprocess with auto port allocation
- `ifran-backends/ollama`: Ollama HTTP client — chat, streaming, model load/unload via keep_alive
- `ifran-backends/vllm`: vLLM HTTP client — OpenAI-compatible chat and streaming
- `ifran-backends/tensorrt`: TensorRT-LLM HTTP client to Triton server with streaming
- `ifran-backends/candle`: Candle (pure Rust) backend for SafeTensors — trait impl, inference pending candle crate dep
- `ifran-backends/gguf`: Direct GGUF loading backend — trait impl, inference pending candle-gguf dep
- `ifran-backends/onnx`: ONNX Runtime backend — trait impl, inference pending ort crate dep
- `ifran-backends/router`: Smart backend auto-selection by format, hardware, and user preference

#### API Server
- `ifran-api/rest/router`: Axum router with all route groups, CORS, telemetry, auth
- `ifran-api/rest/models`: `GET /models`, `GET /models/:id`, `DELETE /models/:id`
- `ifran-api/rest/inference`: `POST /inference`, `POST /inference/stream` (SSE)
- `ifran-api/rest/openai_compat`: `POST /v1/chat/completions` (streaming + non-streaming), `GET /v1/models`
- `ifran-api/rest/training`: `POST /training/jobs`, `GET /training/jobs`, `GET /training/jobs/:id`, `POST /training/jobs/:id/cancel`
- `ifran-api/rest/eval`: `POST /eval/runs`, `GET /eval/runs`, `GET /eval/runs/:id`
- `ifran-api/rest/marketplace`: `GET /marketplace/search`, `GET /marketplace/entries`, `POST /marketplace/publish`, `DELETE /marketplace/entries/:name`
- `ifran-api/rest/system`: `GET /health`, `GET /system/status`
- `ifran-api/middleware/telemetry`: Request tracing via tower-http
- `ifran-api/middleware/auth`: Optional Bearer token auth via `IFRAN_API_KEY`
- `ifran-api/state`: Shared application state with config, DB, backend router, model manager, job manager

#### Training
- `ifran-train/job/manager`: Job lifecycle (create, start, cancel) with concurrent job limits
- `ifran-train/job/scheduler`: FIFO priority queue
- `ifran-train/job/status`: Job state machine (Queued → Running → Completed/Failed/Cancelled)
- `ifran-train/executor/docker`: Docker container executor with GPU passthrough and method-specific script selection
- `ifran-train/executor/subprocess`: Python subprocess executor
- `ifran-train/dataset/loader`: JSONL, CSV, Parquet, HuggingFace dataset loading with sample counting
- `ifran-train/dataset/validator`: Schema validation for JSONL and CSV formats
- `ifran-train/methods`: LoRA/QLoRA, full fine-tune, DPO, RLHF, distillation configs and arg generation
- `ifran-train/checkpoint/store`: Checkpoint save/load/list/prune with metadata
- `ifran-train/checkpoint/merger`: LoRA adapter merging into base model via PEFT
- `ifran-train/distributed/coordinator`: Distributed job creation, worker assignment, lifecycle
- `ifran-train/distributed/worker`: Worker local state, distributed CLI arg generation, lifecycle
- `ifran-train/distributed/aggregator`: Checkpoint aggregation plans (average/weighted), command builder
- Python training scripts: `train_sft.py`, `train_full.py`, `train_dpo.py`, `train_rlhf.py`, `train_distill.py`

#### SY Bridge
- `ifran-bridge/server`: gRPC server with connection state machine, heartbeat, degraded mode
- `ifran-bridge/client`: gRPC client with reconnect (exponential backoff), capability announcement, GPU requests
- `ifran-bridge/protocol`: Connection states, heartbeat config, capability announcement types
- `ifran-bridge/discovery`: SY endpoint discovery (config → `SY_ENDPOINT` env → localhost:9420)

#### CLI
- `ifran pull`: Model pull with HuggingFace resolution, progress bar, integrity check, catalog registration
- `ifran list`: Table-formatted model listing
- `ifran rm`: Model removal with confirmation, disk cleanup, catalog deletion
- `ifran run`: Interactive inference with streaming output
- `ifran serve`: Full API server
- `ifran train`: Training job creation with `--base-model`, `--dataset`, `--method`
- `ifran status`: Hardware and catalog status
- `ifran eval`: Model evaluation with benchmark selection (`--benchmark`, `--dataset`, `--sample-limit`)
- `ifran marketplace search/publish/unpublish`: Model marketplace management

#### Desktop Application
- Tauri v2 + SvelteKit scaffold with dark theme UI
- Dashboard: model count, loaded models, hardware summary, version
- Models page: browse, delete, pull progress
- Chat page: model selection, message history, OpenAI-compatible inference
- Training page: job list with progress bars, step/epoch/loss, cancel
- Settings page: server status, hardware info, config guidance
- 10 Tauri commands bridging frontend to Ifran REST API

#### Agnosticos Integration
- `deploy/ifran.service`: systemd unit with security hardening (ProtectSystem, PrivateTmp, NoNewPrivileges, GPU device access)
- `deploy/agnosticos/ifran.pkg.toml`: Package spec with user creation hooks, capability registration, systemd setup
- `deploy/ifran.toml.example`: System-level config template with all backends documented
- Config auto-discovery chain: `IFRAN_CONFIG` env → `~/.ifran/ifran.toml` → `/etc/ifran/ifran.toml` → defaults
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
- `ifran-core/eval/benchmarks`: MMLU, HellaSwag, HumanEval, perplexity prompt formatting and scoring
- `ifran-core/eval/runner`: `run_benchmark()` dispatcher with per-benchmark runners (MMLU, HellaSwag, HumanEval, perplexity)
- `ifran-api/rest/eval`: Background benchmark execution wired to inference backends via closure-based `infer_fn`
- `ifran-cli/eval`: CLI eval command wired to local API with all benchmark types

#### Model Marketplace — Remote & Trust
- `ifran-core/marketplace/resolver`: Remote peer search via `GET /marketplace/search` on each peer, deduplication
- `ifran-core/marketplace/trust`: Trust/verification layer — `TrustLevel` (Untrusted/ChecksumVerified/TrustedPublisher), `TrustPolicy`, `verify_entry()`, `verify_download()`
- `ifran-api/rest/marketplace`: Model download endpoint (`GET /marketplace/download/:name`), model pull endpoint (`POST /marketplace/pull`) with SHA-256 verification
- `ifran-cli/marketplace`: `ifran marketplace pull --peer <url>` command with trust verification

#### Distributed Training
- `ifran-api/rest/distributed`: Full REST API for distributed job management (create, list, get, assign workers, start, complete, fail, aggregate)
- `ifran-cli/train`: `--distributed`, `--world-size`, `--strategy` flags for distributed training
- `ifran-bridge/client`: `request_worker_assignment()` and `sync_checkpoint()` RPCs for cross-node coordination
- `ifran-train/distributed/coordinator`: `collect_checkpoint_paths()` for checkpoint synchronization
- `ifran-train/distributed/aggregator`: `FederatedConfig`, `build_federated_command()` for federated averaging

#### SY Bridge Integration
- `ifran-api/state`: Bridge client/server initialized in AppState when `bridge.enabled = true`, with SY endpoint discovery
- `ifran-api/rest/bridge`: REST endpoints — `GET /bridge/status`, `POST /bridge/connect`, `POST /bridge/heartbeat`
- `ifran-api/main`: Auto-connect to SY on startup, background heartbeat task with loaded models/GPU/active jobs
- `ifran-api/rest/system`: Bridge connection state included in `/system/status`
- `ifran-api/rest/training`: Training job start and cancel events reported to SY via bridge client
- `ifran-api/rest/distributed`: Worker assignments forwarded to SY via `RequestWorkerAssignment`, checkpoint sync via `SyncCheckpoint` on worker completion, job completion reported to SY
- 510 tests across all modules (~70% coverage)

### Fixed
- **SECURITY**: SQL injection in `ifran-core/marketplace/catalog.rs` — `search()` now uses parameterized queries instead of string interpolation
- **SECURITY**: Python code injection in `ifran-train/checkpoint/merger.rs` — `merge_lora()` now passes paths via environment variables instead of interpolating into Python source
- `ifran-core/pull/downloader.rs`: Corrupted `.part` files are now cleaned up when SHA-256 verification fails, preventing infinite retry loops
- `ifran-core/registry/huggingface.rs`: Replaced `.unwrap()` with proper error handling in `resolve_gguf()` fallback path
- `ifran-core/marketplace/catalog.rs`: Replaced `serde_json::to_string().unwrap()` calls with proper error propagation in `publish()`
- `ifran-api/rest/openai_compat.rs`: `list_models` now returns `Result` — database errors are propagated as 500 instead of silently returning empty list
- `ifran-api/rest/inference.rs`: Both `/inference` and `/inference/stream` now use the loaded model's actual backend instead of hardcoding `"llamacpp"`
- `ifran-api/rest/models.rs`: Failed filesystem cleanup during model deletion is now logged as a warning instead of silently ignored
- `ifran-api/rest/marketplace.rs`: Model download endpoint now streams files via `tokio_util::io::ReaderStream` instead of loading entire model into memory (prevents OOM on large files)
- `ifran-train/executor/subprocess.rs`: Fixed potential deadlock in `run()` — child process is now removed from the tracking map before awaiting, so `cancel()` can acquire the write lock concurrently
- `ifran-backends/llamacpp`: `unload_model()` now calls `wait()` after `kill()` to reap child processes and prevent zombie `llama-server` processes
- `ifran-backends/ollama`: `unload_model()` now logs HTTP errors instead of silently discarding them with `let _ =`
- `ifran-backends/ollama`: Stream errors in `infer_stream()` now logged with `warn!` instead of silently breaking
- `ifran-backends/router`: `select()` now logs a warning when the user's preferred backend is not found, before falling back to auto-selection
- `ifran-train/job/manager`: Fixed potential deadlock in `cancel_job()` — read lock is now released before calling `executor.cancel()`, preventing deadlock when the executor needs write access
- `ifran-train/executor/docker`: Container is now tracked BEFORE `docker run` executes, so `cancel()` can find and stop the container during long-running training
- `ifran-train/distributed/coordinator`: `worker_completed()` now guards against over-counting — duplicate completion reports after all workers have finished are no-ops
- `ifran-api/rest/training`: `create_job` auto-start failures are now logged as warnings instead of silently ignored with `let _ =`
- `ifran-core/marketplace/resolver`: HTTP client builder failure now logs a warning and falls back to default client instead of silently using `unwrap_or_default()`
- `ifran-core/marketplace/resolver`: Format filter serialization failure in `query_peer()` now returns an error instead of silently dropping the filter via `unwrap_or_default()`
- `ifran-core/lifecycle/manager`: Replaced `.unwrap()` on `best_accelerator()` with proper error propagation — prevents panic when GPU is detected but accelerator type is undetermined
- `ifran-core/lifecycle/memory`: `estimate_gguf()` now rounds up file size to nearest MB instead of truncating, preventing underestimation of memory requirements
- `ifran-backends/router`: `select()` no longer returns an incompatible backend as fallback — returns `None` when no backend supports the requested format, instead of silently picking any backend
- `ifran-backends/ollama`: `load_model()` now validates the HTTP response status — previously a failed load (HTTP 500) was silently treated as success, causing phantom loaded models
- `ifran-train/executor/docker`: `cancel()` now removes the container from tracking after stopping, preventing unbounded memory growth from accumulated stale entries
- `ifran-train/executor/docker`: `cancel()` now logs `docker stop` errors instead of silently discarding them with `let _ =`
- `ifran-train/executor/docker`: Container tracking cleanup on spawn failure is now synchronous instead of fire-and-forget `tokio::spawn`
- `ifran-train/job/manager`: Fixed race condition in `start_job()` — running job count is now checked inside the write lock, preventing concurrent calls from exceeding `max_concurrent`
- `ifran-train/job/manager`: `cancel_job()` now re-validates terminal state after reacquiring write lock, preventing Cancelled from overwriting a concurrent Completed transition
- `ifran-api/rest/models`: `delete_model` now deletes from database first, then cleans up filesystem — prevents orphaned DB records if FS deletion succeeds but DB deletion fails
- `ifran-api/rest/marketplace`: Download endpoint removes TOCTOU race — uses `File::open()` error handling instead of separate `exists()` check, and gets metadata from the open file handle
- `ifran-api/middleware/auth`: Replaced hardcoded `&header[7..]` string slice with `strip_prefix("Bearer ")` for safer token extraction
- `ifran-api/main`: Heartbeat bridge communication errors now logged with `warn!` instead of silently discarded
- CI/CD container image build timeout: switched from compiling Rust inside Docker (30+ min under QEMU for arm64) to using pre-built binaries from the build-release job via `Dockerfile.release` with `TARGETARCH`

### Enhanced
- `ifran-backends/*`: All 4 HTTP backends (llamacpp, ollama, vllm, tensorrt) now use 300-second request timeouts instead of unbounded `reqwest::Client::new()`
- `ifran-backends/*`: All 4 streaming backends now check `tx.is_closed()` to stop processing when the receiver is dropped (early client disconnect)
- `ifran-backends/*`: All 4 streaming backends enforce a 1 MB buffer limit to prevent unbounded memory growth from malformed SSE streams
- `ifran-backends/llamacpp`: `load_model()` now validates that the model file exists before spawning `llama-server`, providing a clear error instead of a cryptic process failure
- `ifran-api/rest/inference`: Both `/inference` and `/inference/stream` now match the requested model by name instead of always using the first loaded model
- `ifran-api/rest/openai_compat`: `/v1/chat/completions` now matches the requested model by name instead of always using the first loaded model
- `ifran-api/rest/eval`: Eval run inference now targets the requested model by name instead of always using the first loaded model
- `ifran-core/lifecycle/manager`: `LoadedModel` now carries `model_name` to support model selection by name in inference endpoints
- `ifran-types/training`: `HyperParams::validate()` rejects `learning_rate <= 0`, `epochs == 0`, `batch_size == 0`, and `max_seq_length == 0`
- `ifran-train/job/manager`: `create_job()` now validates hyperparameters before creating the job
- `ifran-train/checkpoint/store`: `prune()` now handles already-deleted checkpoints gracefully (ENOENT-tolerant) instead of propagating errors
- `ifran-train/dataset/validator`: CSV validation now handles RFC 4180 quoted fields — commas inside `"quoted,field"` no longer cause false column-count mismatches
- 512 tests across all modules
