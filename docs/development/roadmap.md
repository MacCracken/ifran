# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,406 tests** across 7 crates. CI threshold: 65%.

---

## Reliability & Hardening

Robustness improvements to reach production confidence.

- [x] **Graceful shutdown orchestration** — `axum::serve` now uses `with_graceful_shutdown`; telemetry and fleet manager stopped on exit
- [x] **Remove `unwrap()` in handler code** — replaced with `.unwrap_or()` in rlhf.rs, experiment.rs; fragile `.keys().last().unwrap()` removed
- [x] **Readiness probe** — `GET /ready` checks database lock and backend registration; returns 503 with reason on failure
- [x] **EvalRunner result buffer limit** — capped at 10,000 results; oldest half drained when full
- [x] **Fleet node eviction** — nodes offline for 2x `offline_timeout` are auto-evicted during health checks
- [x] **Streaming keep-alive consistency** — added `KeepAlive::default()` to inference stream and training events SSE

## Performance & Memory

- [ ] **Fleet list pagination** — `GET /fleet/nodes` returns all nodes unbounded; add `limit`/`offset` using the existing pagination module
- [ ] **Marketplace list pagination** — `GET /marketplace/entries` and `GET /marketplace/search` return unbounded results; add pagination
- [ ] **RLHF / RAG / Experiment list pagination** — `GET /rlhf/sessions`, `GET /rag/pipelines`, `GET /experiments` lack pagination; wire in `PaginatedResponse`
- [ ] **SQLite connection pooling** — each store holds a single `Connection` behind a Mutex; evaluate `r2d2-sqlite` for higher throughput under concurrent load
- [ ] **Rate limiter IP eviction** — per-IP `DashMap` grows indefinitely; add periodic sweep of stale entries (e.g. no requests in 10 minutes)

## API Quality

- [ ] **Consistent response envelope** — some endpoints return raw arrays, others use `{"data": [...]}` wrappers; standardize on a single envelope format
- [ ] **Structured error codes** — `ApiError` (in `error.rs`) defines `code`, `message`, `hint` but most handlers return bare `(StatusCode, String)` tuples; adopt `ApiError` across all endpoints
- [x] **Debug formatting cleanup** — replaced `format!("{:?}", enum).to_lowercase()` with proper `serde_json::to_value()` in models.rs, inference.rs, experiment.rs
- [ ] **Endpoint filtering and sorting** — many list endpoints (models, training jobs, eval runs, fleet nodes) accept no filter or sort params; add `?status=`, `?sort_by=`, `?order=` where relevant
- [ ] **Input validation gaps** — RLHF annotation, dataset augmentation, marketplace pull, and bridge connect endpoints accept unvalidated inputs; extend the existing validation middleware

## Observability

- [ ] **Request / correlation ID** — inject a unique ID per request (from header or generated), propagate through tracing spans, return in response headers for distributed tracing
- [ ] **Prometheus metrics endpoint** — add `GET /metrics` exposing request latency histograms, job queue depth gauges, model load/unload counters, rate limiter rejection counts
- [ ] **Benchmark suite** — CI job exists but no Criterion benchmarks are defined; add benchmarks for inference latency, model pull throughput, and training scheduling

## Testing

- [ ] **Auth / permission integration tests** — no tests cover 401/403 paths, multi-tenant isolation, or admin-key enforcement
- [ ] **Concurrent operation tests** — add tests for race conditions in job scheduling, model loading, and fleet registration under parallel requests
- [ ] **Raise CI coverage threshold** — increase from 65% to 75%; consider per-crate minimums for critical crates (synapse-core, synapse-train)
- [ ] **Fuzzing targets** — add `cargo-fuzz` targets for config parsing, gRPC message handling, and REST JSON input deserialization
- [ ] **Shared test utilities crate** — deduplicate `test_config()`, `test_app()`, mock builders, and fixture data into a workspace-internal `synapse-testutil` crate

## Documentation

- [x] **Hardware acceleration guide** — `docs/hardware-acceleration.md` covering detection, backend configuration, and `ai-hwaccel` feature flag
- [x] **Fleet management guide** — `docs/fleet-management.md` covering node registration, health states, statistics, and `[fleet]` config
- [x] **Multi-tenancy guide** — `docs/multi-tenancy.md` covering tenant creation, API key lifecycle, resource isolation, and budget enforcement
- [x] **Evaluation guide** — `docs/evaluation-guide.md` covering supported benchmarks, dataset format, and result interpretation
- [x] **CLI reference** — `docs/cli-reference.md` with all commands and subcommands
- [x] **Update `deploy/synapse.toml.example`** — added `[fleet]`, `[budget]`, `hardware.telemetry_interval_secs`, `security.require_encrypted_storage`, per-backend sections
- [x] **Update README** — added evaluation, marketplace, fleet, multi-tenancy, RAG, lineage, experiments, hardware backends
- [x] **Update `SECURITY.md`** — documented per-IP rate limiting, tenant in-flight cancellation, and lineage depth limit

## User Experience

- [ ] **CLI output formatting** — `output.rs` is a TODO stub; add table formatting, colored output, and progress helpers for `list`, `status`, `eval` commands
- [ ] **Error hints** — `ApiError.hint` field exists but is rarely populated; add actionable hints to common failure modes (no model loaded, backend unavailable, auth missing)
- [ ] **gRPC service implementation** — proto defines 7 RPCs but `service.rs` is a stub; implement at least `PullModel`, `ListModels`, `Infer`, and `GetStatus` for bridge parity
- [ ] **RAG real embeddings** — `stub_embed()` uses a deterministic hash (not ML-based); integrate a real embedding backend or delegate to an inference backend

## Post-v1 Considerations

- Prompt management and versioning
- Circuit breaker / retry middleware for backend HTTP calls
- Experiment DSL documentation and guide
- Advanced features guide (auto-labeling, data augmentation, A/B testing, drift detection)
- Marketplace and RAG user guides
