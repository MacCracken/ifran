# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,406 tests** across 7 crates. CI threshold: 65%.

---

## Reliability & Hardening

Robustness improvements to reach production confidence.

- [ ] **Graceful shutdown orchestration** — background tasks (eviction loop, fleet heartbeat, bridge heartbeat, telemetry loop) are spawned without tracking; add `tokio::JoinSet` or cancellation tokens so the process drains cleanly on SIGTERM
- [ ] **Remove `unwrap()` in handler code** — `bridge.rs` (lines 199, 216, 231), `rlhf.rs` (lines 86, 119), `experiment.rs` (line 98) use `unwrap()` on `serde_json` serialization in production paths; replace with `.map_err()`
- [ ] **Readiness probe** — `GET /health` is liveness only; add `GET /ready` that checks database connectivity, at least one backend registered, and store initialization
- [ ] **EvalRunner result buffer limit** — `EvalRunState.results: Vec<EvalResult>` grows unbounded during large benchmark runs; add periodic flush to database or a size-capped buffer
- [ ] **Fleet node eviction** — offline nodes remain in `FleetManager.nodes` HashMap forever; add TTL-based removal for nodes past `offline_timeout_secs`
- [ ] **Streaming keep-alive consistency** — training job stream has keep-alive (15s), but inference stream and training events SSE do not; add keep-alive to all SSE endpoints

## Performance & Memory

- [ ] **Fleet list pagination** — `GET /fleet/nodes` returns all nodes unbounded; add `limit`/`offset` using the existing pagination module
- [ ] **Marketplace list pagination** — `GET /marketplace/entries` and `GET /marketplace/search` return unbounded results; add pagination
- [ ] **RLHF / RAG / Experiment list pagination** — `GET /rlhf/sessions`, `GET /rag/pipelines`, `GET /experiments` lack pagination; wire in `PaginatedResponse`
- [ ] **SQLite connection pooling** — each store holds a single `Connection` behind a Mutex; evaluate `r2d2-sqlite` for higher throughput under concurrent load
- [ ] **Rate limiter IP eviction** — per-IP `DashMap` grows indefinitely; add periodic sweep of stale entries (e.g. no requests in 10 minutes)

## API Quality

- [ ] **Consistent response envelope** — some endpoints return raw arrays, others use `{"data": [...]}` wrappers; standardize on a single envelope format
- [ ] **Structured error codes** — `ApiError` (in `error.rs`) defines `code`, `message`, `hint` but most handlers return bare `(StatusCode, String)` tuples; adopt `ApiError` across all endpoints
- [ ] **Debug formatting cleanup** — several handlers use `format!("{:?}", enum).to_lowercase()` instead of serde `Serialize`; replace with proper serialization
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

- [ ] **Hardware acceleration guide** — `docs/hardware-acceleration.md` covering detection, backend configuration, and `ai-hwaccel` feature flag (8 new backends undocumented)
- [ ] **Fleet management guide** — `docs/fleet-management.md` covering node registration, health states, statistics, and `[fleet]` config
- [ ] **Multi-tenancy guide** — `docs/multi-tenancy.md` covering tenant creation, API key lifecycle, resource isolation, and budget enforcement (`[budget]` config)
- [ ] **Evaluation guide** — `docs/evaluation-guide.md` covering supported benchmarks (MMLU, HellaSwag, HumanEval, perplexity, custom), dataset format, and result interpretation
- [ ] **CLI reference** — `docs/cli-reference.md` with all commands and subcommands (`marketplace`, `experiment`, `eval`, `status`)
- [ ] **Update `deploy/synapse.toml.example`** — add missing config sections: `[fleet]`, `[budget]`, `hardware.telemetry_interval_secs`, `security.require_encrypted_storage`, per-backend sections
- [ ] **Update README** — feature list is outdated; add evaluation, marketplace, fleet, multi-tenancy, RAG, lineage, experiments, hardware backends
- [ ] **Update `SECURITY.md`** — document per-IP rate limiting (changed from global), tenant in-flight cancellation, and lineage depth limit

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
