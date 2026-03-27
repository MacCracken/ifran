# Ifran Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,430 tests** across 7 crates. CI threshold: 65%.

---

## Ecosystem Migration

### Hoosh — inference gateway (replaces `ifran-backends` crate)
- [ ] **Replace `BackendRouter` with hoosh** — delegate all inference routing (13+ backends) to `hoosh` as a library dependency; remove `ifran-backends` crate
- [ ] **Drop `ai-hwaccel` direct dep** — hardware detection comes through hoosh; remove ifran's 800-line fallback detection code
- [ ] **Simplify OpenAI-compat API** — `/v1/chat/completions` and `/v1/models` proxy to hoosh instead of reimplementing
- [ ] **Drop `axum` if possible** — evaluate embedding hoosh's server builder (pending hoosh `HooshServer::builder()` API) to serve both inference and management endpoints from one process

### Majra — concurrency primitives (replaces governor, dashmap, manual broadcast)
- [x] **Replace `governor` rate limiting with `majra::ratelimit`** — per-IP token bucket with automatic stale-key eviction
- [x] **Replace fleet heartbeat with `majra::heartbeat`** — Online→Suspect→Offline FSM with configurable timeouts
- [x] **Unify event buses under `majra::pubsub`** — training, GPU, and download progress events on TypedPubSub with MQTT-style topic routing
- [x] **Replace FIFO job scheduler with `majra::queue`** — 5-tier priority queue with soft-delete cancellation
- [x] **Use `majra::barrier` for distributed training** — AsyncBarrierSet replaces manual worker completion counter; deadlock recovery via force-release
- [x] **Use `majra::namespace` for multi-tenant isolation** — namespaced rate limiter keys, pub/sub topics, and fleet node IDs per tenant
- [x] **Wire `majra::metrics::PrometheusMetrics`** — `GET /metrics` endpoint with Prometheus registry; ready for PrometheusMetrics wiring
- [x] **Use `majra::ws` for real-time dashboards** — WsBridge fans out training/GPU/progress events to WebSocket clients via shared PubSub hub
- [x] **Drop `dashmap` dep** — replaced with `RwLock<HashMap>` in BackendRouter (write-once pattern)
- [ ] **Use `majra::fleet` for multi-node GPU scheduling** — `FleetQueue` with work-stealing replaces manual node selection; `ResourcePool`/`ResourceReq` for GPU-aware routing
- [ ] **Use `majra::dag` for training pipelines** — model lifecycle (download → convert → quantise → index) is a DAG; `WorkflowEngine` provides tier-based parallel execution, retry, error policies
- [ ] **Evaluate `majra::redis_backend` for multi-instance** — `RedisRateLimiter` for distributed rate limiting, `RedisHeartbeatTracker` for cross-instance fleet health
- [ ] **Evaluate `majra::postgres` for durable job scheduling** — `PostgresQueueBackend` for `ManagedQueue` persistence (currently SQLite); `PostgresWorkflowStorage` for DAG workflow runs

### Dependency cleanup post-migration
- [x] Remove: `dashmap` (dropped from ifran-backends and workspace)
- [ ] Remove: `ai-hwaccel`, `async-trait` (if no longer needed after hoosh migration)
- [ ] Evaluate removing: `axum` (if hoosh embeds the server)

## Performance & Memory

- [x] **SQL-level pagination on all list endpoints** — marketplace, RLHF, RAG, experiment, tenant, lineage, version, model list endpoints now push `LIMIT`/`OFFSET` + `COUNT(*)` to SQL; handlers use `PaginatedResponse::pre_sliced()`
- [x] **Fleet list pagination** — `GET /fleet/nodes` uses in-memory pagination via `from_slice()` (appropriate for in-memory FleetManager)
- [x] **Hot-path optimizations** — `#[inline]` on cache/cosine/memory/scoring functions; zero-alloc `score_contains_match` (11x speedup); `deserialize_quoted` helper eliminates `format!` in DB row parsing; `Vec::with_capacity` in RAG chunker
- [x] **SQLite connection pooling** — all stores use `r2d2_sqlite` connection pools (max 4 conns); `tokio::Mutex` removed from `AppState` — concurrent requests use separate pool connections
- [x] **Swappable DB backend** — store traits defined in `ifran_core::storage::traits`; SQLite implementations can be swapped for Postgres or other backends
- [ ] **Rate limiter IP eviction** — solved by majra migration (stale-key eviction built in)

## Observability

- [ ] **Request / correlation ID** — inject a unique ID per request (from header or generated), propagate through tracing spans, return in response headers for distributed tracing
- [ ] **Prometheus metrics endpoint** — add `GET /metrics` exposing request latency histograms, job queue depth gauges, model load/unload counters, rate limiter rejection counts
- [x] **Benchmark suite** — Criterion benchmarks in `ifran-core` (cosine similarity, memory estimation, cache ops, eval scoring); `scripts/bench-history.sh` tracks CSV history

## Testing

- [ ] **Auth / permission integration tests** — no tests cover 401/403 paths, multi-tenant isolation, or admin-key enforcement
- [ ] **Concurrent operation tests** — add tests for race conditions in job scheduling, model loading, and fleet registration under parallel requests
- [ ] **Raise CI coverage threshold** — increase from 65% to 75%; consider per-crate minimums for critical crates (ifran-core, ifran-train)
- [ ] **Fuzzing targets** — add `cargo-fuzz` targets for config parsing, gRPC message handling, and REST JSON input deserialization
- [ ] **Shared test utilities crate** — deduplicate `test_config()`, `test_app()`, mock builders, and fixture data into a workspace-internal `ifran-testutil` crate

## Post-v1 Considerations

- Prompt management and versioning
- Circuit breaker / retry middleware for backend HTTP calls
- Experiment DSL documentation and guide
- Advanced features guide (auto-labeling, data augmentation, A/B testing, drift detection)
- Marketplace and RAG user guides
