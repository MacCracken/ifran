# Ifran Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,406 tests** across 7 crates. CI threshold: 65%.

---

## Ecosystem Migration

### Hoosh — inference gateway (replaces `ifran-backends` crate)
- [ ] **Replace `BackendRouter` with hoosh** — delegate all inference routing (13+ backends) to `hoosh` as a library dependency; remove `ifran-backends` crate
- [ ] **Drop `ai-hwaccel` direct dep** — hardware detection comes through hoosh; remove ifran's 800-line fallback detection code
- [ ] **Simplify OpenAI-compat API** — `/v1/chat/completions` and `/v1/models` proxy to hoosh instead of reimplementing
- [ ] **Drop `axum` if possible** — evaluate embedding hoosh's server builder (pending hoosh `HooshServer::builder()` API) to serve both inference and management endpoints from one process

### Majra — concurrency primitives (replaces governor, dashmap, manual broadcast)
- [ ] **Replace `governor` rate limiting with `majra::ratelimit`** — includes automatic stale-key eviction (pending majra roadmap item)
- [ ] **Replace fleet heartbeat with `majra::heartbeat`** — drop custom `Arc<RwLock<HashMap>>` fleet manager and telemetry loop
- [ ] **Unify event buses under `majra::pubsub`** — replace 3 separate `broadcast::channel` instances (training, GPU, progress) with topic-based pub/sub
- [ ] **Replace FIFO job scheduler with `majra::queue`** — get priority queues + DAG scheduling + GPU-aware dequeue (pending majra roadmap items)
- [ ] **Use `majra::barrier` for distributed training** — replace manual worker coordination with N-way barrier sync
- [ ] **Drop `dashmap` dep** — all concurrent map uses migrate to majra primitives or standard `Arc<RwLock<HashMap>>` where appropriate

### Dependency cleanup post-migration
- [ ] Remove: `governor`, `dashmap`, `ai-hwaccel`, `async-trait` (if no longer needed)
- [ ] Evaluate removing: `axum` (if hoosh embeds the server)
- [ ] Add: `hoosh`, `majra`

## Performance & Memory

- [ ] **Fleet list pagination** — `GET /fleet/nodes` returns all nodes unbounded; add `limit`/`offset` using the existing pagination module
- [ ] **Marketplace list pagination** — `GET /marketplace/entries` and `GET /marketplace/search` return unbounded results; add pagination
- [ ] **RLHF / RAG / Experiment list pagination** — `GET /rlhf/sessions`, `GET /rag/pipelines`, `GET /experiments` lack pagination; wire in `PaginatedResponse`
- [ ] **SQLite connection pooling** — each store holds a single `Connection` behind a Mutex; evaluate `r2d2-sqlite` for higher throughput under concurrent load
- [ ] **Rate limiter IP eviction** — solved by majra migration (stale-key eviction built in)

## Observability

- [ ] **Request / correlation ID** — inject a unique ID per request (from header or generated), propagate through tracing spans, return in response headers for distributed tracing
- [ ] **Prometheus metrics endpoint** — add `GET /metrics` exposing request latency histograms, job queue depth gauges, model load/unload counters, rate limiter rejection counts
- [ ] **Benchmark suite** — CI job exists but no Criterion benchmarks are defined; add benchmarks for inference latency, model pull throughput, and training scheduling

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
