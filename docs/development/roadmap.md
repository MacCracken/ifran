# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,406 tests** across 7 crates. CI threshold: 65%.

---

## Performance & Memory

- [ ] **Fleet list pagination** — `GET /fleet/nodes` returns all nodes unbounded; add `limit`/`offset` using the existing pagination module
- [ ] **Marketplace list pagination** — `GET /marketplace/entries` and `GET /marketplace/search` return unbounded results; add pagination
- [ ] **RLHF / RAG / Experiment list pagination** — `GET /rlhf/sessions`, `GET /rag/pipelines`, `GET /experiments` lack pagination; wire in `PaginatedResponse`
- [ ] **SQLite connection pooling** — each store holds a single `Connection` behind a Mutex; evaluate `r2d2-sqlite` for higher throughput under concurrent load
- [ ] **Rate limiter IP eviction** — per-IP `DashMap` grows indefinitely; add periodic sweep of stale entries (e.g. no requests in 10 minutes)

## API Quality

- [ ] **Consistent response envelope** — some endpoints return raw arrays, others use `{"data": [...]}` wrappers; standardize on a single envelope format
- [ ] **Endpoint filtering and sorting** — many list endpoints (models, training jobs, eval runs, fleet nodes) accept no filter or sort params; add `?status=`, `?sort_by=`, `?order=` where relevant

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

## User Experience

- [ ] **Error hints** — `ApiError.hint` field exists but is rarely populated; add actionable hints to common failure modes (no model loaded, backend unavailable, auth missing)
- [ ] **gRPC service implementation** — proto defines 7 RPCs but `service.rs` is a stub; implement at least `PullModel`, `ListModels`, `Infer`, and `GetStatus` for bridge parity
- [ ] **RAG real embeddings** — `stub_embed()` uses a deterministic hash (not ML-based); integrate a real embedding backend or delegate to an inference backend

## Post-v1 Considerations

- Prompt management and versioning
- Circuit breaker / retry middleware for backend HTTP calls
- Experiment DSL documentation and guide
- Advanced features guide (auto-labeling, data augmentation, A/B testing, drift detection)
- Marketplace and RAG user guides
