# Ifran Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,444 tests**, single flat crate. CI threshold: 65%.

---

## Ecosystem Integration

### Hoosh — inference gateway
- [ ] **Route server inference through hoosh** — use `hoosh::Router` + `ProviderRegistry` as the primary inference path, replacing direct backend HTTP calls for remote providers
- [ ] **Simplify OpenAI-compat API** — `/v1/chat/completions` and `/v1/models` proxy to hoosh instead of reimplementing
- [ ] **Drop `ai-hwaccel` direct dep** — hardware detection comes through hoosh; remove ifran's fallback detection code
- [ ] **Evaluate embedding hoosh server** — share one process for inference + management endpoints (pending hoosh `HooshServer::builder()` API)

### Majra — remaining primitives
- [ ] **Use `majra::fleet` for multi-node GPU scheduling** — `FleetQueue` with work-stealing replaces manual node selection; `ResourcePool`/`ResourceReq` for GPU-aware routing
- [ ] **Use `majra::dag` for training pipelines** — model lifecycle (download → convert → quantise → index) is a DAG; `WorkflowEngine` provides tier-based parallel execution, retry, error policies
- [ ] **Evaluate `majra::redis_backend` for multi-instance** — `RedisRateLimiter` for distributed rate limiting, `RedisHeartbeatTracker` for cross-instance fleet health
- [ ] **Evaluate `majra::postgres` for durable job scheduling** — `PostgresQueueBackend` for `ManagedQueue` persistence (currently SQLite); `PostgresWorkflowStorage` for DAG workflow runs

### Dependency cleanup
- [ ] Remove: `ai-hwaccel`, `async-trait` (if no longer needed after hoosh migration)
- [ ] Evaluate removing: `axum` (if hoosh embeds the server)

## Performance & Memory

- [ ] **Rate limiter IP eviction** — solved by majra migration (stale-key eviction built in), wire it through

## Observability

- [ ] **Request / correlation ID** — inject a unique ID per request (from header or generated), propagate through tracing spans, return in response headers
- [ ] **Prometheus metrics wiring** — expose request latency histograms, job queue depth gauges, model load/unload counters, rate limiter rejection counts via existing `GET /metrics` endpoint

## Testing

- [ ] **Auth / permission integration tests** — cover 401/403 paths, multi-tenant isolation, admin-key enforcement
- [ ] **Concurrent operation tests** — race conditions in job scheduling, model loading, fleet registration under parallel requests
- [ ] **Raise CI coverage threshold** — increase from 65% to 75%
- [ ] **Fuzzing targets** — `cargo-fuzz` targets for config parsing, gRPC message handling, REST JSON input deserialization
- [ ] **Shared test utilities** — deduplicate `test_config()`, `test_app()`, mock builders, and fixture data into a `#[cfg(test)]` module

## Post-v1 Considerations

- Prompt management and versioning
- Circuit breaker / retry middleware for backend HTTP calls
- Experiment DSL documentation and guide
- Advanced features guide (auto-labeling, data augmentation, A/B testing, drift detection)
- Marketplace and RAG user guides
