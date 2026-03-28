# Ifran Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,445 tests**, single flat crate. CI threshold: 65%.

---

## Performance & Memory

- [ ] **Rate limiter IP eviction** — wire majra's stale-key eviction through the rate limiter

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
- Redis backend for multi-instance fleet coordination (see ADR-010)
- PostgreSQL backend for durable workflow persistence (see ADR-010)
