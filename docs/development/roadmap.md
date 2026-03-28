# Ifran Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,567 tests**, single flat crate. CI threshold: 65%.

---

## Pre-1.0 — Required Before Release

*Items informed by SY/AgnosAI security and operational parity. Ifran ships before SY — it must be production-hardened independently.*

### Security Hardening

- [x] **Prompt injection detection** — 32-pattern scanner across 5 attack categories (instruction override, role hijack, delimiter injection, data exfiltration, jailbreak). Risk scoring 0.0-1.0; blocks at ≥0.8. No regex dependency.
- [x] **Output filtering / redaction** — scans inference responses for AWS keys, GitHub tokens, Bearer tokens, generic API keys, emails, US phones, SSNs, credit cards, and system prompt leakage. Replaces with `[REDACTED_<CATEGORY>]`.
- [x] **Input sanitization** — 50K char hard cap on all inference endpoints. User content wrapped in `<|user_input_start|>`/`<|user_input_end|>` boundary markers. Combined message length validated on `/v1/chat/completions`.
- [x] **Audit trail** — HMAC-SHA256 linked tamper-evident chain for 9 action types (training jobs, model lifecycle, tenant management, config changes, admin actions). Verification detects any modification. Configurable max entries with eviction.
- [x] **Circuit breaker for backend HTTP calls** — Closed→Open→HalfOpen→Closed FSM. Configurable failure threshold and recovery timeout. Blocks requests when open, allows single probe in half-open state.

### Operational Resilience

- [x] **Retry with exponential backoff** — `RetryConfig` with max retries, base/max delay, deterministic jitter. `is_retryable()` classifies transient errors (connection refused, timeout, 502/503/429).
- [x] **Inference output validation** — `OutputFormat::Text`/`Json`/`JsonSchema{required_keys}`. Validates LLM output before returning. JSON mode rejects invalid JSON; schema mode checks required keys.
- [x] **Graceful degradation** — `BackendHealthTracker` with configurable ring buffer per backend. Tracks last N outcomes; failure rate determines Healthy/Degraded/Unhealthy status. `is_available()` for routing decisions.

### Observability

- [x] **Request / correlation ID** — `X-Request-ID` middleware with character validation (alphanumeric + `-_.`). Reads from client or generates UUID v4.
- [x] **Prometheus metrics wiring** — 9 metrics via `GET /metrics`. Gauges refreshed on scrape from live state.

### Testing

- [ ] **Auth / permission integration tests** — cover 401/403 paths, multi-tenant isolation, admin-key enforcement
- [ ] **Concurrent operation tests** — race conditions in job scheduling, model loading, fleet registration under parallel requests
- [ ] **Raise CI coverage threshold** — increase from 65% to 75%
- [ ] **Fuzzing targets** — `cargo-fuzz` targets for config parsing, gRPC message handling, REST JSON input deserialization
- [x] **Shared test utilities** — `server::test_helpers` module with `test_config()` and `test_state()`, used across 8 handler test modules

### Performance

- [x] **Rate limiter IP eviction** — `start_eviction_loop(5min idle, 60s sweep)` with `AtomicBool` spawn guard
- [ ] **Benchmarks** — Criterion.rs benchmarks for inference routing, model loading, training job scheduling

---

## Post-v1 Considerations

- Prompt management and versioning
- Experiment DSL documentation and guide
- Advanced features guide (auto-labeling, data augmentation, A/B testing, drift detection)
- Marketplace and RAG user guides
- Redis backend for multi-instance fleet coordination (see ADR-010)
- PostgreSQL backend for durable workflow persistence (see ADR-010)
- Human approval gates for high-risk training jobs (RLHF data, production model deployment)
- Token budget enforcement per tenant (currently global only)
