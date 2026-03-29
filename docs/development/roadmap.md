# Ifran Development Roadmap

> Versioning: Semver `MAJOR.MINOR.PATCH`

### Test Coverage

Current: **1,785 tests**, 72.2% coverage. CI threshold: **70%**.

---

## 1.0.0 — All Complete

### Security Hardening
- [x] Prompt injection detection (32 patterns, 5 categories)
- [x] Output filtering / redaction (API keys, PII, system prompt leakage)
- [x] Input sanitization (50K cap, boundary markers, combined length)
- [x] Audit trail (HMAC-SHA256 tamper-evident chain)
- [x] Circuit breaker for backends (Closed→Open→HalfOpen FSM)

### Operational Resilience
- [x] Retry with exponential backoff
- [x] Inference output validation (Text/Json/JsonSchema)
- [x] Graceful degradation (health ring buffer per backend)

### Observability
- [x] Request / correlation ID (`X-Request-ID` middleware)
- [x] Prometheus metrics (9 metrics, `GET /metrics`)
- [x] Rate limiter IP eviction (majra stale-key sweep)

### Testing
- [x] Auth / permission integration tests
- [x] Concurrent operation tests (7 tests, 10-20 parallel tasks)
- [x] Fuzzing targets (5 targets)
- [x] Shared test utilities
- [x] Coverage raised to 72.2% absolute / 75.9% coverable

### Performance
- [x] 26 Criterion benchmarks across 2 bench files

---

## 1.0.1 — Patch (Blocking SY Integration)

- [ ] **Feature-gate rusqlite behind `sqlite` feature** — `rusqlite 0.39` requires `libsqlite3-sys 0.37` which conflicts with consumers using sqlx (`libsqlite3-sys 0.30`). Extract store trait interfaces (`StoreTrait` with `SqliteStore` impl) so modules compile without sqlite. Gate all 13 store modules + r2d2/r2d2_sqlite behind `sqlite = ["dep:rusqlite", "dep:r2d2", "dep:r2d2_sqlite"]`. Add `sqlite` to default features. Library consumers use `ifran = { default-features = false }` to avoid the conflict.

---

## 1.1.0 — P0 Items

### Storage Abstraction (P0)

- [ ] **Database-agnostic store traits** — extract `trait JobStore`, `trait EvalStore`, `trait TenantStore`, etc. from the 13 concrete SQLite implementations. Each trait defines CRUD operations; `SqliteStore` is the default impl. This enables PostgreSQL backends (see below), testing with in-memory stores, and library consumption without sqlite.
- [ ] **PostgreSQL backend** — implement store traits against PostgreSQL via sqlx. Required for production multi-instance deployments (SQLite doesn't support concurrent writes from multiple processes). See ADR-010.
- [ ] **Redis backend for fleet coordination** — replace in-memory fleet registry with Redis pub/sub for multi-instance node discovery and heartbeat propagation. See ADR-010.

### Testing & Coverage

- [ ] **Extract CLI execute() logic into testable functions** — ~200 lines recoverable
- [ ] **Mock-based backend inference tests** — `MockBackend` implementing `InferenceBackend`. ~80 lines
- [ ] **Experiment runner test with mock executor** — ~100 lines
- [ ] **OpenAI-compat streaming test** — `/v1/chat/completions` with `stream: true`. ~30 lines
- [ ] **Docker executor integration test** — mock Docker binary. ~50 lines
- [ ] **Bridge protocol tests** — loopback connect/heartbeat/reconnect. ~80 lines
- [ ] **Property-based tests for prompt guard** — proptest adversarial strings
- [ ] **Raise CI threshold to 78%**

### Operational

- [ ] **Human approval gates** — for high-risk training jobs (RLHF data, production model deployment)
- [ ] **Per-tenant token budget enforcement** — currently global only
- [ ] **Tanur frontend integration** — API contract validation against tanur's expected endpoints

---

## Post-1.1 Considerations

- Prompt management and versioning
- Experiment DSL documentation and guide
- Advanced features guide (auto-labeling, data augmentation, A/B testing, drift detection)
- Marketplace and RAG user guides
