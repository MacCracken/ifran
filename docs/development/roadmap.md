# Ifran Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

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

## 1.1.0 — Coverage & Hardening

### Sprint 1 — Coverage to 75% absolute

- [ ] **Extract CLI execute() logic into testable functions** — `pull::execute()`, `list::execute()`, `serve::execute()` each contain 30-50 lines of pure logic (path construction, config resolution, output formatting) that can be extracted and tested. ~200 lines recoverable.
- [ ] **Mock-based backend inference tests** — create a `MockBackend` implementing `InferenceBackend` that returns canned responses. Test the full inference handler path (cache → budget → backend → filter → cache store) without network. ~80 lines in `inference.rs`.
- [ ] **Experiment runner test with mock executor** — test `ExperimentRunner::run()` with a mock executor that completes trials instantly. Covers the trial loop, grid/random search selection, and result recording. ~100 lines in `runner.rs`.
- [ ] **OpenAI-compat streaming test** — test `/v1/chat/completions` with `stream: true` using a mock backend that yields chunks. ~30 lines in `openai_compat.rs`.
- [ ] **Raise CI threshold to 75%**

### Sprint 2 — Coverage to 78% + hardening

- [ ] **Docker executor integration test** — test `DockerExecutor::run()` with a mock Docker binary (shell script that echoes success). Covers the container lifecycle, timeout, and cancellation paths. ~50 lines.
- [ ] **Bridge protocol tests** — test `BridgeClient::connect()` and `BridgeServer::start()` with loopback. Covers connection state machine, heartbeat, and reconnect. ~80 lines.
- [ ] **Telemetry integration test** — test OTLP init path with a no-op exporter. ~20 lines.
- [ ] **CLI main dispatch test** — test `Cli::parse` → `Commands` dispatch with dry-run mode (parse + validate, don't execute). ~30 lines.
- [ ] **Property-based tests for prompt guard** — proptest strategies generating adversarial strings; verify scanner never panics or returns risk_score outside [0, 1].
- [ ] **Raise CI threshold to 78%**

---

## Post-1.1 Considerations

- Prompt management and versioning
- Experiment DSL documentation and guide
- Advanced features guide (auto-labeling, data augmentation, A/B testing, drift detection)
- Marketplace and RAG user guides
- Redis backend for multi-instance fleet coordination (see ADR-010)
- PostgreSQL backend for durable workflow persistence (see ADR-010)
- Human approval gates for high-risk training jobs (RLHF data, production model deployment)
- Token budget enforcement per tenant (currently global only)
