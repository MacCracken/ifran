# Ifran Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,445 tests**, single flat crate. CI threshold: 65%.

---

## Pre-1.0 — Required Before Release

*Items informed by SY/AgnosAI security and operational parity. Ifran ships before SY — it must be production-hardened independently.*

### Security Hardening

- [ ] **Prompt injection detection** — scan inference inputs for injection patterns (instruction override, role hijack, delimiter injection). SY/AgnosAI has 30+ pattern scanner; Ifran has none. Critical for any system accepting user prompts and forwarding to LLMs
- [ ] **Output filtering / redaction** — scan inference responses for leaked system prompts, API keys (AWS/GitHub/Bearer patterns), and PII (email, phone, SSN). AgnosAI's `OutputFilter` is the reference. Without this, a prompt injection can exfiltrate secrets via model output
- [ ] **Input sanitization** — enforce max input length (50K chars), wrap user content in boundary markers to prevent confusion with system instructions. Currently no length limits on `/inference` or `/v1/chat/completions`
- [ ] **Audit trail** — HMAC-SHA256 linked tamper-evident audit chain for training jobs, model deployments, and admin actions. SY's `sy-audit` crate is the reference. Lineage module tracks provenance but doesn't provide tamper detection
- [ ] **Circuit breaker for backend HTTP calls** — inference backends (llama.cpp, Ollama, vLLM, etc.) can hang or crash. Add circuit breaker with failure threshold, recovery timeout, and half-open probing. Currently listed as post-v1 but should be pre-v1 — a hung backend blocks the entire request pipeline

### Operational Resilience

- [ ] **Retry with exponential backoff** — configurable retry for transient backend failures (connection refused, timeout, 503). AgnosAI has `RetryConfig` with jitter, max retries, retryable heuristic. Currently Ifran has no retry on backend calls
- [ ] **Inference output validation** — validate LLM output against expected schema (JSON mode), auto-retry on parse failure. AgnosAI retries up to 2x on parse failure. Critical for structured output workflows (training data generation, eval scoring)
- [ ] **Graceful degradation** — when a backend goes unhealthy, route to next available backend instead of returning 500. Health ring buffer (N-point, M failures → unhealthy) per backend, similar to AgnosAI/hoosh provider health tracking

### Observability (Promote from existing)

- [ ] **Request / correlation ID** — inject a unique ID per request (from header or generated), propagate through tracing spans, return in response headers. SY propagates correlation IDs end-to-end including through the gRPC bridge — Ifran should match
- [ ] **Prometheus metrics wiring** — expose request latency histograms, job queue depth gauges, model load/unload counters, rate limiter rejection counts via existing `GET /metrics` endpoint

### Testing (Promote from existing)

- [ ] **Auth / permission integration tests** — cover 401/403 paths, multi-tenant isolation, admin-key enforcement
- [ ] **Concurrent operation tests** — race conditions in job scheduling, model loading, fleet registration under parallel requests
- [ ] **Raise CI coverage threshold** — increase from 65% to 75%
- [ ] **Fuzzing targets** — `cargo-fuzz` targets for config parsing, gRPC message handling, REST JSON input deserialization
- [ ] **Shared test utilities** — deduplicate `test_config()`, `test_app()`, mock builders, and fixture data into a `#[cfg(test)]` module

### Performance

- [ ] **Rate limiter IP eviction** — wire majra's stale-key eviction through the rate limiter
- [ ] **Benchmarks** — Criterion.rs benchmarks for inference routing, model loading, training job scheduling. AgnosAI has 90 benchmarks; Ifran has none in-tree

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
