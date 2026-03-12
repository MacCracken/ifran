# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

## Remaining — External / Collaborative

### SecureYeoman Bridge (SY side)
> Synapse side complete — remaining work pushed to SecureYeoman roadmap

Synapse-side bridge integration is fully wired:
- REST endpoints for bridge status/connect/heartbeat
- Auto-connect + heartbeat on server startup
- Training job events reported to SY (start/cancel/complete)
- Distributed training worker coordination via SY (assign/checkpoint sync)
- System status includes bridge state

Remaining (SY side):
- [ ] SY-side integration (SY receives Synapse capability announcements, delegates jobs)
- [ ] Bidirectional job delegation (collaborative: SecureYeoman + Synapse)
- [ ] Wire actual tonic gRPC transport (currently stub — framework ready, awaits SY server)

### Test Coverage → 80%

Current: **510 tests** across 7 crates. CI threshold: 70%.

#### Coverage by Crate

| Crate | Tests | Status | Next targets |
|---|---|---|---|
| synapse-types | 49 | Serde roundtrips, error Display, all enums | Proto-generated code tests |
| synapse-core | 123 | Registry, storage, lifecycle, eval, marketplace, pull | Edge cases, concurrent access |
| synapse-backends | 72 | Router, all backends with mock HTTP, load/infer/unload | Streaming tests |
| synapse-train | 97 | Job scheduling, training methods, distributed, executors | Executor run with mock subprocess |
| synapse-api | 108 | Integration tests, middleware, state, system, models, inference, openai, training, distributed, marketplace | Streaming inference, eval, bridge endpoints |
| synapse-bridge | 36 | Protocol, discovery, client/server, state transitions | Tonic mock for gRPC integration |
| synapse-cli | 25 | Clap arg parsing, output formatting helpers | Command logic extraction + unit tests |

#### Staged Backlog

##### Milestone 1: 30% — Foundation (current)
- [x] All type serde roundtrips and enum coverage (`synapse-types`)
- [x] Error variant Display strings and `From` conversions
- [x] CLI arg parsing verification for all commands
- [x] Backend helper functions (`build_messages`, `parse_completion_response`)
- [x] Backend trait type tests (`ModelHandle`)
- [x] `LlamaCppBackend` construction, capabilities, port allocation
- [x] CI threshold set to 30%

##### Milestone 2: 40% — Core Logic (complete)
- [x] `synapse-core/pull/downloader.rs` — chunked download with resume (mock HTTP via mockito)
- [x] `synapse-core/pull/verifier.rs` — SHA256/BLAKE3 verification, verify_auto, edge cases
- [x] `synapse-core/registry/huggingface.rs` — HF API resolution (mock HTTP via mockito)
- [x] `synapse-core/registry/scanner.rs` — filesystem model scanning, format detection
- [x] `synapse-api/state.rs` — AppState construction, cloneability, database creation
- [x] `synapse-api/rest/system.rs` — health/status endpoint unit tests
- [x] CI threshold → 40%

##### Milestone 3: 50% — Backend Integration (complete)
- [x] Mock HTTP server test harness (mockito for all backends)
- [x] `synapse-backends/ollama` — load/unload/infer/health with mock HTTP, message building, capabilities
- [x] `synapse-backends/vllm` — load/unload/infer/health with mock HTTP, OpenAI response parsing
- [x] `synapse-backends/llamacpp` — infer cycle with mock server (mock instance injection), process lifecycle
- [x] `synapse-api/rest/models.rs` — list/get/delete unit tests with real AppState
- [x] CI threshold → 50%

##### Milestone 4: 60% — API & Training (complete)
- [x] `synapse-api/rest/inference.rs` — InferenceBody serde, no-model error paths
- [x] `synapse-api/rest/openai_compat.rs` — ChatCompletionRequest serde, list_models, no-model error paths
- [x] `synapse-api/rest/training.rs` — full CRUD lifecycle (create/list/get/cancel), auto_start, job_to_response conversion
- [x] `synapse-train/executor/docker.rs` — construction, container naming, cancel behavior, config serialization
- [x] `synapse-train/executor/subprocess.rs` — construction, cancel with process kill, config serialization
- [x] `synapse-train/executor/mod.rs` — extracted `script_for_method()` with full coverage
- [x] CI threshold → 60%

##### Milestone 5: 70% — Bridge & CLI (complete)
- [x] `synapse-bridge` — protocol, client, server, discovery additional coverage
- [x] `synapse-cli` — output formatting helper tests (format_size, truncate)
- [x] `synapse-api/rest/distributed.rs` — full handler unit tests (create, list, get, assign, start, fail, worker_completed, serde)
- [x] `synapse-api/rest/marketplace.rs` — handler unit tests (search, list, unpublish, serde, entry_to_response)
- [x] CI threshold → 70%

##### Milestone 6: 80% — Hardening
- [ ] Error paths and edge cases across all crates
- [ ] Concurrent access patterns (DashMap, RwLock contention)
- [ ] Streaming inference end-to-end tests
- [ ] `synapse-api/rest/eval.rs` — evaluation endpoint tests
- [ ] `synapse-api/rest/bridge.rs` — bridge REST endpoint tests
- [ ] Property-based tests for type invariants
- [ ] CI threshold → 80%

---

## Post-v1 Considerations

- Prompt management and versioning
- RAG pipeline integration
- WebAssembly builds for browser-based inference
- RLHF annotation UI in desktop app
- Multi-tenant support for shared deployments
