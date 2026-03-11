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

Current: **327 tests** across 7 crates. CI threshold: 30%.

#### Coverage by Crate

| Crate | Tests | Status | Next targets |
|---|---|---|---|
| synapse-types | 49 | Serde roundtrips, error Display, all enums | Proto-generated code tests |
| synapse-core | 83 | Registry, storage, lifecycle, eval, marketplace | `pull/downloader.rs`, `registry/huggingface.rs` |
| synapse-backends | 39 | Router, all backend stubs, llamacpp helpers | Mock HTTP server for backend integration |
| synapse-train | 80 | Job scheduling, training methods, distributed | `executor/docker.rs`, `executor/subprocess.rs` |
| synapse-api | 39 | Integration tests, middleware auth | REST handler unit tests, OpenAI compat |
| synapse-bridge | 19 | Protocol, discovery, client/server | Tonic mock for gRPC integration |
| synapse-cli | 18 | Clap arg parsing for all commands | Command logic extraction + unit tests |

#### Staged Backlog

##### Milestone 1: 30% — Foundation (current)
- [x] All type serde roundtrips and enum coverage (`synapse-types`)
- [x] Error variant Display strings and `From` conversions
- [x] CLI arg parsing verification for all commands
- [x] Backend helper functions (`build_messages`, `parse_completion_response`)
- [x] Backend trait type tests (`ModelHandle`)
- [x] `LlamaCppBackend` construction, capabilities, port allocation
- [x] CI threshold set to 30%

##### Milestone 2: 40% — Core Logic
- [ ] `synapse-core/pull/downloader.rs` — chunked download with resume (mock HTTP)
- [ ] `synapse-core/pull/verifier.rs` — SHA256 verification
- [ ] `synapse-core/registry/huggingface.rs` — HF API resolution (mock HTTP)
- [ ] `synapse-core/registry/scanner.rs` — filesystem model scanning
- [ ] `synapse-api/state.rs` — AppState construction
- [ ] `synapse-api/rest/system.rs` — health/status endpoints
- [ ] CI threshold → 40%

##### Milestone 3: 50% — Backend Integration
- [ ] Mock HTTP server test harness (shared utility)
- [ ] `synapse-backends/ollama` — Ollama API integration tests
- [ ] `synapse-backends/vllm` — vLLM API integration tests
- [ ] `synapse-backends/llamacpp` — full load/infer cycle with mock server
- [ ] `synapse-api/rest/models.rs` — model CRUD endpoint tests
- [ ] CI threshold → 50%

##### Milestone 4: 60% — API & Training
- [ ] `synapse-api/rest/inference.rs` — inference endpoint with mock backend
- [ ] `synapse-api/rest/openai_compat.rs` — OpenAI-compatible API tests
- [ ] `synapse-api/rest/training.rs` — training lifecycle endpoints
- [ ] `synapse-train/executor/docker.rs` — Docker executor (mock subprocess)
- [ ] `synapse-train/executor/subprocess.rs` — subprocess executor tests
- [ ] CI threshold → 60%

##### Milestone 5: 70% — Bridge & CLI
- [ ] `synapse-bridge` — tonic mock for gRPC client/server integration
- [ ] `synapse-cli` — extract command logic into testable functions
- [ ] `synapse-cli` — command output formatting tests
- [ ] `synapse-api/rest/distributed.rs` — distributed training endpoint tests
- [ ] `synapse-api/rest/marketplace.rs` — marketplace endpoint tests
- [ ] CI threshold → 70%

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
