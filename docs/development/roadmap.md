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

Current: **543 tests** across 7 crates. CI threshold: 70%.

#### Coverage by Crate

| Crate | Tests | Status | Next targets |
|---|---|---|---|
| synapse-types | 59 | Serde roundtrips, error Display, all enums, experiment types | Proto-generated code tests |
| synapse-core | 130 | Registry, storage, lifecycle, eval, marketplace, pull, experiment store | Edge cases, concurrent access |
| synapse-backends | 72 | Router, all backends with mock HTTP, load/infer/unload | Streaming tests |
| synapse-train | 110 | Job scheduling, training methods, distributed, executors, experiment search/runner | Executor run with mock subprocess |
| synapse-api | 108 | Integration tests, middleware, state, system, models, inference, openai, training, distributed, marketplace, experiment | Streaming inference, eval, bridge endpoints |
| synapse-bridge | 36 | Protocol, discovery, client/server, state transitions | Tonic mock for gRPC integration |
| synapse-cli | 25 | Clap arg parsing, output formatting helpers | Command logic extraction + unit tests |

#### Remaining Milestone: 80% — Hardening
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
