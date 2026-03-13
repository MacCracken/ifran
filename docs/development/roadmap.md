# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage → 80%

Current: **543+ tests** across 7 crates. CI threshold: 70%.

#### Coverage by Crate

| Crate | Tests | Status | Next targets |
|---|---|---|---|
| synapse-types | 59+ | Serde roundtrips, error Display, all enums, experiment types, RAG types, RLHF types | Proto-generated code tests |
| synapse-core | 130+ | Registry, storage, lifecycle, eval, marketplace, pull, experiment store, RAG store/chunker/pipeline, RLHF store/generator | Edge cases, concurrent access |
| synapse-backends | 72+ | Router, all backends with mock HTTP, load/infer/unload, wasm backend | Streaming tests |
| synapse-train | 110 | Job scheduling, training methods, distributed, executors, experiment search/runner | Executor run with mock subprocess |
| synapse-api | 108+ | Integration tests, middleware, state, system, models, inference, openai, training, distributed, marketplace, experiment, RAG, RLHF | Streaming inference, eval, bridge endpoints |
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
- Multi-tenant support for shared deployments
