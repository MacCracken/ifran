# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage — 73% Achieved

Current: **700+ tests** across 7 crates. CI threshold: 65%.

#### Coverage by Crate

| Crate | Tests | Status |
|---|---|---|
| synapse-types | 120+ | Serde roundtrips, error Display, all enums, experiment types, RAG types, RLHF types, property-based tests (proptest), invalid JSON rejection |
| synapse-core | 140+ | Registry, storage, lifecycle, eval, marketplace, pull, experiment store, RAG store/chunker/pipeline, RLHF store/generator, concurrent RwLock tests |
| synapse-backends | 80+ | Router, all backends with mock HTTP, load/infer/unload, wasm backend, concurrent DashMap tests |
| synapse-train | 120+ | Job scheduling, training methods, distributed, executors, experiment search/runner, concurrent access, error path validation |
| synapse-api | 130+ | Integration tests, middleware, state, system, models, inference, openai, training, distributed, marketplace, experiment, RAG, RLHF, eval, bridge, error paths |
| synapse-bridge | 36 | Protocol, discovery, client/server, state transitions |
| synapse-cli | 25 | Clap arg parsing, output formatting helpers |

---

## Post-v1 Considerations

- Prompt management and versioning
- Multi-tenant support for shared deployments
