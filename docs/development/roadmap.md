# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **958 tests** across 7 crates. CI threshold: 65%.

#### Coverage by Crate

| Crate | Tests | Status |
|---|---|---|
| synapse-types | 136 | Serde roundtrips, error Display, all enums, experiment types, RAG types, RLHF types, tenant types, property-based tests (proptest), invalid JSON rejection |
| synapse-core | 207 | Registry, storage, lifecycle, eval, marketplace, pull, experiment store, RAG store/chunker/pipeline, RLHF store/generator, tenant store, concurrent RwLock tests |
| synapse-backends | 61 | Router, all backends with mock HTTP, load/infer/unload, wasm backend, concurrent DashMap tests |
| synapse-train | 138 | Job scheduling, training methods, distributed, executors, experiment search/runner, concurrent access, error path validation |
| synapse-api | 131+ | Integration tests, middleware, state, system, models, inference, openai, training, distributed, marketplace, experiment, RAG, RLHF, eval, bridge, tenants, error paths |
| synapse-bridge | 36 | Protocol, discovery, client/server, state transitions |
| synapse-cli | 25+ | Clap arg parsing, output formatting helpers |

---

## Post-v1 Considerations

- Prompt management and versioning
