# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

## Remaining — External / Collaborative

### SecureYeoman Bridge (SY side)
> Pushed to SecureYeoman roadmap

- [ ] SY-side integration (SY receives Synapse capability announcements, delegates jobs)
- [ ] Bidirectional job delegation (collaborative: SecureYeoman + Synapse)

### Test Coverage → 80%

Current: **~45%** across 164 tests. CI threshold: 45%.

Raise by 5% per milestone. Priority by coverage gap:

| Crate | Current | Priority targets |
|---|---|---|
| synapse-backends | ~10% | `llamacpp`, `ollama`, `vllm` — need mock HTTP servers |
| synapse-cli | 0% | All commands — extract testable logic or integration harness |
| synapse-api | ~64% | `rest/inference.rs`, `rest/openai_compat.rs` |
| synapse-train | ~50% | `executor/docker.rs`, `executor/subprocess.rs` |

Milestones:
- [ ] **50%** — Mock HTTP server tests for backends (wiremock/mockito), downloader tests
- [ ] **60%** — API integration tests for inference/streaming, training lifecycle, openai compat
- [ ] **70%** — Bridge client/server with tonic mock, CLI command logic extraction + unit tests
- [ ] **80%** — Edge cases, error paths, concurrent access patterns

---

## In Progress — New Features

### Model Evaluation Benchmarks
> Types, core runner, SQLite store, REST API, CLI stub — all wired

- [x] Eval types: `EvalConfig`, `EvalResult`, `EvalStatus`, `BenchmarkKind` (Perplexity/MMLU/HellaSwag/HumanEval/Custom)
- [x] Eval runner: create/start/complete/fail runs, `run_custom_benchmark()` with closure-based inference
- [x] Eval store: SQLite `eval_results` table, CRUD
- [x] Benchmarks: `load_samples()` from JSONL, `score_exact_match()`, `score_contains_match()`
- [x] REST: `POST/GET /eval/runs`, `GET /eval/runs/:id`
- [x] CLI: `synapse eval --benchmark <kind> --dataset <path>`
- [ ] Wire inference backends to eval runner for real benchmarks
- [ ] Implement perplexity calculation
- [ ] Standard benchmark datasets (MMLU, HellaSwag, HumanEval)

### Model Marketplace
> Types, catalog, publisher, resolver, REST API, CLI — all wired

- [x] Marketplace types: `MarketplaceEntry`, `MarketplaceQuery`
- [x] SQLite catalog: publish/search/list/unpublish/count
- [x] Publisher: `create_entry()` from `ModelInfo`
- [x] Resolver: peer management, remote search stub
- [x] REST: `GET /marketplace/search`, `GET/POST /marketplace/entries`, `DELETE /marketplace/entries/:name`
- [x] CLI: `synapse marketplace search/publish/unpublish`
- [ ] Remote peer search (query `GET /marketplace/search` on peers)
- [ ] Model download/pull from marketplace entries
- [ ] Trust/verification layer for remote models

### Distributed Training
> Types and core modules complete, needs API/CLI wiring and real execution

- [x] Types: `DistributedTrainingConfig`, `DistributedStrategy`, `WorkerAssignment`, `DistributedJobState`
- [x] Coordinator: job creation, worker assignment, lifecycle management
- [x] Worker: local state, extra CLI args for distributed scripts, lifecycle
- [x] Aggregator: `AggregationPlan`, checkpoint merging command builder
- [ ] REST endpoints for distributed job management
- [ ] CLI `--distributed` flag for `synapse train`
- [ ] Wire to SY bridge for cross-node worker coordination
- [ ] Checkpoint synchronization between workers
- [ ] Federated averaging implementation

---

## Post-v1 Considerations

- Prompt management and versioning
- RAG pipeline integration
- WebAssembly builds for browser-based inference
- RLHF annotation UI in desktop app
- Multi-tenant support for shared deployments
