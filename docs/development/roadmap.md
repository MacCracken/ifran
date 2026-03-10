# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

## Remaining — External / Collaborative

### SecureYeoman Bridge (SY side)
> Pushed to SecureYeoman roadmap

- [ ] SY-side integration (SY receives Synapse capability announcements, delegates jobs)
- [ ] Bidirectional job delegation (collaborative: SecureYeoman + Synapse)

### Test Coverage → 80%

Current: **~45%** across 132 tests. CI threshold: 45%.

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

## Post-v1 Considerations

- Model marketplace / shared registry between Synapse instances
- Distributed training across multiple Synapse nodes (via SY orchestration)
- Model evaluation benchmarks (automated quality scoring)
- Prompt management and versioning
- RAG pipeline integration
- WebAssembly builds for browser-based inference
- RLHF annotation UI in desktop app
- Multi-tenant support for shared deployments
