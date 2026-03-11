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

Current: **~55%** across 236 tests. CI threshold: 45%.

Raise by 5% per milestone. Priority by coverage gap:

| Crate | Current | Priority targets |
|---|---|---|
| synapse-backends | ~10% | `llamacpp`, `ollama`, `vllm` — need mock HTTP servers |
| synapse-cli | 0% | All commands — extract testable logic or integration harness |
| synapse-api | ~64% | `rest/inference.rs`, `rest/openai_compat.rs` |
| synapse-train | ~50% | `executor/docker.rs`, `executor/subprocess.rs` |

Milestones:
- [x] **55%** — Dataset loader/validator tests, integration tests for distributed + marketplace flows
- [ ] **60%** — API integration tests for inference/streaming, training lifecycle, openai compat
- [ ] **70%** — Bridge client/server with tonic mock, CLI command logic extraction + unit tests
- [ ] **80%** — Edge cases, error paths, concurrent access patterns

---

## Post-v1 Considerations

- Prompt management and versioning
- RAG pipeline integration
- WebAssembly builds for browser-based inference
- RLHF annotation UI in desktop app
- Multi-tenant support for shared deployments
