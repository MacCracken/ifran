# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

## Completed

- **MVP (Phases 1–4)** ✓ — Project scaffold, model pulling, llama.cpp inference, REST API
- **Phase 5: Additional Backends** ✓ — Ollama, vLLM, TensorRT (full HTTP); Candle, GGUF, ONNX (trait impl, inference pending native deps); smart router
- **Phase 6: Training** ✓ — Job manager, Docker/subprocess executors, dataset loading/validation, LoRA/QLoRA/DPO/RLHF/distillation, checkpoints, training API endpoints, Python scripts in Docker image
- **Phase 7: SY Bridge (Synapse side)** ✓ — gRPC server/client, protocol state machine, endpoint discovery
- **Phase 8: Desktop Application** ✓ — Tauri v2 + SvelteKit with models, chat, training, settings pages
- **Phase 10: Polish & Release** ✓ — Integration tests, API docs, Docker multi-arch, CI/CD, auth middleware, CHANGELOG

---

## Remaining — External / Collaborative

### Phase 7: SecureYeoman Bridge (SY side)
> Pushed to SecureYeoman roadmap

- [ ] SY-side integration (SY receives Synapse capability announcements, delegates jobs)
- [ ] Bidirectional job delegation (collaborative: SecureYeoman + Synapse)

### Phase 9: Agnosticos Integration
> Pushed to Agnosticos roadmap

- [ ] systemd service unit (`synapse.service`)
- [ ] Agnosticos package spec (`synapse.pkg.toml`)
- [ ] Agent-runtime registration as capability provider
- [ ] `/etc/synapse/synapse.toml` system-level config
- [ ] Model storage at `/var/lib/synapse/models/`

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
