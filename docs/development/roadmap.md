# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **788 tests** across 7 crates. CI threshold: 65%.

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

## Bridge Completion (High Priority)

Items that block full SecureYeoman integration.

- [ ] **gRPC bridge server implementation** — `bridge/server.rs` is stubbed; SY callbacks (GPU allocation, progress reporting, scale-out, model registration) need real handlers
- [ ] **gRPC bridge client RPCs** — `bridge/client.rs` stubs for `RequestWorkerAssignment`, `SyncCheckpoint`, `ReportProgress`, `RequestGpuAllocation`, `RegisterCompletedModel`
- [ ] **REST SSE for training jobs** — SY dashboard relays training events via SSE; Synapse only has gRPC log streaming, needs `/training/jobs/:id/stream` SSE endpoint
- [ ] **Job crash recovery** — persist in-flight job state to SQLite so jobs survive process restarts (currently in-memory only)

## Training Feature Parity (Medium Priority)

Features SecureYeoman has that Synapse should support natively.

- [ ] **Dataset refresh & curation pipeline** — periodic re-curation, incremental updates from new data, deduplication (SY: `dataset-refresh-manager.ts`, `dataset-curator.ts`)
- [ ] **Pipeline lineage tracking** — dataset → training → evaluation → deployment provenance chain (SY: `pipeline-lineage.ts`)
- [ ] **Model versioning** — track model lineage per consumer/personality with version comparison (SY: `model-version-manager.ts`)
- [ ] **A/B testing support** — serve model variants with traffic splitting, quality tracking, auto-promotion (SY: `ab-test-manager.ts`)
- [ ] **Drift detection** — z-score-based inference quality drift from baseline, alerting, snapshot comparison (SY: `drift-detection-manager.ts`)
- [ ] **Conversation/inference quality scoring** — automatic quality scoring of inference sessions for training data filtering (SY: `conversation-quality-scorer.ts`)
- [ ] **Preference pair management** — standalone preference store for DPO/RLHF data beyond annotation sessions (SY: `preference-manager.ts`)

## Evaluation & Responsible AI (Medium Priority)

- [ ] **LLM-as-judge evaluation** — pairwise model comparison using LLM scoring rubrics (SY: `llm-judge-manager.ts`)
- [ ] **Responsible AI reporting** — cohort error analysis, fairness metrics (disparate impact, demographic parity), SHAP token attribution, data provenance (SY: `responsible-ai-manager.ts`)
- [ ] **RAG eval improvements** — adopt Mneme's retrieval optimizer pattern (Thompson Sampling bandit across ranking strategies)

## Training Pipeline Enhancements (Lower Priority)

- [ ] **Continual/online learning** — incremental training on new high-quality data with replay buffers and gradient accumulation (SY: `online-update-manager.ts`)
- [ ] **Approval gates** — human-in-the-loop approval before model deployment in experiment/training pipelines (SY: `approval-manager.ts`)
- [ ] **ML workflow orchestration** — DAG-based pipeline with step types: curate → train → eval → approve → deploy (SY has 5 ML-specific workflow step types)
- [ ] **Ollama adapter registration** — post-training integration to register LoRA adapters with local Ollama instance (SY: `registerWithOllama()`)
- [ ] **Cost-aware backend routing** — extend backend router to consider token cost alongside format/hardware when selecting inference backend

## Agnosticos OS Integration (Medium Priority)

Deeper integration with the host OS when running on Agnosticos.

- [ ] **GPU scheduling via daimon** — register GPU requirements with agent-runtime resource manager instead of direct device access; prevents OOM from concurrent inference + training jobs competing for VRAM
- [ ] **OTLP observability export** — export training metrics (loss curves, per-step timing, GPU utilization), inference latency histograms, and memory pressure events to daimon's OTLP collector for unified dashboard
- [ ] **Separate sandbox profiles** — distinct Landlock/seccomp profiles for inference (stateless, tighter) vs training (needs checkpoint writes, more resources); currently a single `GpuCompute` profile covers both
- [ ] **Dynamic capability advertising** — extend capability registration beyond static `["model-management", "inference", "training"]` to include available backends, loaded models, quantization formats, and training methods so daimon can route intelligently
- [ ] **Systemd health check** — add `ExecHealthCheck` to systemd unit using `curl http://127.0.0.1:8420/health` for restart reliability instead of relying on watchdog timeout alone
- [ ] **Token budget pool integration** — coordinate with hoosh's accounting layer so training jobs respect GPU budget pools and get backpressure when resources are exhausted
- [ ] **Encrypted model storage** — leverage LUKS-backed volumes via daimon key management for `/var/lib/synapse/models` to protect proprietary/sensitive model weights at rest
- [ ] **Service discovery via daimon** — replace hardcoded `http://127.0.0.1:9420` SY endpoint with dynamic discovery through agent-runtime service registry (`GET /v1/discover`)
- [ ] **Hardware allocator implementation** — wire up `hardware/allocator.rs` stub to coordinate with daimon's resource forecaster for multi-GPU device assignment and fair queuing across jobs

## Infrastructure Stubs to Complete

Existing code that needs wiring up.

- [ ] **OCI registry client** (`registry/oci.rs`) — Docker Registry API for container-hosted model pulling
- [ ] **Direct URL registry** (`registry/direct.rs`) — HTTP client for direct URL model downloads
- [ ] **Storage cache** (`storage/cache.rs`) — LRU model eviction policy
- [ ] **Model pool** (`lifecycle/pool.rs`) — hot-swapping loaded models based on demand

## Post-v1 Considerations

- Prompt management and versioning
