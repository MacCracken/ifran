# 010 — Majra Persistence Backends (Redis / PostgreSQL)

## Status: Deferred (demand-gated)

## Context

Majra provides optional persistence backends:

- **Redis** (`majra::redis_backend`): `RedisPubSub`, `RedisQueue`, `RedisRateLimiter`, `RedisHeartbeatTracker` — cross-process coordination via Redis.
- **PostgreSQL** (`majra::postgres_backend`): `PostgresWorkflowStorage`, `PostgresQueueBackend` — durable DAG workflow runs and managed queue persistence.

Ifran currently uses:
- In-memory `PubSub` for events (training, GPU, progress)
- In-memory `PriorityQueue` / `ManagedQueue` for job scheduling
- In-memory `HeartbeatTracker` for fleet health
- In-memory `InMemoryWorkflowStorage` for training workflows
- SQLite (`rusqlite` + `r2d2`) for model DB, experiments, annotations, lineage, versions

## Evaluation

### Redis backend

**Pros:**
- Cross-instance pub/sub — training events visible across a multi-node cluster
- Distributed rate limiting — consistent per-IP limits across multiple ifran-server instances
- Fleet heartbeats survive server restarts — Redis TTL handles expiry natively
- Low-latency — Redis round-trip is sub-millisecond on LAN

**Cons:**
- New infrastructure dependency — operators must provision and manage Redis
- Increases attack surface (Redis auth, network exposure)
- In-memory events are sufficient for single-instance deployments (current target)
- No durability guarantees (Redis is volatile by default; persistence modes add complexity)

**Verdict:** Useful when ifran runs as a multi-instance fleet (3+ nodes). Not needed for single-instance or dev use. Gate behind a `redis` feature flag when demand arises.

### PostgreSQL backend

**Pros:**
- Durable workflow runs — training pipeline state survives crashes, enables resume
- Durable job queue — crash recovery without SQLite WAL edge cases
- Better concurrency — PostgreSQL handles concurrent writers better than SQLite
- `PostgresWorkflowStorage` is a drop-in for `InMemoryWorkflowStorage` (same trait)

**Cons:**
- Heavy infrastructure dependency — PostgreSQL is a major operational burden for a local tool
- SQLite handles ifran's write load comfortably (training jobs are minutes-to-hours, not milliseconds)
- `InMemoryWorkflowStorage` + `TrainingWorkflow` already provides crash-recovery via `resume()`
- Migration complexity — need schema versioning, connection pooling config, credential management

**Verdict:** Consider when ifran becomes a shared team service with 10+ concurrent training jobs and crash recovery is business-critical. Not justified for local/small-team use.

## Decision

Defer both backends. The current in-memory + SQLite stack is correct for ifran's current deployment model (single-instance local LLM platform). Add Redis support when multi-instance fleet is a real requirement. Add PostgreSQL when durability of training workflows becomes critical.

When adopting:
1. Add `redis` feature flag gating `majra/redis-backend` + `redis` dep
2. Add `postgres` feature flag gating `majra/postgres` + `tokio-postgres`/`deadpool-postgres` deps
3. Make storage backend selection config-driven (`[storage] backend = "sqlite" | "postgres"`)

## Consequences

- No new infrastructure dependencies for operators
- Single-instance performance is unaffected
- Multi-instance deployments must use external coordination (e.g., shared filesystem, load balancer sticky sessions) until Redis backend is adopted
- Training workflow state is lost on crash (can be mitigated by periodic SQLite snapshots of workflow runs)
