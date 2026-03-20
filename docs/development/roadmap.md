# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,290 tests** across 7 crates. CI threshold: 65%.

---

## Engineering Backlog

Hardening and operational improvements identified during security audit.

- [x] **Per-IP rate limiting** — replaced global `NotKeyed` limiter with per-IP `DashMap`-backed buckets; IP extracted from `ConnectInfo<SocketAddr>`
- [x] **Job memory eviction** — background eviction loop removes terminal jobs after configurable TTL (`job_eviction_ttl_secs`, default 24h); cleans both in-memory map and SQLite store
- [x] **Disabled tenant in-flight cancellation** — `disable_tenant` now calls `cancel_tenant_jobs()` to cancel all non-terminal jobs for the tenant
- [x] **Lineage ancestry depth limit** — `get_ancestry()` accepts `max_depth` (default 10,000 nodes); exposed as `?max_depth=` query param on `GET /lineage/:id/ancestry`

## Post-v1 Considerations

- Prompt management and versioning
