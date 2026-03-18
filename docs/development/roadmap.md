# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

### Test Coverage

Current: **1,266 tests** across 7 crates. CI threshold: 65%.

---

## Engineering Backlog

Hardening and operational improvements identified during security audit.

- [ ] **Per-IP rate limiting** — current rate limiter is global (`NotKeyed`); replace with per-IP keyed limiter using `DashMap`-backed state to prevent one client starving others
- [ ] **Job memory eviction** — `JobManager` keeps all jobs (including terminal) in memory forever; add TTL-based eviction for completed/failed/cancelled jobs (e.g., 24h retention)
- [ ] **Disabled tenant in-flight cancellation** — when a tenant is disabled, cancel their running training jobs; currently in-flight jobs run to completion
- [ ] **Lineage ancestry depth limit** — `get_ancestry()` traverses the full graph unboundedly; add configurable max depth (e.g., 10,000 nodes) to prevent OOM on deep/wide DAGs

## Post-v1 Considerations

- Prompt management and versioning
