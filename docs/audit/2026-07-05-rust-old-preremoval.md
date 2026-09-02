# rust-old/ pre-removal audit — 2026-07-05

> Re-derived module-by-module sweep of `rust-old/` (53.6k lines) against the
> shipped Cyrius 2.1.0 surface, run BEFORE the `rust-old/` removal window
> (open as of 2.1.0) is exercised. Method: three parallel readers over
> train/ · {dataset,eval,experiment,preference,rlhf}/ · plumbing
> (lifecycle/budget/audit/registry/storage/versioning/cli/config/types/
> events/pull/lineage), with load-bearing claims re-verified against source —
> per the audit discipline, the port-ledger's dispositions were *diffed
> against the tree*, not trusted.

## Headline findings

1. **`lifecycle/` (645 lines) was MISSING from the disposition map** — the
   one genuine ledger gap. It is model load/unload/swap orchestration across
   inference *backends* (VRAM budgeting, loaded-model registry, tenant
   accounting): the dead broker's companion, not a control-plane feature.
   **Correct disposition: → the hoosh/murti-seam lane** (the
   "load↔place↔dispatch↔pool seam prototypes inside hoosh's local provider"
   decision) as *reference*, nothing for ifran. Ledger row added.
2. **The SY bridge contract lives only in rust-old/** — SY's
   `routes/ifran_proxy.rs` still proxies `/api/v1/ifran/*` to the Rust
   server: models CRUD/pull · inference (+stream) · health/status/GPU
   telemetry · training jobs (list/submit/get/**cancel**/checkpoints/
   metrics/**stream**) · eval runs · experiments. `docs/sy-integration.md`
   (88 lines) + `docs/bridge-protocol.md` + `docs/api-reference.md` document
   it. **These three docs are the Lane-4 (bote-MCP re-wire) parity target and
   must be preserved out of rust-old/ before deletion.**
3. **No fake was missed** — the two known fakes (membership-counter "dedup",
   sliding-window "perplexity") stay banned; the Cyrius dedup is real and
   better. `registry/huggingface.rs` is all-mockito (sound but never
   exercised against the real API) — dies with the tree, mela re-derives.
4. **Zero business-logic loss in the shipped scope.** Everything in the
   M1–M6 charter is covered or deliberately simpler (flat sha256 store vs
   LRU-cache/layout/encryption; CYML per-invocation specs vs global TOML;
   identity-as-hash vs a version tree).

## Genuinely-missed features (real code, no Cyrius home, no skip-list entry)

Ranked; each is a *decision*, not an automatic port:

| # | Feature | rust-old anchor | Assessment |
|---|---------|-----------------|------------|
| 1 | **Sweep leaderboard** (best-trial ranking, Minimize/Maximize direction) | `experiment/store.rs:164-194,404-447` | Small + real. Natural additive: a `sweep best <id>`/leaderboard query joining sweep-tagged runs to their eval metrics. |
| 2 | **Sweep budgets** (`max_trials` cap, experiment-level time budget) | `types/experiment.rs:104-115` | Small + real. Guards runaway grids; per-run `timeout_s` (2.1) covers the per-trial half already. |
| 3 | **Experiment auto-loop** (train → eval → compare per trial, one orchestration) | `experiment/runner.rs` | Medium. Today a sweep runs combos and evals are separate gates; the loop that ties them is the piece. Builds directly on #1. |
| 4 | **Dataset validation** (`dataset validate <id>` — JSONL/CSV format checks, first-N error report) | `dataset/validator.rs` | Small + real. Catches format errors before a training job burns time. |
| 5 | **Job priority queue** (Low/Normal/High/Critical, FIFO within tier) | `job/scheduler.rs:16-57` | Real but premature: fork-per-invocation ifran has no queue to prioritize. Revisit when a scheduler/daemon lands (L6/L7 adjacent). |
| 6 | **4-state pair labels** (Tie / BothBad beyond chosen/rejected) + pair confidence (`score_delta`) | `types/rlhf.rs:23-31`, `preference/store.rs:20` | Small + real. Extends `pref pair`; KTO ±1 covers the unary side. Matters if human annotation gets serious. |
| 7 | **Annotation sessions** (lifecycle, next-unannotated cursor, stats) | `rlhf/store.rs:87-365` | Real; an *interactive* collection workflow over the pref store. The graphical surface is **tanur** (the broken-out desktop model-studio, desktop-stage) — ifran stays CLI; tanur consumes it. |
| 8 | **Structured benchmark harness** (MMLU/HellaSwag/HumanEval formatters + exact/contains scorers) | `eval/benchmarks.rs:34-183` | Real scorers, but the harness consumes *inference results* — gated on a serving story (hoosh), not free-standing. The exit-code-gate + metric-extraction design is deliberate; revisit with Lane 4/serving. |

## Deferred-with-owner (correctly dispositioned; re-verified real)

- **Approval gates** (reviewer/comment/timestamps — `train/approval/gate.rs`)
  + **budget/quota checker** (hoosh GPU-hours query, permissive fallback —
  `budget/checker.rs`) + **HMAC-chained audit trail** (in-memory only —
  `audit.rs`): all real, all **Lane 6** (approval/quotas/libro-journal). The
  libro journal should be persistent + fuzz-gated where the Rust chain was
  RAM-only; `audit.rs` is reference-quality for it.
- **Job cancel** (executor `cancel()`, SIGKILL/docker-stop): the 2.1 timeout
  infra is the mechanism; an `ifran cancel <run-id>` verb is a natural L6/L4
  companion (SY's proxy exposes cancel).
- **lineage/store.rs** (DAG nodes/edges, real schema) → **itihas**: keep as
  the schema reference until the itihas bridge lands.
- **Docker executor** → the ecosystem sandbox is **kavach**, not docker;
  re-derive there if job isolation is wanted.
- **training_events / pull / registry / storage-cache-layout-encryption /
  versioning / server / marketplace / fleet / rag / hardware / tenant /
  distributed / methods+scripts**: dead or owned elsewhere, as mapped.

## Non-src surfaces

- `tests/` (server-feature REST + async-concurrency tests), `benches/`
  (cosine/GGUF-estimate/cache), `fuzz/` (config/audit/inference targets):
  die with the tree. One carry-forward *pattern*: the Lane-6 journal should
  ship with a fuzz gate like `fuzz_audit_chain` had.
- `docs/adr/` (10 Rust ADRs): already relocated into `rust-old/docs/`;
  dismissed with it by design.
- **`docs/{sy-integration,bridge-protocol,api-reference}.md`: preserve before
  removal** (finding #2).

## Verdict

`rust-old/` is safe to remove once (a) the three SY-contract docs are copied
out (→ `docs/development/reference/`, suggested), and (b) the user has
triaged the 8-item missed list above. **Triage DONE 2026-07-05 (user):**
items 1/2/4/6 (leaderboard, sweep budgets, dataset validate, 4-state pairs +
conf) **SHIPPED as 2.2.0** (suite 100→120, all proven live); items 3/5/7/8 **ROADMAPPED with blockers** in
[`../development/roadmap.md`](../development/roadmap.md) § Pre-removal audit
triage (auto-loop → sweep-eval spec decision; priority → scheduler/daemon;
annotation sessions → tanur, the broken-out desktop model-studio app,
desktop-stage BACKLOGGED — ifran stays CLI; benchmark harness → serving path).
Remaining precondition: copy out the SY-contract docs. Nothing else in the
tree carries unique value: the remaining ~45k lines are dead scope, owned
elsewhere, or deliberately simplified away.
