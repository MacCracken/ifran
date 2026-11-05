# ifran — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile — rewrite in place, don't
> accrete; history lives in the CHANGELOG + port-ledger).

## Version

**2.3.1 — 2026-11-04 (toolchain maintenance).** cyrius `6.7.9` → `6.7.18`;
deps moved to their latest tags (see below); `lib/` re-vendored; bayan's
deprecated getters replaced with their `_by_cstr` equivalents (no behavior
change). No surface change, but `dataset validate` now fails JSONL lines
with a duplicate top-level key (bayan 1.11.1). Suite **120/120**, plus
bench, fuzz and `cyrius deps --locked`. Detail in CHANGELOG `[2.3.1]`.

**2.3.0 — 2026-11-04 (the cyrius 6.7 line).** cyrius `6.6.20` → `6.7.9`;
git-first resolution, so local builds compile the tags as CI does. No
surface change. Detail in CHANGELOG `[2.3.0]`.

**2.2.1–2.2.3 — 2026-09-11 → 11-04 (toolchain maintenance).** cyrius
`6.4.3` → `6.6.2` (the `Result` value form; `rust-old/` removed) → `6.6.20`,
with the deps following each pin. Detail in CHANGELOG `[2.2.1]`–`[2.2.3]`.

**2.2.0 — RELEASED 2026-07-05 (the pre-removal-audit additives).** Sweep
leaderboard (`sweep best <id> <metric> [min|max]` — eval-extractor metrics
ranked by direction) · sweep budgets (`max_trials` announced cap +
`time_budget_s` wall-clock stop) · `dataset validate <id>` (NUL/line/JSONL
checks) · 4-state preferences (`pref tie|bothbad`, exported with `a`/`b`
fields) + pair `conf 0-100` (new column, migrated). Suite **120**; all proven
live incl. tarka `--pref` skipping the new kinds loudly. The other 4 audit
items are roadmapped with blockers (roadmap § audit triage). Detail in
CHANGELOG `[2.2.0]`.

**2.1.0 — RELEASED 2026-07-05 (Lane 1 executor hardening).** `timeout_s`
enforcement (poll-paced capture, SIGKILL → `timed-out`/137) · child PID
recorded on the run row + `ifran reap` (signal-0 probe; orphans only
definitely-dead rows) · quoted `args` (escapes; unterminated rejects loud) ·
`ifran show <id>` (full record + 4 KB line-aligned log tail). Suite 77→**100**;
all proven live. Detail in CHANGELOG `[2.1.0]`. The `rust-old/` removal window
is now OPEN (~2.1/2.2 per the hold) — a user call, not automatic.

**2.0.0 — RELEASED + TAGGED 2026-07-05.** The Cyrius port shipped whole (the
M0–M6 arc + stabilization + cmdit CLI, 2026-07-04→05); cut detail in
CHANGELOG `[2.0.0]`, the port record in [`port-ledger.md`](port-ledger.md).
Post-release doc sweep done 2026-07-05 (roadmap = the post-2.0 lanes; Rust-era
ADRs/docs relocated to `rust-old/docs/`; README/CLAUDE/guides/dep-watch
rewritten; ADR 0001 + `examples/` seeded).

## Surface (frozen 2.x — `docs/api.md`)

Jobs (`run`/`runs`/`show`/`reap`, quoted args + `timeout_s` — 2.1) · operator
keys (`keys`) · signed model store (`store`) · datasets + real dedup
(`dataset`) · sweeps (`sweep`/`sweeps`) · evals + the benchmark store
(`eval`/`evals`) · preferences + JSONL export (`pref`).
CLI on **cmdit** (generated help/version/errors). Exit codes: `run` = child's,
`eval` = the gate, usage = 2.

## Toolchain & deps

- **Cyrius pin**: `6.7.18` (cyrius.cyml).
- Deps: patra 1.16.6 · sigil 3.13.15 · tula 1.1.2 · cmdit 1.5.0, the latest
  tag of each (pins + watch: [`dependency-watch.md`](dependency-watch.md)).
- **Git first**: a default `cyrius deps` / `build` / `test` resolves the
  tags, sibling checkouts or not. `CYRIUS_LOCAL=1` / `--local` builds the
  `../<dep>` checkouts from `build/local-deps/` instead.
- License: **GPL-3.0-only** (policy: AGPL = desktop apps; the removed Rust
  line keeps its original AGPL in git history).

## Tests / quality

- `tests/ifran.tcyr` — **120/120** on cyrius 6.7.18 (jobspec, executor incl. failure paths, keys,
  model store incl. tamper-detection, datasets incl. dedup, sweeps incl.
  seeded-random reproducibility, evals, prefs incl. JSON-escaping round-trip).
  All against `/tmp`.
- Audit PASS (`docs/audit/2026-07-05-audit.md`, 0 changes) · benchmarks
  captured (`docs/benchmarks.md` — orchestration overhead ~0.6 ms/job).

## Consumers / proof

The v1.0-acceptance workspace ran **attn11** (train + 3-combo sweep),
**anukūlana** (HF-fidelity oracle as eval 1, `maxrel=0.000001049`; NF4
checkpoint + adapter in the signed store), and **tarka** (ALL GATES PASS) —
entirely as recorded ifran jobs over ifran-curated datasets, with a preference
set exported. **SY** still HTTP-proxies the Rust ifran — re-wiring is Lane 4's
trigger.

## In flight / next

Nothing in flight — the audit-triage additives shipped as 2.2.0 (see Version
above); the other 4 audit items are roadmapped with blockers (roadmap § audit
triage; annotation sessions = **tanur**, the broken-out desktop model-studio
— ifran stays CLI-only, tanur consumes it; desktop-stage, BACKLOGGED). Remaining lanes with their triggers
live in [`roadmap.md`](roadmap.md). **`rust-old/` REMOVED 2026-07-05**
(post-2.2.0): the pre-removal audit is DONE
([`../audit/2026-07-05-rust-old-preremoval.md`](../audit/2026-07-05-rust-old-preremoval.md)):
zero business-logic loss; one ledger gap fixed (`lifecycle` → hoosh/murti-seam
reference); the 3 SY-contract docs (`sy-integration`/`bridge-protocol`/`api-reference` —
the Lane-4 parity target) are preserved at
[`reference/`](reference/); the 8-item triage is DONE (4 landed in 2.2.0,
4 roadmapped); the tree survives in git history. **Lane 2 CLOSED 2026-07-05**
(anukūlana 1.1.1 `--sk` — operator-key
producer signing; `store add` records `verified` end-to-end, proven on the
real checkpoint). **Lane 3 CLOSED same day** (tarka **1.1.2** `--pref` JSONL
ingestion — DPO/IPO/KTO train from an ifran-curated export; e2e proven).
Remaining cross-repo lane (bote-MCP) waits on the SY re-wire trigger.
