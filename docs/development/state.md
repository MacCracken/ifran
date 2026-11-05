# ifran — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile — rewrite in place, don't
> accrete; history lives in the CHANGELOG + port-ledger).

## Version

**2.3.3 — 2026-11-04 (housekeeping).** `scripts/` and
`release-automation.yml` ported from cargo to cyrius; CI resolves with
`cyrius deps --locked`; CONTRIBUTING rewritten for the Cyrius project; the
roadmap is forward-looking only, by release line. Suite **129/129**. Detail in
CHANGELOG `[2.3.3]`.

**2.3.0–2.3.2 — 2026-11-04.** cyrius `6.6.20` → `6.7.9` (git-first resolution)
→ `6.7.18`, with the deps following each pin; bayan's deprecated getters
replaced (no behavior change), and `dataset validate` now fails duplicate-key
JSONL lines (bayan 1.11.1); sweep combos honor the template's `timeout_s`
again (broken since 2.1.0). Detail in CHANGELOG `[2.3.0]`–`[2.3.2]`.

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
items are roadmapped with blockers ([`roadmap.md`](roadmap.md)). Detail in
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

- `tests/ifran.tcyr` — **129/129** on cyrius 6.7.18 (jobspec, executor incl. failure paths, keys,
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

Nothing in flight. The next work is in [`roadmap.md`](roadmap.md), by
release line: **2.4.x** is ready to pick up (the experiment auto-loop once its
spec shape is chosen, corrupt-DB length hardening, porting or retiring the
Rust-era `docker/` + `deploy/` packaging, the `[test]` manifest scope, CI
leftovers); 2.5.x waits on cross-repo triggers (Lane 4 bote-MCP on the SY
re-wire, the benchmark harness on a serving path); 2.6.x is the server stage
(Lane 6, the priority queue); later 2.x is demand- or platform-gated (Lane 5,
Lane 7, annotation sessions on tanur, agnos). Closed so far: Lanes 1–3, the
`rust-old/` removal and its audit triage — history in the CHANGELOG.
