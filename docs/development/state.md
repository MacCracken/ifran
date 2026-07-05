# ifran — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile — rewrite in place, don't
> accrete; history lives in the CHANGELOG + port-ledger).

## Version

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

- **Cyrius pin**: `6.4.3` (cyrius.cyml).
- Deps: patra 1.12.8 · sigil 3.10.0 · tula 1.0.0 · cmdit 1.1.0 (pins +
  watch: [`dependency-watch.md`](dependency-watch.md)).
- License: **GPL-3.0-only** (policy: AGPL = desktop apps; `rust-old/` keeps
  its original AGPL).

## Tests / quality

- `tests/ifran.tcyr` — **77/77** (jobspec, executor incl. failure paths, keys,
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

Nothing in flight — Lane 1 shipped as 2.1.0 (see Version above); remaining
lanes with their triggers live in [`roadmap.md`](roadmap.md), and the
`rust-old/` removal window is open (user call). **Lane 2 CLOSED 2026-07-05**
(anukūlana 1.1.1 `--sk` — operator-key
producer signing; `store add` records `verified` end-to-end, proven on the
real checkpoint). **Lane 3 CLOSED same day** (tarka **1.1.2** `--pref` JSONL
ingestion — DPO/IPO/KTO train from an ifran-curated export; e2e proven).
Remaining cross-repo lane (bote-MCP) waits on the SY re-wire trigger.
