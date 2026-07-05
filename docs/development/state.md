# ifran — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile — rewrite in place, don't
> accrete; history lives in the CHANGELOG + port-ledger).

## Version

**2.0.0 — RELEASED + TAGGED 2026-07-05.** The Cyrius port shipped whole (the
M0–M6 arc + stabilization + cmdit CLI, 2026-07-04→05); cut detail in
CHANGELOG `[2.0.0]`, the port record in [`port-ledger.md`](port-ledger.md).
Post-release doc sweep done 2026-07-05 (roadmap = the post-2.0 lanes; Rust-era
ADRs/docs relocated to `rust-old/docs/`; README/CLAUDE/guides/dep-watch
rewritten; ADR 0001 + `examples/` seeded).

## Surface (frozen 2.x — `docs/api.md`)

Jobs (`run`/`runs`) · operator keys (`keys`) · signed model store (`store`) ·
datasets + real dedup (`dataset`) · sweeps (`sweep`/`sweeps`) · evals + the
benchmark store (`eval`/`evals`) · preferences + JSONL export (`pref`).
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

Nothing in flight. The post-2.0 lanes (ordered, with triggers) live in
[`roadmap.md`](roadmap.md): Lane 1 executor hardening (timeout/reaper,
quoting, `show`) is the natural 2.1 opener; `rust-old/` removal window opens
~2.1/2.2. Cross-repo lanes (anukūlana `--sk`, tarka pref-ingestion, bote-MCP)
wait on their triggers/authorizations.
