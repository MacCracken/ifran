# ifran — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**2.0.0 — RELEASED 2026-07-05 (THE CYRIUS PORT SHIPS: the AGNOS training
control plane).** Full cut detail in CHANGELOG `[2.0.0]`. Port history: **M0 DONE
2026-07-04**: `cyrius port` ran — the 53.6k-line Rust tree (+ Cargo/Cross/
cargo-config/osv strays) lives at `rust-old/` (reference oracle, eventual
dismissal); skeleton builds + 2/2 scaffold tests; manifest corrected
(`${file:VERSION}`, AGPL license matched to the LICENSE file — relicensing
flagged as a maintainer call). Disposition map + milestone tracker: [`port-ledger.md`](port-ledger.md).
**M1 (the job core) ✅ DONE same day**: jobspec (bayan CYML) + executor (own
fork+pipe+execve with exit-code capture + stderr merge) + patra run store +
CLI (`run`/`runs`); suite **18/18**; **proof met** — `ifran run
examples/lora-demo.cyml` drove the real `anukulana gpt2-lora` end-to-end
(33.6 s, exit 0, training log captured, run recorded). **M2 (checkpoint/model store) ✅ DONE same day**: operator key management
(`keys init/show` — getrandom → sigil ed25519, 0600 secret, no-clobber) + the
store (`store add/ls/verify` — tula structural validation, Ed25519 verify with
honest per-artifact status, sigil-sha256 content addressing + dedup,
tamper-detecting verify); suite **29/29**; **proof met** — keys → `gpt2-tula`
job (80 s) → both artifacts ingested + verified. **M3 (datasets) ✅ DONE 2026-07-05**: content-addressed text corpora with
honest stats + a REAL exact-line dedup (the Rust fake replaced) +
id-referenced from job specs (`dataset = N` / `{dataset}` substitution);
suite **43/43**; **proof met** — a 51 KB docs corpus deduped 567→356 lines
and attn11 TRAINED on it as an ifran job. **M4 (sweeps) ✅ DONE 2026-07-05**: grid + seeded-deterministic random over
job templates, within-token `{axis}` substitution, combos as sweep-tagged
first-class runs, pre-M4 schema migration; suite **53/53**; **proof met** — a
3-combo attn11 steps-sweep trained on the M3 dataset (3/3 exit 0, durations
scaling with steps). **M5 (eval runner) ✅ DONE 2026-07-05**: exit-code gates + verbatim metric
extraction into the `evals` benchmark store (referencing run records); suite
**64/64**; **proof met** — anukulana's HF-fidelity oracle as eval 1 (PASS,
maxrel=0.000001049 captured). **M6 (preference store) ✅ DONE 2026-07-05**: sets + DPO/IPO pairs + KTO
thumbs + escaped JSONL export (bayan parse-back proven); suite **77/77**.
**The v1.0 acceptance is DEMONSTRATED** — attn11 (train+sweep) + anukūlana
(fidelity eval + stored artifacts) + tarka (ALL GATES PASS) ran entirely as
ifran jobs in one workspace. **The stabilization pass is COMPLETE (2026-07-05)**: relicensed
**GPL-3.0-only** (policy: AGPL = desktop apps only; rust-old keeps AGPL);
`rust-old/` **HELD** for a release or two (its docs relocated into it); README
+ cli-reference rewritten; api.md/STABILITY/SECURITY/audit (PASS, 0 changes)/
benchmarks (orchestration overhead ~0.6 ms/job) all landed. Suite **77/77**.
**CLI rebuilt on cmdit** (verb dispatch + generated help/version/errors;
`[deps.cmdit]` 1.1.0; behavior parity regressed; embedded `IFRAN_VERSION`
literal joins cut mechanics). **2.0.0 CUT 2026-07-05** (VERSION + `IFRAN_VERSION` literal bumped; tagged by the maintainer). Next arc: post-2.0 additive lane (api.md) + `rust-old/` removal at ~2.1/2.2. NOT porting: `backends/` (dead broker), server/lineage/marketplace/
fleet/rag/hardware (owned homes) — see the ledger.

## Toolchain

- **Cyrius pin**: `6.4.3` (in `cyrius.cyml [package].cyrius`)

## Source

- Rust reference: 53612 lines at `rust-old/` (frozen, do not edit).
- Cyrius port: scaffold only — `src/main.cyr` stub.

## Tests

_Replace with parity test status once tests land._

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, vec, str, syscalls, io, args, assert

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md). The first milestone is typically Rust→Cyrius surface parity for the 53612-line subset.
