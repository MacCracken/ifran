# ifran — Port Ledger (Rust → Cyrius)

> The module-by-module disposition of the 53.6k-line Rust tree (`rust-old/`,
> reference-only) and the running record of what has landed in Cyrius. The
> strategic plan lives in agnosticos
> [`planning/ifran-port.md`](https://github.com/MacCracken/agnosticos/blob/main/docs/development/planning/ifran-port.md);
> this file is the in-repo tracker. `rust-old/` is the **reference oracle** —
> read it, never modify it; it is dismissed entirely when the port reaches v1.0
> parity-of-purpose (NOT line parity — most of it deliberately does not port).

## Disposition map (from the 2026-07-04 survey)

### PORTS — the control-plane core (this repo's Cyrius side)

| `rust-old/src/` | Cyrius home (planned) | Milestone | Status |
|---|---|---|---|
| `train/job`, `train/executor`, `train/approval` | `src/jobspec.cyr` + `src/run.cyr` + `src/runstore.cyr` | M1 | **✅ done** (approval/quotas later) |
| `train/checkpoint`, `registry`, `storage`, `versioning` | `src/keys.cyr` + `src/store.cyr` — tula+sigil store (**crisp internal boundary** — the named first-extraction candidate) | M2 | **✅ done** |
| `train/dataset`, `dataset` | `src/dataset.cyr` | M3 | **✅ done** (text corpora; akshara-packed stores if a consumer wants them) |
| `train/experiment`, `experiment` | `src/sweep.cyr` (grid + seeded-random; BBO stays a separate lane) | M4 | **✅ done** |
| `eval` | `src/eval.cyr` | M5 | **✅ done** |
| `preference`, `rlhf` | `src/pref.cyr` (sets/pairs/thumbs/JSONL export) | M6 | **✅ done** |
| `budget`, `audit` (thin parts) | job quotas + a libro-backed run journal | M1/M2 | pending |
| `types`, `config`, `cli` | `src/main.cyr` (CLI); specs are bayan CYML in `src/jobspec.cyr` | M0–M1 | **✅ done** |

### DOES NOT PORT (owned elsewhere, or dead by design)

| `rust-old/src/` | Fate |
|---|---|
| `backends/` (15-engine broker + router/cost/health/circuit-breaker) | **DEAD** — anti-sovereign (murti re-derivation); local inference = rosnet/tentib via hoosh; foreign engines → mehman |
| `server/` (axum REST, ~21 groups) | not carried by default — interface question open (CLI-first now; bote-MCP when SY re-wires; decide by M2) |
| `lineage` | → **itihas** (integrate) |
| `marketplace` | → **mela** |
| `fleet` | → **seema** (its own later port) |
| `rag` | → **mneme** (future retrieval lane) |
| `hardware` | → **ai-hwaccel** |
| `tenant` | deferred (single-operator sovereign box first) |
| `train/methods`, `train/scripts` (the Python shells), `train/distributed` | the siblings ARE the methods; distributed → seema-stage |
| `bridge`, `pull`, `training_events`, OTLP telemetry | dead / re-derived minimally (events → the journal) |
| **fake dedup / fake perplexity** (per the 2026-06-25 mining) | **do not port** — build real ones or omit |

### Rust-era docs (in `docs/`)

`backends.md`, `fleet-management.md`, `multi-tenancy.md`, `bridge-protocol.md`,
`hardware-acceleration.md`, `api-reference.md` describe surfaces that do NOT
port — they are reference for `rust-old/` and get pruned as milestones close
(same eventual-dismissal path as the code). `training.md` /
`evaluation-guide.md` / `cli-reference.md` get rewritten per milestone.

## Milestones

- **M0 — scaffold + inventory. ✅ DONE 2026-07-04.** `cyrius port` ran (Rust
  tree + Cargo/Cross/cargo-config/osv strays → `rust-old/`); skeleton builds
  (`build/ifran` prints ready, 2/2 scaffold tests); `cyrius.cyml` corrected to
  `${file:VERSION}` + the repo's actual license; this ledger committed.
- **M1 — job core. ✅ DONE 2026-07-04.** `src/jobspec.cyr` (CYML specs via
  bayan) + `src/run.cyr` (own fork+pipe+execve capture — stdlib `exec_capture`
  discards exit status + stderr; exit code propagated) + `src/runstore.cyr`
  (patra `runs` table, `id INT AUTOINCREMENT`, insert-as-running →
  update-on-exit) + CLI (`run`/`runs`). Suite 18/18 (incl. the exit-127
  failure path). **Proof met:** `ifran run examples/lora-demo.cyml` drove the
  real `anukulana gpt2-lora` (33.6 s, exit 0, full log captured, run recorded).
  Porting notes (patra 0-based binds / no implicit rowid; bayan `toml_get`
  returns Str; `print` is 2-arg) → CHANGELOG.
- **M2 — checkpoint/model store. ✅ DONE 2026-07-04.** `src/keys.cyr`
  (operator Ed25519, getrandom entropy, 0600 secret, no-clobber) +
  `src/store.cyr` (validate/verify/content-address/ingest; honest sig status;
  tamper-detecting `verify`). Suite 29/29. **Proof met:** keys → job
  (`gpt2-tula`, 80 s) → both artifacts ingested → verified. Follow-on:
  producers signing with the operator key (additive anukulana `--sk`, or
  sign-on-ingest).
- **M3 — datasets. ✅ DONE 2026-07-05.** `src/dataset.cyr`: content-addressed
  ingest + honest stats + REAL exact-line dedup (replacing the Rust fake) +
  id-referencing from job specs (`dataset = N` / `{dataset}`). Suite 43/43.
  **Proof met:** a 51 KB docs corpus deduped 567→356 lines and attn11 TRAINED
  on it as an ifran job (31.7 s, exit 0).
- **M4 — sweeps. ✅ DONE 2026-07-05.** `src/sweep.cyr`: grid (cartesian) +
  seeded-deterministic random over job templates; within-token `{axis}`
  substitution; combos = first-class runs tagged with the sweep id; pre-M4 db
  migration (ALTER TABLE). Suite 53/53. **Proof met:** a 3-combo attn11
  steps-sweep trained on the M3 dataset, 3/3 exit 0, durations scaling.
- **M5 — eval runner. ✅ DONE 2026-07-05.** `src/eval.cyr`: exit-code gates +
  verbatim metric extraction into the `evals` benchmark store (records
  reference run ids). Suite 64/64. **Proof met:** anukulana's HF-fidelity
  oracle ran as eval 1 — PASS, maxrel=0.000001049 captured.
- **M6 — preference store. ✅ DONE 2026-07-05.** `src/pref.cyr`: sets, DPO/IPO
  pairs, KTO thumbs, per-kind counts, escaped JSONL export (bayan-DOM
  parse-back proven). Suite 77/77. tarka file-ingestion = flagged follow-on.
- **v1.0 — ACCEPTANCE DEMONSTRATED 2026-07-05**: one workspace ran attn11
  (train + sweep), anukūlana (fidelity gate as eval, artifacts in the signed
  store), and tarka (full gate suite, "ALL GATES PASS") entirely as ifran
  jobs, over ifran-curated datasets, with an exported preference set.
  **Remaining for the cut**: the stabilization pass (api.md/STABILITY/audit/
  benchmarks per first-party standards), the license + interface maintainer
  calls, and the `rust-old/` dismissal decision.

## Open questions (maintainer calls)

1. ~~License~~ — **RESOLVED (maintainer, 2026-07-05): GPL-3.0-only.** The
   ecosystem policy is GPL-3.0-only with **AGPL reserved for desktop apps**;
   ifran is a control-plane CLI/service. LICENSE + manifest swapped;
   `rust-old/` keeps its original AGPL.
2. **Interface surface**: operating CLI-first (the Rust REST boundary was not
   carried); the standing plan is **bote-MCP when SY re-wires** — formal
   ratification still the maintainer's, but nothing blocks on it.
3. **Versioning**: the stabilization pass is COMPLETE (api.md/STABILITY/
   SECURITY/audit/benchmarks, 2026-07-05) — **2.0.0 is ready to cut**
   (maintainer bumps + tags). **`rust-old/` HELD (maintainer, 2026-07-05)**:
   retained (with its docs, now in `rust-old/docs/`) for a release or two
   past 2.0.0 before removal.
