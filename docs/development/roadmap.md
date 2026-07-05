# ifran — Roadmap

> Sequencing for the shipped control plane. Live state →
> [`state.md`](state.md); the Rust→Cyrius port record →
> [`port-ledger.md`](port-ledger.md); the frozen surface →
> [`../api.md`](../api.md) + [`STABILITY.md`](../../STABILITY.md).

## Shipped — 2.0.0 (2026-07-05): the Cyrius port

The full arc ran 2026-07-04 → 05: `cyrius port` (M0) → job core (M1) → signed
model store + operator keys (M2) → datasets + real dedup (M3) → sweeps (M4) →
eval runner + benchmark store (M5) → preference store (M6) → stabilization
(api freeze, audit, benchmarks, GPL relicense) → CLI on cmdit → **2.0.0**.
Acceptance proof: attn11 trained (job + sweep), anukūlana's fidelity oracle
gated + benchmarked, its artifacts in the signed store, tarka ALL-GATES-PASS —
entirely as recorded ifran jobs. Detail: CHANGELOG `[2.0.0]` +
[`port-ledger.md`](port-ledger.md).

## Post-2.0 lanes (all additive on the frozen 2.x surface)

Ordered by expected pull, each with its trigger. None is calendared; a lane
opens when its trigger fires or the maintainer pulls it forward.

### Lane 1 — executor hardening (2.1-track; no external gate)
- **Job timeout + reaper** — `timeout_s` in the job spec, enforced (kill +
  `timed-out` status); also un-sticks `running` rows orphaned by a killed
  ifran. The audit's one named limitation.
- **Quoted args** — lift the space-split limitation (quoting in `args`).
- **`ifran show <run-id>`** — one run's full record + log tail (the listing
  tables exist; this is the drill-down).

### Lane 2 — operator-key producer signing (cross-repo, user-confirmed)
- anukūlana grows an additive `--sk <path>` so its artifacts sign with the
  operator key → `store add` records **`verified`** end-to-end (today:
  `signed-unknown-key`, honest but weaker). Small additive anukūlana minor;
  needs the user's per-repo go.

### Lane 3 — tarka preference ingestion (cross-repo, user-authorized)
- tarka grows a file-ingestion flag for the JSONL export (`pref export`) so
  DPO/IPO/KTO train from ifran-curated sets. tarka is 1.x-frozen — this is a
  user-authorized additive tarka cut; ifran's side is already done.

### Lane 4 — the bote-MCP interface (trigger: SY re-wires)
- The agnos-native control surface: expose run/store/dataset/eval verbs as
  MCP tools via **bote** when SecureYeoman re-wires off its Rust-ifran HTTP
  proxy. CLI-first remains the primary surface; the Rust REST boundary stays
  un-carried.

### Lane 5 — black-box optimization sweeps (trigger: real demand)
- GP-BO / CMA-ES over job templates — the one true unmapped gap from the
  2026-06-25 mining (agnosticos `planning/ml-product-mining.md`). Deliberately
  NOT smuggled into M4 (grid + seeded-random match the honest Rust surface);
  opens when a sibling sweep actually needs it. Prototype on rosnet+tyche.

### Lane 6 — approval / quotas / journal depth (server-stage)
- The `train/approval` + `budget` remainders from the disposition map: job
  approval gates, per-set quotas, and a **libro**-backed run journal (audit
  chain). Single-operator boxes don't need them; multi-user/server-stage does
  (aegis/kavach seams then too).

### Lane 7 — distributed execution (seema-stage)
- Multi-node job placement rides the **seema** port (the remaining Tier-A
  Rust target) — not ifran-local work. The model store is already the shared
  substrate seema's fleet distribution needs.

### Housekeeping (window opens at 2.1)
- **`rust-old/` removal** — held (maintainer, 2026-07-05) for a release or
  two; remove the tree + its `rust-old/docs/` (incl. the relocated Rust-era
  ADRs) at ~2.1/2.2 as its own confirmed step.
- **agnos-side story** — Linux-first today (fork/exec/patra all Linux); the
  on-agnos control plane rides patra's + the kernel's own arcs, not ifran's.

## The extraction watch (second-consumer rule)

The **model store** (`keys.cyr` + `store.cyr`, kept behind a crisp boundary)
is the named first-extraction candidate — it lifts to a shared lib when a
second consumer needs the load/verify path without going through ifran (the
murti-seam / hoosh "load the checkpoint I just produced" case). Do not extract
ahead of that consumer.
