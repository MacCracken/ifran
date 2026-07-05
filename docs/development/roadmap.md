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

### Lane 1 — executor hardening (2.1-track) — ✅ DONE 2026-07-05 (`[Unreleased]`)
- **Job timeout + reaper** — `timeout_s` enforced (poll-paced capture,
  SIGKILL + `timed-out`/exit-137; the audit's one named limitation closed);
  child PID recorded on the row at fork, and `ifran reap` orphans `running`
  rows whose PID is definitely gone (signal-0 probe; live rows left alone).
- **Quoted args** — `"double quotes"` group spaced args (escapes; unterminated
  rejects loud). Carry through CYML with `'''…'''`.
- **`ifran show <run-id>`** — full record (incl. pid) + 4 KB line-aligned log
  tail. Suite 77→100; all proven live (printf-grouping, 1 s kill of a 30 s
  sleep, kill-ifran-mid-job → reap).

### Lane 2 — operator-key producer signing (cross-repo) — ✅ DONE 2026-07-05
- Shipped as **anukūlana 1.1.1**: `gpt2-tula … --sk <operator.sk>` signs with
  the operator key (64 B seed||pk, exactly `keys init`'s layout; loader
  `anuk_sk_load`, pk = sk+32). Proven end-to-end: fresh `keys init` →
  `gpt2-tula --sk` on the real checkpoint → `store add` records **`verified`**
  for both artifacts → `store verify` sig-verified.

### Lane 3 — tarka preference ingestion (cross-repo) — ✅ DONE 2026-07-05
- Shipped as **tarka 1.1.2** (additive): `tarka --pref <prefs.jsonl>`
  ingests the `pref export` JSONL (bayan parse + akshara byte vocab) and trains
  DPO/IPO/KTO full-batch via the existing FD-gated primitives. Proven
  end-to-end on a real ifran-curated set (3 pairs + 4 thumbs): DPO/IPO rank
  3/3 with loss to ~0, KTO gap 0→238 — the curate→export→align loop is closed.

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
