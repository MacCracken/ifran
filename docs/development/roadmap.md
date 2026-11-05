# ifran — Roadmap

> What comes next, by release line. Shipped work lives in the
> [CHANGELOG](../../CHANGELOG.md) (the port record in
> [`port-ledger.md`](port-ledger.md)); live state →
> [`state.md`](state.md); the frozen surface →
> [`../api.md`](../api.md) + [`STABILITY.md`](../../STABILITY.md).

Everything here is additive on the frozen 2.x surface. Nothing is calendared:
an item opens when its trigger fires or the maintainer pulls it forward. A
release line is the earliest an item could land, not a promise that it will.

## 2.4.x — patch-sized follow-ups (each waits on a maintainer decision)

- **CI leftovers.** `sbom.yml` runs GitHub's dependency review, which does not
  read `cyrius.cyml`, so it checks nothing (and its actions are unpinned):
  retire it, or replace it with a check that sees cyrius deps (CI's
  `cyrius deps --locked` already guards the lock). The remote branch
  `chore/dependency-update`, left by the deleted Rust workflow, can go.
- **Packaging: port or retire `docker/` and `deploy/`.** Both still describe
  the Rust product: an `ifran-server` with REST/gRPC ports 8420/8421, a
  Python trainer image, an `ifran serve` systemd unit with `RUST_LOG`, and
  `deploy/agnosticos/ifran.pkg.toml` with `type = "cargo"` and
  `license = "MIT"`. The Cyrius ifran is a GPL-3.0-only CLI with no server.
  Decide between a runtime image plus an agnosticos package carrying the Cyrius
  binary (the `Dockerfile.release` pattern on the agnosticos base) and removing
  them.

## 2.5.0 — the next minor (each waits on a maintainer decision)

- **Experiment auto-loop** (train → eval → compare per combo, as one
  orchestration). *Blocked on a spec decision*: how an eval template sits inside
  `[sweep]` — which metric, which gate, per combo or post hoc. Its query half
  (`sweep best`) shipped in 2.2.0. The natural 2.5.0 headline once the spec
  shape is chosen.
- **Strict checkpoint ingestion with tula 1.2.0.** `store add` validates with
  `tula_open` and records the signature status: `verified`, `unsigned`, or
  `signed-unknown-key`, all accepted. tula 1.2.0 adds `tula_validate_strict`
  (canonical layout: no overlaps, gaps or reordering, exact lengths, unique
  names) and `tula_verify_required` (unsigned is refused). *Decision*: whether
  non-canonical files are refused, accepted with a recorded flag (a column
  migration), or left alone, and whether an opt-in `--require-signed` exists.
  Either one changes what the frozen `store` verbs accept, so it ships as an
  additive minor with the default stated.

## 2.6.x — cross-repo triggers

- **Lane 4 — the bote-MCP interface** (*trigger*: SecureYeoman re-wires off
  its Rust-ifran HTTP proxy). Expose the run/store/dataset/eval verbs as MCP
  tools via **bote**. The CLI stays the primary surface; the Rust REST boundary
  is not carried over. Parity target:
  [`reference/`](reference/) (`sy-integration`, `bridge-protocol`,
  `api-reference`). `proto/` still holds the Rust line's gRPC definitions
  (`bridge`, `ifran`, `training`): keep them as part of that reference, or
  retire them once Lane 4 picks its transport.
- **Structured benchmark harness** (MMLU/HellaSwag/HumanEval formatters and
  scorers). *Blocked on a serving path*: the harness scores inference results,
  so it needs hoosh serving (or the Lane 4 re-wire) to have something to score.
  The exit-code gate + metric extraction stays the default.

## 2.7.x — server stage

- **Lane 6 — approval, quotas, journal depth.** Job approval gates, per-set
  quotas, and a **libro**-backed run journal (an audit chain), with the
  aegis/kavach seams. A single-operator box doesn't need them; multi-user,
  server-stage deployments do.
- **Job priority queue** (Low/Normal/High/Critical). *Blocked on a scheduler or
  daemon existing*: fork-per-invocation ifran has no queue to prioritize. Lands
  with Lane 6 or Lane 7.

## Later 2.x — demand- or platform-gated

- **Lane 5 — black-box optimization sweeps** (GP-BO / CMA-ES over job
  templates; *trigger*: a sibling sweep actually needs it). The one unmapped gap
  from the 2026-06-25 mining (agnosticos `planning/ml-product-mining.md`).
  Prototype on rosnet + tyche. Grid and seeded-random stay the honest default.
- **Lane 7 — distributed execution** (*rides the seema port*). Multi-node job
  placement is seema's work, not ifran-local; the model store is already the
  shared substrate seema's fleet distribution needs.
- **Annotation sessions** (lifecycle, next-unannotated cursor, stats). *Blocked
  on **tanur***, the desktop model-studio app (agnosticos `planning/tanur.md`;
  desktop stage). ifran never grows a GUI: it stays the CLI control plane, and
  tanur consumes it as its backend. The 4-state preference store is ready.
- **agnos-side story.** Linux-first today (fork/exec and patra are Linux); an
  on-agnos control plane rides patra's and the kernel's own arcs, not ifran's.

## Standing rules

- **The extraction watch (second-consumer rule).** The model store
  (`keys.cyr` + `store.cyr`, kept behind a crisp boundary) is the named
  first-extraction candidate. It lifts to a shared lib when a second consumer
  needs the load/verify path without going through ifran (the murti-seam /
  hoosh "load the checkpoint I just produced" case). Do not extract ahead of
  that consumer.
- **Toolchain pins** move per [`dependency-watch.md`](dependency-watch.md):
  `cyrius lib sync --full`, `cyrius deps`, then `cyrius deps --locked`.
