# ifran

**عرفان — *gnosis*. The AGNOS training control plane.**

**2.0.0** — the Rust→Cyrius port, shipped 2026-07-05 (GPL-3.0-only).

ifran owns **the orchestration around a training run — never the math**. The
sovereign ML siblings ([attn11](https://github.com/MacCracken/attn11) /
[tarka](https://github.com/MacCracken/tarka) /
[tentib](https://github.com/MacCracken/tentib) /
[prajna](https://github.com/MacCracken/prajna) /
[anukūlana](https://github.com/MacCracken/anukulana)) prove the learning
primitives; ifran drives their binaries as **jobs** and holds everything around
them:

- **Jobs** — CYML specs → fork+pipe+execve with full output capture, exit-code
  propagation, quoted-args + `timeout_s` enforcement (2.1), and a patra-backed
  run store (`ifran run` / `runs` / `show` / `reap`).
- **Model store** — signed `.tula` artifacts: tula structural validation,
  Ed25519 verification against the **operator key** (`ifran keys`), sha256
  content addressing with dedup, tamper-detecting re-verification
  (`store add` / `ls` / `verify`).
- **Datasets** — content-addressed text corpora with honest stats and a REAL
  exact-line dedup deriving curated children (`dataset add` / `dedup` / `ls`);
  jobs reference datasets by id (`{dataset}` substitution).
- **Sweeps** — grid (cartesian) or seeded-deterministic random expansion of a
  job template; every combo a sweep-tagged first-class run (`sweep` / `sweeps`).
- **Evals** — sibling gates as recorded benchmarks: exit code = the gate,
  optional metric extracted verbatim from the run log (`eval` / `evals`).
- **Preferences** — DPO/IPO pairs + KTO thumbs, exported as escaped JSONL for
  tarka's preference surface (`pref new/pair/good/bad/ls/export`).

**Proven end-to-end** (the port's acceptance): in one workspace, attn11
trained (job + 3-combo sweep) on an ifran-curated dataset, anukūlana's
HF-fidelity oracle ran as a gated eval (`maxrel=0.000001049` captured), its
NF4 checkpoint + adapter landed in the signed store, and tarka's full gate
suite passed — **all as recorded ifran jobs**.

What ifran deliberately does **not** own: serving (hoosh), lineage (itihas),
marketplace (mela), fleet (seema), GPU (mabda/ai-hwaccel), foreign inference
engines (mehman) — and no training math, ever. The Rust line (53.6k lines,
AGPL) is preserved at `rust-old/` with its docs for reference, held for a
release or two before removal; the Cyrius port is **GPL-3.0-only** (ecosystem
policy: AGPL is reserved for desktop apps).

## Build & test

```sh
cyrius deps                              # resolve patra/sigil/tula + stdlib
cyrius build src/main.cyr build/ifran
cyrius test tests/ifran.tcyr             # 77 checks
./build/ifran                            # usage
```

## Docs

- [`docs/guides/getting-started.md`](docs/guides/getting-started.md) — the golden path (+ runnable [`examples/`](examples/))
- [`docs/cli-reference.md`](docs/cli-reference.md) — the command surface
- [`docs/api.md`](docs/api.md) — the frozen 2.x surface · [`STABILITY.md`](STABILITY.md)
- [`SECURITY.md`](SECURITY.md) · [`docs/audit/`](docs/audit/) · [`docs/benchmarks.md`](docs/benchmarks.md)
- [`docs/adr/0001-port-decomposition.md`](docs/adr/0001-port-decomposition.md) — why the port decomposed
- [`docs/development/roadmap.md`](docs/development/roadmap.md) — the post-2.0 lanes
- [`docs/development/port-ledger.md`](docs/development/port-ledger.md) — the
  Rust→Cyrius disposition map + milestone record

## License

GPL-3.0-only (`rust-old/` retains its original AGPL-3.0).
