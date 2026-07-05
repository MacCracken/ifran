# ifran — CLI Reference (the Cyrius control plane)

> The Rust-era CLI reference lives with its code in `rust-old/docs/`. This is
> the ported surface — frozen at the 2.0 cut (see `docs/api.md`).

Everything operates on the current working directory's workspace: `ifran.db`
(patra), `ifran-store/` (signed artifacts), `ifran-data/` (corpora),
`runs/logs/` (captured output). Operator keys live in `$HOME/.ifran/`.

## Jobs
- `ifran run <job.cyml>` — execute a job spec: spawn the sibling binary
  (fork+pipe+execve, stdout+stderr captured to `runs/logs/run-<id>.log`),
  record the run; **ifran's exit = the child's exit**.
- `ifran runs` — list run records (id, name, status, exit, ms, sweep, log).

Job spec (`[job]`): `name` · `bin` (absolute) · `args` (space-split;
`{dataset}` and sweep `{axis}` placeholders substitute within tokens) ·
`logdir` (default `runs/logs`) · `dataset` (an `ifran dataset` id).

## Keys & the model store
- `ifran keys init` / `keys show` — operator Ed25519 keypair (getrandom →
  sigil; secret 0600; init refuses to overwrite).
- `ifran store add <file.tula> [name]` — validate (tula) → verify (Ed25519 vs
  the operator key; honest status: `verified` / `signed-unknown-key` /
  `unsigned`; structural garbage rejected) → content-address (sha256) → ingest.
- `ifran store ls` · `ifran store verify <id>` (re-checks hash + structure +
  signature — bit-rot/tamper detection).

## Datasets
- `ifran dataset add <file> [name]` — ingest a text corpus (content-addressed,
  byte/line stats).
- `ifran dataset dedup <id>` — derive an exact-line-deduped child (first
  occurrence kept, parent recorded, in/out counts reported).
- `ifran dataset ls`.

## Sweeps
- `ifran sweep <spec.cyml>` — expand a `[sweep]` template over `[sweep.grid]`
  axes (grid = cartesian; `mode = "random"` + `samples` + `seed` =
  deterministic draws) and run every combo as a sweep-tagged run.
- `ifran sweeps`.

## Evals
- `ifran eval <spec.cyml>` — run an `[eval]` gate (exit code = the gate;
  optional `metric = "key"` extracts the number after `key` in the log into
  the benchmark store). **ifran's exit = the gate.**
- `ifran evals`.

## Preferences
- `ifran pref new <name>` · `pref pair <set> <prompt> <chosen> <rejected>` ·
  `pref good|bad <set> <prompt> <completion>` · `pref ls` ·
  `pref export <set> <out.jsonl>` (escaped JSONL).
