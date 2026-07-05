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
- `ifran show <run-id>` *(2.1)* — one run's full record (incl. pid) + the log
  tail (last 4 KB, line-aligned).
- `ifran reap` *(2.1)* — un-stick `running` rows orphaned by a killed ifran:
  a row is orphaned only when its recorded PID is definitely gone (signal-0
  probe); live PIDs are left alone (another ifran may own them). Idempotent.

Job spec (`[job]`): `name` · `bin` (absolute) · `args` (space-split, with
`"double quotes"` grouping spaced args *(2.1)* — `\"`/`\\` escape inside
quotes; carry quotes through CYML with a TOML `'''…'''` literal string;
`{dataset}` and sweep `{axis}` placeholders substitute within tokens) ·
`logdir` (default `runs/logs`) · `dataset` (an `ifran dataset` id) ·
`timeout_s` *(2.1)* (0/absent = none; on expiry the child is SIGKILLed and
the run records `timed-out`, exit 137).

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
- `ifran dataset validate <id>` *(2.2)* — format checks before training: NUL
  scan (hard fail), line stats, JSONL per-line parse when the corpus opens
  with `{` (first 10 offenders reported; any = fail). Exit 0 = VALID.

## Sweeps
- `ifran sweep <spec.cyml>` — expand a `[sweep]` template over `[sweep.grid]`
  axes (grid = cartesian; `mode = "random"` + `samples` + `seed` =
  deterministic draws) and run every combo as a sweep-tagged run. Budgets
  *(2.2)*: `max_trials` caps combos run (announced); `time_budget_s` stops
  launching new combos when the wall clock passes (the running combo
  finishes).
- `ifran sweep best <sweep-id> <metric-key> [min|max]` *(2.2)* — post-hoc
  leaderboard: extract the metric from every combo log (eval-extractor
  syntax), rank by direction (default min), print the board + best run.
  Metric-less combos are skipped loudly.
- `ifran sweeps`.

## Evals
- `ifran eval <spec.cyml>` — run an `[eval]` gate (exit code = the gate;
  optional `metric = "key"` extracts the number after `key` in the log into
  the benchmark store). **ifran's exit = the gate.**
- `ifran evals`.

## Preferences
- `ifran pref new <name>` · `pref pair <set> <prompt> <chosen> <rejected>
  [conf 0-100]` *(conf: preference strength, 2.2)* ·
  `pref tie|bothbad <set> <prompt> <a> <b>` *(4-state, 2.2 — exported as
  `"kind":"tie"/"bothbad"` with `a`/`b` fields; pair/unary-only consumers
  skip them loudly)* ·
  `pref good|bad <set> <prompt> <completion>` · `pref ls` ·
  `pref export <set> <out.jsonl>` (escaped JSONL).
