# ifran — CLI Reference (the Cyrius control plane)

> The ported surface — frozen at the 2.0 cut, with additive 2.x minors (see
> `docs/api.md`). The Rust-era CLI was removed with `rust-old/` after 2.2.0.

Everything operates on the current working directory's workspace: `ifran.db`
(patra), `ifran-store/` (signed artifacts), `ifran-data/` (corpora),
`runs/logs/` (captured output). Operator keys live in `$HOME/.ifran/`.

**`IFRAN_HOME`** *(2.5)*: when set, `ifran.db`, `ifran-store/`, `ifran-data/`
and the keys (`.ifran/`) live under it instead, whatever the working directory
(it is created if missing; an empty value is the same as unset). A job spec's
relative `logdir` stays relative to the working directory.

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
- `ifran store add <file.tula> [name] [--strict] [--require-signed]` —
  validate (tula) → verify (Ed25519 vs the operator key; honest status:
  `verified` / `signed-unknown-key` / `unsigned`; structural garbage rejected)
  → content-address (sha256) → copy (atomic, 2.5) → ingest. Every add also runs
  tula's strict canonical-layout check and records the result *(2.5)*.
  `--strict` refuses a file that is not canonical; `--require-signed` refuses
  anything not `verified`. Refusals exit 1 and index nothing; without the
  flags, everything 2.4 accepted is still accepted.
- `ifran store ls` (the last column is the layout: `canonical`,
  `non-canonical`, or `-` for rows ingested before 2.5) · `ifran store verify
  <id>` (re-checks hash + structure + signature — bit-rot/tamper detection —
  and reports the layout, re-derived).

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
- **The experiment auto-loop** *(2.5)*: add a `[sweep.eval]` table (the
  `[eval]` keys — `bin`, `args`, `metric`, `logdir`, `timeout_s`; `name`
  defaults to `<sweep name>-eval` — plus `direction = "min"|"max"`, default
  min). After each combo's training run exits 0, ifran runs the eval as a
  recorded eval, linked to the sweep and the training run it judged. Its
  `args` take the combo's `{axis}` values plus `{run}` (the training run id)
  and `{log}` (the training run's log path). A failing gate fails the combo,
  and a failed training run gets no eval. With a `metric`, the sweep ends by
  ranking the combos on their eval values.
- `ifran sweep best <sweep-id> <metric-key> [min|max]` *(2.2)* — post-hoc
  leaderboard: extract the metric from every combo's training log
  (eval-extractor syntax), rank by direction (default min), print the board +
  best run. Metric-less combos are skipped loudly.
- `ifran sweep best <sweep-id>` *(2.5)* — re-rank a sweep by the evals its
  `[sweep.eval]` recorded, in the direction it recorded.
- `ifran sweeps`.

## Evals
- `ifran eval <spec.cyml>` — run an `[eval]` gate (exit code = the gate;
  optional `metric = "key"` extracts the number after `key` in the log into
  the benchmark store). **ifran's exit = the gate.**
- `ifran evals` (2.5: also lists the sweep and training run an auto-loop
  eval judged; 0 for a standalone eval).

## Preferences
- `ifran pref new <name>` · `pref pair <set> <prompt> <chosen> <rejected>
  [conf 0-100]` *(conf: preference strength, 2.2)* ·
  `pref tie|bothbad <set> <prompt> <a> <b>` *(4-state, 2.2 — exported as
  `"kind":"tie"/"bothbad"` with `a`/`b` fields; pair/unary-only consumers
  skip them loudly)* ·
  `pref good|bad <set> <prompt> <completion>` · `pref ls` ·
  `pref export <set> <out.jsonl>` (escaped JSONL).
