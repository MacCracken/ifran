# ifran — Public API (2.x FROZEN)

> **Frozen at the 2.0.0 port cut (2026-07-05).** ifran is CLI-first (the
> operating interface — the Rust REST boundary was not carried; the standing
> plan opens a bote-MCP surface when SecureYeoman re-wires off its HTTP
> proxy — still the maintainer's call to ratify). The frozen
> surface = the CLI (commands, arguments, gate/exit semantics, spec formats,
> table schemas, export formats) + the module functions below. Additions land
> as 2.x minors (additive only); `_`-prefixed internals are out-of-freeze.
> See [`STABILITY.md`](../STABILITY.md).

## The CLI (primary surface — built on cmdit)

The CLI is cmdit-generated (verb table → help/version/error text; exit 2 =
usage per the AGNOS convention). `--help` works at the top level and per
grouped verb (`ifran store --help`).

See [`cli-reference.md`](cli-reference.md) for the full command table. Frozen
semantics worth naming:

- **Exit codes**: `ifran run` exits with the CHILD's exit code (127 = exec
  failed, 128+sig = signaled, −1→1 orchestration failure); `ifran eval` exits
  with the GATE; `ifran sweep` exits 0 iff every combo passed (its training
  run exited 0, and its `[sweep.eval]` gate too when the spec has one, 2.5).
  `ifran store add` exits 1 when a file is rejected or refused (2.5:
  `--strict`, `--require-signed`). CI-composable by construction.
- **Spec formats** (bayan TOML/CYML): `[job]`
  (name/bin/args/logdir/dataset/timeout_s — `timeout_s` added 2.1),
  `[sweep]` + `[sweep.grid]` (mode/samples/seed + axes; 2.2 budgets:
  `max_trials`, `time_budget_s`), `[eval]` (+ metric), and an optional
  `[sweep.eval]` (2.5: the experiment auto-loop — `[eval]`'s keys with `name`
  optional, defaulting to `<sweep name>-eval`, plus `direction = "min"|"max"`,
  default min; it runs after each combo whose training exited 0).
  `{dataset}` / `{axis}` placeholders substitute within tokens; a
  `[sweep.eval]`'s `args` also take `{run}` (the combo's training run id) and
  `{log}` (its log path) (2.5); `args` supports `"double quotes"` for spaced
  args (2.1).
- **Run statuses**: `running` → `done` / `failed` / `spawn-failed` /
  `timed-out` (2.1: `timeout_s` expiry, SIGKILL) / `orphaned` (2.1: set by
  `ifran reap` on rows whose recorded PID is gone).
- **Workspace layout**: `ifran.db` (patra tables: runs, sweeps, evals, models,
  datasets, psets, prefs), `ifran-store/<sha16>.tula`, `ifran-data/<sha16>.txt`,
  `runs/logs/run-<id>.log`, `$HOME/.ifran/operator.{sk,pk}`. All of these sit
  in the working directory, except the keys. With `IFRAN_HOME` set (2.5),
  `ifran.db`, `ifran-store/`, `ifran-data/` and the keys (`.ifran/`) live under
  it instead; a job spec's relative `logdir` stays relative to the working
  directory.
- **Columns added in 2.5** (in-place migrations; older rows read the
  defaults): `models.canonical` (1 = passed tula's strict layout check, −1 =
  failed it, 0 = not checked, i.e. ingested before 2.5), `evals.sweep` +
  `evals.subject` (the sweep and the training run a `[sweep.eval]` judged; 0
  for `ifran eval`), `sweeps.eval_metric` + `sweeps.eval_max` (the
  `[sweep.eval]` metric and direction; empty and 0 without one).
- **Export formats**: preference JSONL, escaped — `{"kind":"pair",...}`
  (+`"conf":N` when set, 2.2) / `{"kind":"unary",...,"label":±1}` /
  `{"kind":"tie"/"bothbad",...,"a":...,"b":...}` (4-state, 2.2; pair/unary-only
  consumers skip the new kinds loudly).

## Module surface (frozen)

- **jobspec** — `jobspec_parse(path)` / `jobspec_from_pairs(pairs)` →
  spec record; accessors `jobspec_{name,bin,argv,logdir,dataset}`.
- **run** — `run_job(js)` → child exit (records the run; resolves
  `{dataset}`); `run_last_id()` / `run_last_log()`.
- **runstore** — `runstore_open/set_path/set_sweep`, `runstore_start/finish`,
  `runstore_list`.
- **keys** — `keys_init/show/sk/pk/set_dir` (0600 secret; init no-clobber).
- **store** — `store_add(path, name)` → id (validate/verify/content-address;
  garbage rejected; honest sigstat), `store_add_policy(path, name, strict,
  require_signed)` → id or −1 (2.5; `store_add` is it with both 0),
  `store_ls`, `store_verify(id)`, `store_set_dir`.
- **dataset** — `dataset_add(path, name)` → id, `dataset_dedup(id)` → child id,
  `dataset_path(id)`, `dataset_ls`, `dataset_set_dir`.
- **sweep** — `sweep_parse(path)`, `sweep_run(path)` (0 iff all combos pass),
  `sweep_ls`, `sweep_subst(tok, name, value)`, `sweep_best(sid, key, max)`,
  `sweep_best_evals(sid)` (2.5).
- **eval** — `eval_run(path)` → gate, `eval_run_spec(js, metric, sweep,
  subject)` → gate or −1 (2.5), `eval_ls`, `eval_extract_metric(buf, n, key)`.
- **state root** (2.5) — `ifr_home()` (`IFRAN_HOME`, or 0 when unset/empty),
  `ifr_home_path(rel)`.
- **pref** — `pref_new(name)` → set id, `pref_pair(sid, p, c, r)`,
  `pref_unary(sid, p, c, ±1)`, `pref_ls`, `pref_export(sid, path)` → count.

## Out of freeze

- `_`-prefixed internals; buffer caps (`RUN_LOG_MAX`) and dir defaults
  (overridable via the `*_set_*` fns).
- Known M-scope limitations (documented, lift as 2.x minors):
  single-predicate WHERE usage, space-separated axis values. *(Lifted in 2.1:
  quoted args + job timeout/reaper — Lane 1.)*
- **Named additive lane**: tarka file-ingestion of the preference export
  (user-authorized tarka cut) · producers signing with the operator key
  (anukūlana `--sk`) · a bote-MCP interface when SY re-wires · the BBO sweep
  lane (its own trigger).
