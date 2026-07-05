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
  with the GATE. CI-composable by construction.
- **Spec formats** (bayan TOML/CYML): `[job]` (name/bin/args/logdir/dataset),
  `[sweep]` + `[sweep.grid]` (mode/samples/seed + axes), `[eval]` (+ metric).
  `{dataset}` / `{axis}` placeholders substitute within tokens.
- **Workspace layout**: `ifran.db` (patra tables: runs, sweeps, evals, models,
  datasets, psets, prefs), `ifran-store/<sha16>.tula`, `ifran-data/<sha16>.txt`,
  `runs/logs/run-<id>.log`, `$HOME/.ifran/operator.{sk,pk}`.
- **Export formats**: preference JSONL
  (`{"kind":"pair",...}` / `{"kind":"unary",...,"label":±1}`, escaped).

## Module surface (frozen)

- **jobspec** — `jobspec_parse(path)` / `jobspec_from_pairs(pairs)` →
  spec record; accessors `jobspec_{name,bin,argv,logdir,dataset}`.
- **run** — `run_job(js)` → child exit (records the run; resolves
  `{dataset}`); `run_last_id()` / `run_last_log()`.
- **runstore** — `runstore_open/set_path/set_sweep`, `runstore_start/finish`,
  `runstore_list`.
- **keys** — `keys_init/show/sk/pk/set_dir` (0600 secret; init no-clobber).
- **store** — `store_add(path, name)` → id (validate/verify/content-address;
  garbage rejected; honest sigstat), `store_ls`, `store_verify(id)`,
  `store_set_dir`.
- **dataset** — `dataset_add(path, name)` → id, `dataset_dedup(id)` → child id,
  `dataset_path(id)`, `dataset_ls`, `dataset_set_dir`.
- **sweep** — `sweep_parse(path)`, `sweep_run(path)` (0 iff all combos exit 0),
  `sweep_ls`, `sweep_subst(tok, name, value)`.
- **eval** — `eval_run(path)` → gate, `eval_ls`,
  `eval_extract_metric(buf, n, key)`.
- **pref** — `pref_new(name)` → set id, `pref_pair(sid, p, c, r)`,
  `pref_unary(sid, p, c, ±1)`, `pref_ls`, `pref_export(sid, path)` → count.

## Out of freeze

- `_`-prefixed internals; buffer caps (`RUN_LOG_MAX`) and dir defaults
  (overridable via the `*_set_*` fns).
- Known M-scope limitations (documented, lift as 2.x minors): space-split args
  (no quoting), no job timeout enforcement, single-predicate WHERE usage,
  space-separated axis values.
- **Named additive lane**: tarka file-ingestion of the preference export
  (user-authorized tarka cut) · producers signing with the operator key
  (anukūlana `--sk`) · a bote-MCP interface when SY re-wires · the BBO sweep
  lane (its own trigger).
