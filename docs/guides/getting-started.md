# Getting started with ifran

ifran is the AGNOS training control plane: it runs the sovereign ML siblings'
binaries as **jobs** and holds everything around them — runs, artifacts,
datasets, sweeps, evals, preferences. It never does the math.

## Build

```sh
cyrius deps                              # resolve patra/sigil/tula/cmdit + stdlib
cyrius build src/main.cyr build/ifran
cyrius test tests/ifran.tcyr             # 77 checks
```

## The workspace

ifran operates on the **current directory**: `ifran.db` (the patra database),
`ifran-store/` (signed artifacts), `ifran-data/` (corpora), `runs/logs/`
(captured output). One directory = one workspace. Operator keys are global:
`$HOME/.ifran/`.

## The golden path

```sh
ifran keys init                          # once: the operator Ed25519 keypair

# 1. curate a corpus
ifran dataset add corpus.txt my-corpus   # content-addressed + stats
ifran dataset dedup 1                    # derive an exact-line-deduped child (id 2)

# 2. train as a job (see examples/)
ifran run examples/train-job.cyml        # spawns the sibling, captures the log,
ifran runs                               # records exit/duration; exit = child's

# 3. sweep a hyperparameter
ifran sweep examples/sweep-demo.cyml     # grid over {steps}; combos = tagged runs

# 4. gate + benchmark
ifran eval examples/eval-fidelity.cyml   # exit = the gate; metric -> the store
ifran evals

# 5. store the artifacts
ifran store add ckpt.tula my-model       # tula-validate + Ed25519-verify + dedup
ifran store verify 1                     # re-check hash/structure/signature

# 6. preference data for tarka
ifran pref new align && ifran pref pair 1 "prompt" "chosen" "rejected"
ifran pref export 1 prefs.jsonl
```

Every command's `--help` is generated from the verb table (`ifran --help`,
`ifran store --help`). Full command surface:
[`../cli-reference.md`](../cli-reference.md).

## Job specs in one minute

```toml
[job]
name = "attn11-train"
bin = "/home/you/Repos/attn11/build/attn11"     # absolute path, always
args = "--preset --corpus {dataset} --steps 30" # {dataset} resolves to the store path
logdir = "runs/logs"
dataset = 2                                      # an `ifran dataset` id
```

Sweeps wrap the same shape (`[sweep]` + `[sweep.grid]` axes); evals add
`metric = "loss"` to pull a number from the log. See `examples/`.

## Layout

- `src/main.cyr` — the cmdit CLI; `src/{jobspec,run,runstore}.cyr` — the job
  core; `src/{keys,store}.cyr` — the signed model store; `src/dataset.cyr` ·
  `src/sweep.cyr` · `src/eval.cyr` · `src/pref.cyr`.
- `tests/ifran.tcyr` — the suite (all against `/tmp`).
- `rust-old/` — the preserved Rust line (reference-only; held until ~2.1/2.2).
