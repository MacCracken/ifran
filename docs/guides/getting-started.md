# Getting started with ifran

ifran is the AGNOS training control plane: it runs the sovereign ML siblings'
binaries as **jobs** and holds everything around them — runs, artifacts,
datasets, sweeps, evals, preferences. It never does the math.

## Build

```sh
cyrius deps                              # resolve patra/sigil/tula/cmdit + stdlib
cyrius build src/main.cyr build/ifran
cyrius test tests/ifran.tcyr             # 183 checks
```

## The workspace

ifran operates on the **current directory**: `ifran.db` (the patra database),
`ifran-store/` (signed artifacts), `ifran-data/` (corpora), `runs/logs/`
(captured output). One directory = one workspace. Operator keys are global:
`$HOME/.ifran/`. To pin the state to one place whatever the working directory,
set `IFRAN_HOME`: `ifran.db`, `ifran-store/`, `ifran-data/` and the keys
(`.ifran/`) then live under it (2.5).

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
ifran sweep examples/sweep-eval-demo.cyml  # 2.5: + a [sweep.eval] per combo, ranked
ifran sweep best 2                       # re-rank that sweep by its evals

# 4. gate + benchmark
ifran eval examples/eval-fidelity.cyml   # exit = the gate; metric -> the store
ifran evals

# 5. store the artifacts
ifran store add ckpt.tula my-model       # tula-validate + Ed25519-verify + dedup
                                         # (2.5: --strict, --require-signed)
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

## Containers and services

ifran is one static binary, so it packages simply. Each container run or unit
start is one command against the state in `/var/lib/ifran` (`ifran.db`, run
logs, keys under `.ifran/`).

```sh
# image from source (docker/Dockerfile), state on a named volume
docker build -f docker/Dockerfile --build-arg VERSION="$(cat VERSION)" -t ifran .
docker run --rm -v ifran-data:/var/lib/ifran ifran runs

# GPU jobs: stage sibling binaries in docker/siblings/, pass the render node
docker build -f docker/Dockerfile.trainer -t ifran-trainer .
docker run --rm -v ifran-data:/var/lib/ifran --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" ifran-trainer run jobs/train.cyml
```

`docker/docker-compose.yml` wraps both (`--profile gpu` for the trainer).
On a host, `deploy/` has the systemd units: `ifran-job@.service` runs
`/etc/ifran/jobs/<name>.cyml` as a unit that exits with the job's code
(`deploy/ifran-training.conf` is its GPU drop-in), and `ifran.timer` runs
`ifran reap` at boot and hourly. `deploy/agnosticos/ifran.cyml` is the package
recipe that installs all of it.

## Layout

- `src/main.cyr` — the cmdit CLI; `src/{jobspec,run,runstore}.cyr` — the job
  core; `src/{keys,store}.cyr` — the signed model store; `src/dataset.cyr` ·
  `src/sweep.cyr` · `src/eval.cyr` · `src/pref.cyr`.
- `tests/ifran.tcyr` — the suite (all against `/tmp`).
- `docker/` · `deploy/` — container images, systemd units, the package recipe.
- The Rust line (`rust-old/`) was removed after 2.2.0; it survives in git history.
