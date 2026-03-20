# Synapse CLI Reference

The `synapse` command-line interface manages local LLM inference, training, evaluation, and model distribution.

```
synapse <COMMAND>
```

---

## synapse pull

Download a model from a remote registry (Hugging Face) to local storage.

### Usage

```
synapse pull <MODEL> [--quant <QUANT>]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<MODEL>` | Yes | Model repo ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`) |
| `-q, --quant <QUANT>` | No | Quantization filter. Accepted values: `f32`, `f16`, `bf16`, `q8_0`, `q6k`, `q5_k_m`, `q5_k_s`, `q4_k_m`, `q4_k_s`, `q4_0`, `q3_k_m`, `q3_k_s`, `q2k`, `iq4_xs`, `iq3_xxs` |

### Examples

```sh
# Pull a model at default precision
synapse pull meta-llama/Llama-3.1-8B-Instruct

# Pull a specific quantization
synapse pull meta-llama/Llama-3.1-8B-Instruct --quant q4_k_m
```

---

## synapse list

List all locally available models. Displays name, format, quantization level, size, and pull date.

### Usage

```
synapse list
```

### Examples

```sh
synapse list
```

Output columns: `NAME`, `FORMAT`, `QUANT`, `SIZE`, `PULLED`.

---

## synapse run

Start an interactive inference session with a local model. Prompts are entered line-by-line; responses stream back in real time. Press `Ctrl+D` to quit.

### Usage

```
synapse run <MODEL>
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<MODEL>` | Yes | Model name or UUID |

### Examples

```sh
synapse run meta-llama/Llama-3.1-8B-Instruct
```

---

## synapse serve

Start the Synapse API server. Exposes a REST API and an OpenAI-compatible `/v1/chat/completions` endpoint.

### Usage

```
synapse serve [--bind <ADDRESS>]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `-b, --bind <ADDRESS>` | No | Bind address and port. Defaults to the value in the Synapse config (typically `0.0.0.0:8420`). |

### Examples

```sh
# Start with default bind address
synapse serve

# Start on a custom port
synapse serve --bind 0.0.0.0:9000
```

---

## synapse train

Start a local or distributed training job.

### Usage

```
synapse train --base-model <MODEL> --dataset <PATH> [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--base-model <MODEL>` | Yes | Base model name or path |
| `--dataset <PATH>` | Yes | Path to a JSONL dataset file |
| `--method <METHOD>` | No | Training method. One of: `lora` (default), `qlora`, `full`, `dpo`, `rlhf`, `distillation` |
| `--distributed` | No | Enable distributed training |
| `--world-size <N>` | No | Number of workers for distributed training (default: `2`) |
| `--strategy <STRATEGY>` | No | Distributed strategy. One of: `data_parallel` (default), `model_parallel`, `pipeline_parallel` |

### Examples

```sh
# LoRA fine-tune (default method)
synapse train --base-model meta-llama/Llama-3.1-8B-Instruct --dataset ./data/train.jsonl

# QLoRA fine-tune
synapse train --base-model meta-llama/Llama-3.1-8B-Instruct --dataset ./data/train.jsonl --method qlora

# Distributed training with 4 workers using model parallelism
synapse train \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --dataset ./data/train.jsonl \
  --method full \
  --distributed \
  --world-size 4 \
  --strategy model_parallel
```

When `--distributed` is used, the local node is automatically assigned as rank 0 (coordinator). If `--world-size` is greater than 1, additional workers must be assigned via the API before training starts.

---

## synapse status

Show system status including detected hardware and the number of models in the local catalog.

### Usage

```
synapse status
```

### Examples

```sh
synapse status
```

---

## synapse remove

Remove a locally stored model from disk and the catalog.

### Usage

```
synapse remove <MODEL> [-y]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<MODEL>` | Yes | Model name or UUID to remove |
| `-y, --yes` | No | Skip the confirmation prompt |

### Examples

```sh
# Remove with confirmation prompt
synapse remove meta-llama/Llama-3.1-8B-Instruct

# Remove without confirmation
synapse remove meta-llama/Llama-3.1-8B-Instruct -y
```

---

## synapse eval

Run an evaluation benchmark against a model. Requires the API server to be running (`synapse serve`), as evaluation sends inference requests over HTTP.

### Usage

```
synapse eval <MODEL> [--benchmark <BENCHMARK>] [--dataset <PATH>] [--sample-limit <N>]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<MODEL>` | Yes | Model name |
| `--benchmark <BENCHMARK>` | No | Benchmark type. One of: `perplexity` (default), `mmlu`, `hellaswag`, `humaneval`, `custom`. Aliases: `ppl` for perplexity, `human_eval` for humaneval. |
| `--dataset <PATH>` | No | Path to a JSONL dataset file. Required for all benchmarks. |
| `--sample-limit <N>` | No | Maximum number of samples to evaluate |

### Examples

```sh
# Run default perplexity benchmark
synapse eval llama-8b --dataset ./data/eval.jsonl

# Run MMLU with a sample limit
synapse eval llama-8b --benchmark mmlu --dataset ./data/mmlu.jsonl --sample-limit 500

# Run a custom benchmark
synapse eval llama-8b --benchmark custom --dataset ./data/my-eval.jsonl
```

---

## synapse marketplace

Peer-to-peer model marketplace commands. Search, publish, unpublish, and pull models across Synapse nodes.

### synapse marketplace search

Search the local marketplace catalog.

```
synapse marketplace search [QUERY]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `[QUERY]` | No | Search query string. Omit to list all entries. |

```sh
synapse marketplace search llama
synapse marketplace search
```

### synapse marketplace publish

Publish a local model to the marketplace so other peers can discover and pull it.

```
synapse marketplace publish <MODEL>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<MODEL>` | Yes | Name of a local model to publish |

```sh
synapse marketplace publish my-finetuned-llama
```

### synapse marketplace unpublish

Remove a model listing from the marketplace.

```
synapse marketplace unpublish <MODEL>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<MODEL>` | Yes | Name of the model to unpublish |

```sh
synapse marketplace unpublish my-finetuned-llama
```

### synapse marketplace pull

Pull a model from a remote marketplace peer. Downloads the model and verifies integrity using SHA-256 if available.

```
synapse marketplace pull <MODEL> --peer <URL>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<MODEL>` | Yes | Model name to pull |
| `--peer <URL>` | Yes | Peer URL (e.g. `http://node-2:8420`) |

```sh
synapse marketplace pull llama-8b-finetuned --peer http://node-2:8420
```

---

## synapse experiment

Autonomous hyperparameter experiment commands. Runs are defined in TOML program files that specify a search space, objective metric, and time budget.

### synapse experiment run

Launch an experiment from a TOML program file. Blocks until the experiment completes, printing the best score as trials finish.

```
synapse experiment run <PROGRAM>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<PROGRAM>` | Yes | Path to an experiment program TOML file |

```sh
synapse experiment run experiments/lr-sweep.toml
```

### synapse experiment list

List all experiments with their ID, name, status, and best score.

```
synapse experiment list
```

```sh
synapse experiment list
```

### synapse experiment status

Show detailed experiment status including per-trial results. If no ID is given, shows the latest experiment.

```
synapse experiment status [ID]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `[ID]` | No | Experiment UUID. Shows the latest experiment if omitted. |

```sh
# Show latest experiment
synapse experiment status

# Show a specific experiment
synapse experiment status a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### synapse experiment leaderboard

Display a ranked leaderboard of trials for a given experiment, sorted by the objective metric.

```
synapse experiment leaderboard <ID> [--limit <N>]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<ID>` | Yes | Experiment UUID |
| `--limit <N>` | No | Maximum number of results to show (default: `20`) |

```sh
synapse experiment leaderboard a1b2c3d4-e5f6-7890-abcd-ef1234567890
synapse experiment leaderboard a1b2c3d4-e5f6-7890-abcd-ef1234567890 --limit 10
```

### synapse experiment stop

Stop a running experiment. Trials already in progress will complete before the experiment fully stops.

```
synapse experiment stop <ID>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<ID>` | Yes | Experiment UUID |

```sh
synapse experiment stop a1b2c3d4-e5f6-7890-abcd-ef1234567890
```
