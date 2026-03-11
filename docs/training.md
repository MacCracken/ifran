# Training

Synapse orchestrates LLM training via subprocess/Docker executors, avoiding embedding a Python runtime in Rust.

## Training Methods

| Method | Description | Script |
|--------|-------------|--------|
| **LoRA** | Low-rank adaptation, parameter-efficient | `train_sft.py` |
| **QLoRA** | 4-bit quantized LoRA, lower VRAM | `train_sft.py` |
| **Full Fine-Tune** | Full parameter training | `train_full.py` |
| **DPO** | Direct Preference Optimization | `train_dpo.py` |
| **RLHF** | Reinforcement Learning from Human Feedback (PPO) | `train_rlhf.py` |
| **Distillation** | Knowledge transfer from teacher to student | `train_distill.py` |

## Dataset Formats

- **JSONL** — `{"instruction": "...", "response": "..."}` or `{"text": "..."}` or `{"messages": [...]}`
- **CSV** — columns auto-detected
- **Parquet** — columnar format
- **HuggingFace** — dataset hub references

The dataset loader auto-detects column structure. The validator checks schema before training begins.

## Execution Model

Training jobs are submitted as configurations and executed by one of two executors:

1. **Docker** (default) — launches a container with the `synapse-trainer` image containing Python + PyTorch + PEFT + TRL. Method-specific script is selected automatically.
2. **Subprocess** — directly spawns `python3 scripts/train_<method>.py` for environments without Docker.

Both executors pass training config as JSON (`--config-json` argument or `TRAINING_CONFIG` env var).

## Job Lifecycle

```
Queued → Running → Completed / Failed / Cancelled
```

The `JobManager` enforces a concurrent job limit (default: 2). The `JobScheduler` uses FIFO ordering. Checkpoints are saved periodically and tracked in the `CheckpointStore`, which supports listing, pruning, and LoRA adapter merging.

## Python Scripts

Located in `crates/synapse-train/src/scripts/`:

- **`train_sft.py`** — SFT with LoRA/QLoRA/full support via transformers + PEFT + TRL
- **`train_full.py`** — Full parameter fine-tuning with gradient checkpointing
- **`train_dpo.py`** — DPO with reference model and QLoRA by default
- **`train_rlhf.py`** — PPO-based RLHF with reward model (two-phase)
- **`train_distill.py`** — Knowledge distillation with custom KL divergence + CE loss

Config is passed via `--config-json`, `--config-file`, or `TRAINING_CONFIG` env var.

## Distributed Training

Synapse supports distributed training across multiple nodes using data parallelism.

### Architecture

- **Coordinator**: Manages the distributed job lifecycle, worker assignments, and aggregation
- **Workers**: Each Synapse instance participates as a worker with a unique rank
- **Aggregator**: Merges checkpoints from all workers after training completes

### Strategies

| Strategy | Description |
|----------|-------------|
| **Data Parallel** | Each worker trains on a shard of the dataset with the full model |

### CLI Usage

```bash
# Start distributed training (local node becomes rank 0)
synapse train --base-model meta-llama/Llama-3.1-8B --dataset data.jsonl \
  --distributed --world-size 2 --strategy data_parallel
```

### API Workflow

1. Create a distributed job: `POST /training/distributed/jobs`
2. Assign workers (one per rank): `POST /training/distributed/jobs/:id/workers`
3. Start the job: `POST /training/distributed/jobs/:id/start`
4. Workers report completion: `POST /training/distributed/jobs/:id/workers/:rank/complete`
5. Aggregate checkpoints: `POST /training/distributed/jobs/:id/aggregate`

### Checkpoint Synchronization

Workers save checkpoints to `<output_dir>/worker-<rank>/`. After all workers complete, the aggregator merges checkpoints using averaging or weighted averaging. The SY bridge coordinates checkpoint sync between nodes via `SyncCheckpoint` RPCs.

### Federated Averaging

For scenarios where workers train independently on local data, federated averaging combines model weights:

```bash
# Aggregate with equal weights (default)
POST /training/distributed/jobs/:id/aggregate
{"output_dir": "/output", "method": "average"}

# Aggregate with custom weights per worker
POST /training/distributed/jobs/:id/aggregate
{"output_dir": "/output", "method": "weighted_average"}
```

## SecureYeoman Integration

SY can delegate training jobs to Synapse via the gRPC bridge. Synapse reports progress back and can request additional GPU allocation from SY when needed. For distributed training, SY coordinates cross-node worker assignments via `RequestWorkerAssignment` and checkpoint synchronization via `SyncCheckpoint` RPCs.
