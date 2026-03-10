# Training

Synapse orchestrates LLM training via subprocess/Docker executors, avoiding embedding a Python runtime in Rust.

## Training Methods

- **Full Fine-Tune** — full parameter training
- **LoRA / QLoRA** — low-rank adaptation (4-bit quantized)
- **DPO** — Direct Preference Optimization
- **RLHF** — Reinforcement Learning from Human Feedback
- **Distillation** — model knowledge distillation

## Execution Model

Training jobs are submitted as configurations and executed by one of three executors:

1. **Docker** (default) — launches a container with the `synapse-trainer` image containing Python + PyTorch + Unsloth + PEFT + TRL
2. **Subprocess** — directly spawns `python3 scripts/train_sft.py` for environments without Docker
3. **Native** — in-process Rust training via candle/burn (experimental, for smaller models)

## Job Lifecycle

`Queued → Preparing → Running → Completed/Failed/Cancelled`

Jobs can be paused and resumed. Checkpoints are saved periodically and tracked in the checkpoint store.

## SecureYeoman Integration

SY can delegate training jobs to Synapse via the gRPC bridge. Synapse reports progress back and can request additional GPU allocation from SY when needed.
