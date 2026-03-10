# Training Guide

## Overview

Synapse orchestrates LLM training through Docker containers or subprocess execution. It supports LoRA, QLoRA, full fine-tuning, DPO, RLHF, and distillation.

## Quick Start: LoRA Fine-Tune

### 1. Prepare your dataset

Create a JSONL file with instruction-response pairs:

```jsonl
{"instruction": "What is Rust?", "response": "Rust is a systems programming language focused on safety and performance."}
{"instruction": "Explain LoRA", "response": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method..."}
```

### 2. Create a training config

```toml
# train-config.toml
base_model = "meta-llama/Llama-3.1-8B-Instruct"
method = "lora"
output_name = "my-custom-model"

[dataset]
path = "./my-dataset.jsonl"
format = "jsonl"

[hyperparams]
learning_rate = 2e-4
epochs = 3
batch_size = 4
gradient_accumulation_steps = 4
warmup_steps = 100
weight_decay = 0.01
max_seq_length = 2048

[lora]
rank = 16
alpha = 32.0
dropout = 0.05
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### 3. Submit the job

```bash
synapse train train-config.toml
```

### 4. Monitor progress

```bash
synapse status
```

Or via the API: `GET http://localhost:8420/training/jobs/:id`

## Training Methods

| Method | Use Case | Config `method` |
|--------|----------|----------------|
| LoRA | Most fine-tuning tasks | `lora` |
| QLoRA | Limited VRAM (4-bit quantized) | `qlora` |
| Full | Maximum quality, requires lots of VRAM | `full_fine_tune` |
| DPO | Preference alignment | `dpo` |
| RLHF | Reward model training | `rlhf` |
| Distillation | Compress a larger model | `distillation` |

## Executors

Training runs via one of three executors (configured in `synapse.toml`):

- **docker** (default): Runs in `synapse-trainer` container with all Python deps
- **subprocess**: Spawns Python directly on the host
- **native**: In-process Rust training (experimental)

## SecureYeoman Integration

SY can delegate training jobs to Synapse via the gRPC bridge:

```
SY → Synapse: SubmitTrainingJob
Synapse → SY: ReportProgress (streaming)
Synapse → SY: RequestGpuAllocation (if more resources needed)
Synapse → SY: RegisterCompletedModel (on success)
```
