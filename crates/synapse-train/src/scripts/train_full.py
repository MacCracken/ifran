#!/usr/bin/env python3
"""Full parameter fine-tuning script for Synapse.

Trains all model parameters (no LoRA adapter). Requires significant VRAM.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def parse_config():
    parser = argparse.ArgumentParser(description="Synapse full fine-tuning")
    parser.add_argument("--config-json", type=str, help="Training config as JSON string")
    parser.add_argument("--config-file", type=str, help="Path to JSON config file")
    args = parser.parse_args()

    if args.config_json:
        return json.loads(args.config_json)
    elif args.config_file:
        return json.loads(Path(args.config_file).read_text())
    elif os.environ.get("TRAINING_CONFIG"):
        return json.loads(os.environ["TRAINING_CONFIG"])
    else:
        print("Error: provide --config-json, --config-file, or TRAINING_CONFIG env", file=sys.stderr)
        sys.exit(1)


def load_training_dataset(dataset_cfg):
    fmt = dataset_cfg["format"]
    path = dataset_cfg["path"]
    max_samples = dataset_cfg.get("max_samples")

    if fmt == "huggingface":
        ds = load_dataset(path, split=dataset_cfg.get("split", "train"))
    elif fmt == "jsonl":
        ds = load_dataset("json", data_files=path, split="train")
    elif fmt == "csv":
        ds = load_dataset("csv", data_files=path, split="train")
    elif fmt == "parquet":
        ds = load_dataset("parquet", data_files=path, split="train")
    else:
        raise ValueError(f"Unsupported dataset format: {fmt}")

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    return ds


def format_instruction(example):
    if "instruction" in example and "response" in example:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        if example.get("input"):
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['response']}"
        return {"text": text}
    elif "text" in example:
        return {"text": example["text"]}
    elif "messages" in example:
        msgs = example["messages"]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        return {"text": text}
    else:
        keys = [k for k, v in example.items() if isinstance(v, str)]
        if len(keys) >= 2:
            return {"text": f"### Instruction:\n{example[keys[0]]}\n\n### Response:\n{example[keys[1]]}"}
        return {"text": str(example)}


def main():
    config = parse_config()
    job_id = os.environ.get("JOB_ID", "unknown")
    print(f"[synapse] Job {job_id}: Starting full fine-tuning")

    hp = config["hyperparams"]
    base_model = config["base_model"]
    output_name = config.get("output_name", f"synapse-full-{job_id}")
    output_dir = f"/workspace/checkpoints/{output_name}"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model — full precision, all parameters trainable
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[synapse] Total parameters: {total_params:,}")
    print(f"[synapse] Trainable parameters: {trainable_params:,}")

    # Load dataset
    dataset = load_training_dataset(config["dataset"])
    dataset = dataset.map(format_instruction)

    # Training arguments — full fine-tuning uses lower LR and more gradient accumulation
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hp.get("epochs", 3),
        per_device_train_batch_size=hp.get("batch_size", 1),
        gradient_accumulation_steps=hp.get("gradient_accumulation_steps", 16),
        learning_rate=hp.get("learning_rate", 2e-5),
        warmup_steps=hp.get("warmup_steps", 100),
        weight_decay=hp.get("weight_decay", 0.01),
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        report_to="none",
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=hp.get("max_seq_length", 2048),
    )

    print(f"[synapse] Job {job_id}: Full fine-tuning started — {len(dataset)} samples, {hp.get('epochs', 3)} epochs")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[synapse] Job {job_id}: Full fine-tuning complete — saved to {output_dir}")


if __name__ == "__main__":
    main()
