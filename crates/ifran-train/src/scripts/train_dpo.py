#!/usr/bin/env python3
"""Direct Preference Optimization (DPO) training script for Ifran.

Expects a dataset with 'prompt', 'chosen', and 'rejected' columns.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import DPOConfig, DPOTrainer

import time


class TimeBudgetCallback(TrainerCallback):
    """Stops training when wall-clock time budget is exceeded."""

    def __init__(self, budget_secs):
        self.budget_secs = budget_secs
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.monotonic() - self.start_time
            if elapsed >= self.budget_secs:
                print(f"[ifran] Time budget ({self.budget_secs}s) reached after {elapsed:.0f}s — stopping")
                control.should_training_stop = True


def parse_config():
    parser = argparse.ArgumentParser(description="Ifran DPO training")
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


def main():
    config = parse_config()
    job_id = os.environ.get("JOB_ID", "unknown")
    print(f"[ifran] Job {job_id}: Starting DPO training")

    hp = config["hyperparams"]
    base_model = config["base_model"]
    output_name = config.get("output_name", f"ifran-dpo-{job_id}")
    output_dir = f"/workspace/checkpoints/{output_name}"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (QLoRA by default for DPO to save memory)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA config for DPO
    lora_cfg = config.get("lora", {})
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
    )

    # Load preference dataset (must have prompt, chosen, rejected columns)
    dataset = load_training_dataset(config["dataset"])

    # Time budget and max steps support
    max_steps_override = config.get("max_steps")
    time_budget_secs = config.get("time_budget_secs")

    callbacks = []
    if time_budget_secs:
        callbacks.append(TimeBudgetCallback(time_budget_secs))
        print(f"[ifran] Time budget: {time_budget_secs}s")

    # DPO training config
    num_epochs = hp.get("epochs", 1)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=999 if max_steps_override else num_epochs,
        max_steps=max_steps_override or -1,
        per_device_train_batch_size=hp.get("batch_size", 2),
        gradient_accumulation_steps=hp.get("gradient_accumulation_steps", 8),
        learning_rate=hp.get("learning_rate", 5e-6),
        warmup_steps=hp.get("warmup_steps", 100),
        weight_decay=hp.get("weight_decay", 0.01),
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        report_to="none",
        beta=0.1,
        max_length=hp.get("max_seq_length", 1024),
        max_prompt_length=512,
    )

    # Train
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks if callbacks else None,
    )

    print(f"[ifran] Job {job_id}: DPO training started — {len(dataset)} preference pairs")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[ifran] Job {job_id}: DPO training complete — saved to {output_dir}")


if __name__ == "__main__":
    main()
