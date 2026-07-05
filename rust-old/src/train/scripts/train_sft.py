#!/usr/bin/env python3
"""Supervised fine-tuning (SFT) script for Ifran training jobs.

Supports LoRA, QLoRA, and full fine-tuning via the --config-json flag or
TRAINING_CONFIG environment variable.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers import TrainerCallback
from trl import SFTTrainer

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
    parser = argparse.ArgumentParser(description="Ifran SFT training")
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
    """Format dataset examples into instruction-response pairs."""
    if "instruction" in example and "response" in example:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        if example.get("input"):
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['response']}"
        return {"text": text}
    elif "text" in example:
        return {"text": example["text"]}
    elif "messages" in example:
        # Chat format
        msgs = example["messages"]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        return {"text": text}
    else:
        # Use first two string columns as instruction/response
        keys = [k for k, v in example.items() if isinstance(v, str)]
        if len(keys) >= 2:
            return {"text": f"### Instruction:\n{example[keys[0]]}\n\n### Response:\n{example[keys[1]]}"}
        return {"text": str(example)}


def main():
    config = parse_config()
    job_id = os.environ.get("JOB_ID", "unknown")
    print(f"[ifran] Job {job_id}: Starting SFT training")
    print(f"[ifran] Base model: {config['base_model']}")
    print(f"[ifran] Method: {config['method']}")

    hp = config["hyperparams"]
    method = config["method"]
    base_model = config["base_model"]
    output_name = config.get("output_name", f"ifran-sft-{job_id}")
    output_dir = f"/workspace/checkpoints/{output_name}"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization if QLoRA
    model_kwargs = {"trust_remote_code": True}

    if method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif method == "lora":
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Apply LoRA if applicable
    if method in ("lora", "qlora"):
        if method == "qlora":
            model = prepare_model_for_kbit_training(model)

        lora_cfg = config.get("lora", {})
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("rank", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load dataset
    dataset = load_training_dataset(config["dataset"])
    dataset = dataset.map(format_instruction)

    # Time budget and max steps support
    max_steps_override = config.get("max_steps")
    time_budget_secs = config.get("time_budget_secs")

    callbacks = []
    if time_budget_secs:
        callbacks.append(TimeBudgetCallback(time_budget_secs))
        print(f"[ifran] Time budget: {time_budget_secs}s")

    # Training arguments
    num_epochs = hp.get("epochs", 3)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=999 if max_steps_override else num_epochs,
        max_steps=max_steps_override or -1,
        per_device_train_batch_size=hp.get("batch_size", 4),
        gradient_accumulation_steps=hp.get("gradient_accumulation_steps", 4),
        learning_rate=hp.get("learning_rate", 2e-4),
        warmup_steps=hp.get("warmup_steps", 100),
        weight_decay=hp.get("weight_decay", 0.01),
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        report_to="none",
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=hp.get("max_seq_length", 2048),
        callbacks=callbacks if callbacks else None,
    )

    print(f"[ifran] Job {job_id}: Training started — {len(dataset)} samples, {hp.get('epochs', 3)} epochs")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[ifran] Job {job_id}: Training complete — saved to {output_dir}")


if __name__ == "__main__":
    main()
