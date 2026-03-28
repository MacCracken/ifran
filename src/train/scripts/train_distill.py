#!/usr/bin/env python3
"""Knowledge distillation training script for Ifran.

Trains a smaller student model to mimic a larger teacher model's output
distribution using KL divergence loss.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback

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
    parser = argparse.ArgumentParser(description="Ifran knowledge distillation")
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


class DistillationTrainer(Trainer):
    """Custom trainer that computes distillation loss (KL divergence + CE)."""

    def __init__(self, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Student forward pass
        outputs = model(**inputs)
        student_logits = outputs.logits
        ce_loss = outputs.loss

        if self.teacher_model is not None:
            # Teacher forward pass
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

            # KL divergence loss
            T = self.temperature
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)

            # Combined loss
            loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss


def main():
    config = parse_config()
    job_id = os.environ.get("JOB_ID", "unknown")
    print(f"[ifran] Job {job_id}: Starting knowledge distillation")

    hp = config["hyperparams"]
    base_model = config["base_model"]  # Student model
    teacher_model_name = config.get("teacher_model", None)
    output_name = config.get("output_name", f"ifran-distill-{job_id}")
    output_dir = f"/workspace/checkpoints/{output_name}"

    # Load tokenizer (from teacher or student)
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name or base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load student model
    student = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Optional LoRA on student
    lora_cfg = config.get("lora")
    if lora_cfg:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("rank", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        )
        student = get_peft_model(student, peft_config)
        student.print_trainable_parameters()

    # Load teacher model (if specified)
    teacher = None
    if teacher_model_name:
        print(f"[ifran] Loading teacher model: {teacher_model_name}")
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # Load and tokenize dataset
    dataset = load_training_dataset(config["dataset"])

    def tokenize_fn(examples):
        key = "text" if "text" in examples else list(examples.keys())[0]
        return tokenizer(
            examples[key],
            padding="max_length",
            truncation=True,
            max_length=hp.get("max_seq_length", 1024),
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")

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
        per_device_train_batch_size=hp.get("batch_size", 2),
        gradient_accumulation_steps=hp.get("gradient_accumulation_steps", 8),
        learning_rate=hp.get("learning_rate", 1e-4),
        warmup_steps=hp.get("warmup_steps", 100),
        weight_decay=hp.get("weight_decay", 0.01),
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        report_to="none",
        gradient_checkpointing=True,
    )

    # Distillation trainer
    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=config.get("distillation_temperature", 2.0),
        alpha=config.get("distillation_alpha", 0.5),
        model=student,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        callbacks=callbacks if callbacks else None,
    )

    teacher_info = f"teacher={teacher_model_name}" if teacher_model_name else "self-distillation"
    print(f"[ifran] Job {job_id}: Distillation started — {len(dataset)} samples, {teacher_info}")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[ifran] Job {job_id}: Distillation complete — saved to {output_dir}")


if __name__ == "__main__":
    main()
