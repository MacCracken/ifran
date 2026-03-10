#!/usr/bin/env python3
"""RLHF (PPO-based) training script for Synapse.

Two-phase pipeline:
1. Reward model training on preference data
2. PPO fine-tuning against the reward model
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


def parse_config():
    parser = argparse.ArgumentParser(description="Synapse RLHF training")
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
    print(f"[synapse] Job {job_id}: Starting RLHF (PPO) training")

    hp = config["hyperparams"]
    base_model = config["base_model"]
    output_name = config.get("output_name", f"synapse-rlhf-{job_id}")
    output_dir = f"/workspace/checkpoints/{output_name}"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # LoRA config
    lora_cfg = config.get("lora", {})
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
    )

    # Load policy model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        peft_config=peft_config,
    )

    # Load reward model (uses base model as reward if no separate reward model specified)
    reward_model_name = config.get("reward_model", base_model)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load prompt dataset (expects 'query' or 'prompt' column)
    dataset = load_training_dataset(config["dataset"])

    def tokenize_fn(examples):
        prompt_key = "query" if "query" in examples else "prompt"
        return tokenizer(examples[prompt_key], padding=True, truncation=True, max_length=512)

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # PPO config
    ppo_config = PPOConfig(
        learning_rate=hp.get("learning_rate", 1e-5),
        batch_size=hp.get("batch_size", 4),
        mini_batch_size=max(1, hp.get("batch_size", 4) // 2),
        gradient_accumulation_steps=hp.get("gradient_accumulation_steps", 4),
        ppo_epochs=hp.get("epochs", 4),
        log_with=None,
    )

    # PPO Trainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    print(f"[synapse] Job {job_id}: PPO training started — {len(dataset)} prompts")

    # PPO training loop
    generation_kwargs = {
        "max_new_tokens": hp.get("max_seq_length", 256),
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }

    for epoch in range(hp.get("epochs", 1)):
        for batch_idx, batch in enumerate(trainer.dataloader):
            query_tensors = [torch.tensor(ids) for ids in batch["input_ids"]]

            # Generate responses
            response_tensors = trainer.generate(query_tensors, **generation_kwargs)

            # Compute rewards
            texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            reward_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                rewards_output = reward_model(**reward_inputs.to(reward_model.device))
                rewards = [r.squeeze() for r in rewards_output.logits]

            # PPO step
            stats = trainer.step(query_tensors, response_tensors, rewards)

            if batch_idx % 10 == 0:
                mean_reward = sum(r.item() for r in rewards) / len(rewards)
                print(f"[synapse] Job {job_id}: epoch={epoch} batch={batch_idx} mean_reward={mean_reward:.4f}")

    # Save
    trainer.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[synapse] Job {job_id}: RLHF training complete — saved to {output_dir}")


if __name__ == "__main__":
    main()
