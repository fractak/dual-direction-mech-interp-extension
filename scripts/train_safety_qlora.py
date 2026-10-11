#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    p.add_argument(
        "--dataset",
        default="data/safety_finetuning/train_sft.json",
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/safety_finetuning/llama31_8b_instruct_safety_lora",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=-1)
    return p.parse_args()


def load_training_dataset(path):
    raw = json.loads(Path(path).read_text())

    rows = []
    for row in raw["rows"]:
        rows.append({
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "completion": [{"role": "assistant", "content": row["response"]}],
        })

    return Dataset.from_list(rows), raw["config"]


def main():
    args = parse_args()

    dataset, dataset_config = load_training_dataset(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_length=1024,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        packing=False,
        completion_only_loss=True,
        model_init_kwargs={
            "quantization_config": quant_config,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        },
    )

    trainer = SFTTrainer(
        model=args.model_id,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model": args.model_id,
        "dataset": args.dataset,
        "dataset_config": dataset_config,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_steps": args.max_steps,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
        },
    }

    Path(args.output_dir, "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("Saved adapter:", args.output_dir)


if __name__ == "__main__":
    main()
