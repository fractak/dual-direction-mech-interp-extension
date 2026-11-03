#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    p.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[1] / "data/safety_finetuning/train_sft.json"),
    )
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "artifacts/adapters/safety_lora"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--revision", help="Immutable model Hub commit SHA")
    p.add_argument("--kind", choices=("all", "benign", "safety"), default="all")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=-1)
    return p.parse_args()


def load_training_dataset(path, kind="all"):
    raw = json.loads(Path(path).read_text())

    rows = []
    for row in raw["rows"]:
        if kind != "all" and row["kind"] != kind:
            continue
        rows.append(
            {
                "prompt": [{"role": "user", "content": row["prompt"]}],
                "completion": [{"role": "assistant", "content": row["response"]}],
            }
        )

    if not rows:
        raise ValueError("No training rows selected")

    from datasets import Dataset

    return Dataset.from_list(rows), raw["config"]


def main():
    args = parse_args()
    output = Path(args.output_dir)
    if output.resolve().is_relative_to((ROOT / "evidence").resolve()):
        raise ValueError("Training output cannot replace frozen evidence")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Adapter output is not empty; choose a new --output-dir")

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    dataset, dataset_config = load_training_dataset(args.dataset, args.kind)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
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
            "revision": args.revision,
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

    trainer.save_state()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    from ddmi.provenance import adapter_manifest, runtime_metadata

    metadata = {
        "checkpoint": "safety_lora",
        "base_model": args.model_id,
        "model_revision": args.revision,
        "runtime": runtime_metadata(),
        "training_subset": args.kind,
        "selected_rows": len(dataset),
        "selected_rows_sha256": hashlib.sha256(json.dumps(list(dataset), sort_keys=True).encode()).hexdigest(),
        "trainer_log_history": trainer.state.log_history,
        "adapter_manifest": adapter_manifest(args.output_dir),
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "effective_global_steps": trainer.state.global_step,
        "effective_training_args": training_args.to_dict(),
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
