#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from ddmi.data.loaders import (
    load_epistemic_prompts,
    load_safety_prompts,
    prompts_to_text_list,
)
from ddmi.editing.apply_edit import ActivationDirectionAblator, EditSpec
from ddmi.evaluation.detectors import (
    abstention_rate,
    is_abstention,
    is_refusal,
    refusal_rate,
)
from ddmi.models.generation import TextGenerationConfig, generate_batch
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer


ROOT = Path("artifacts/safety_finetuning")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-per-behavior", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strength", type=float, default=1.5)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--adapter",
        default=str(ROOT / "llama31_8b_instruct_safety_lora"),
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "cross_checkpoint_ablation.json"),
    )
    return p.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text())


def top_direction(path):
    d = read_json(path)
    layer = d["ranked_layers"][0]["name"]
    vector = d["directions"][layer]["direction"]
    return layer, vector


def load_checkpoint(adapter=None):
    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            load_in_4bit=True,
            torch_dtype="float16",
            attn_implementation="eager",
        )
    )

    if adapter:
        peft = PeftModel.from_pretrained(model, adapter)
        model = peft.base_model.model
        model.eval()

    return model, tokenizer


def evaluate(model, tokenizer, harmful, unanswerable, config):
    harmful_responses = generate_batch(
        model, tokenizer, harmful, config=config
    )
    epistemic_responses = generate_batch(
        model, tokenizer, unanswerable, config=config
    )

    return {
        "rates": {
            "safety_refusal": refusal_rate(harmful_responses),
            "epistemic_abstention": abstention_rate(epistemic_responses),
        },
        "labels": {
            "safety_refusal": [
                is_refusal(x) for x in harmful_responses
            ],
            "epistemic_abstention": [
                is_abstention(x) for x in epistemic_responses
            ],
        },
        "responses": {
            "harmful": harmful_responses,
            "unanswerable": epistemic_responses,
        },
    }


def evaluate_ablation(
    model,
    tokenizer,
    harmful,
    unanswerable,
    config,
    layer,
    direction,
    strength,
):
    ablator = ActivationDirectionAblator(
        module_names=[layer],
        direction=direction,
        spec=EditSpec(strength=strength),
    ).attach(model)

    try:
        return evaluate(
            model,
            tokenizer,
            harmful,
            unanswerable,
            config,
        )
    finally:
        ablator.close()


def release(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    safety = load_safety_prompts(
        limit=2 * args.eval_per_behavior,
        seed=args.seed,
    )
    epistemic = load_epistemic_prompts(
        limit=2 * args.eval_per_behavior,
        seed=args.seed,
    )

    harmful = prompts_to_text_list(safety["harmful"])
    unanswerable = prompts_to_text_list(epistemic["unanswerable"])

    directions = {
        "B_safety": top_direction(ROOT / "B_safety.json"),
        "C_safety": top_direction(ROOT / "C_safety.json"),
        "B_epistemic": top_direction(ROOT / "B_epistemic.json"),
        "C_epistemic": top_direction(ROOT / "C_epistemic.json"),
    }

    print("DIRECTIONS")
    for name, (layer, _) in directions.items():
        print(f"  {name:14} {layer}")

    gen_config = TextGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    results = {}

    for checkpoint, adapter in [
        ("B_instruct", None),
        ("C_safety_lora", args.adapter),
    ]:
        print(f"\nLoading {checkpoint}...")
        model, tokenizer = load_checkpoint(adapter)

        checkpoint_results = {}

        print("  baseline")
        checkpoint_results["baseline"] = evaluate(
            model,
            tokenizer,
            harmful,
            unanswerable,
            gen_config,
        )

        for direction_name, (layer, direction) in directions.items():
            print(f"  ablate {direction_name} @ {layer}")
            checkpoint_results[f"ablate_{direction_name}"] = (
                evaluate_ablation(
                    model,
                    tokenizer,
                    harmful,
                    unanswerable,
                    gen_config,
                    layer,
                    direction,
                    args.strength,
                )
            )

        results[checkpoint] = checkpoint_results
        release(model)
        del tokenizer

    output = {
        "config": {
            "eval_per_behavior": args.eval_per_behavior,
            "seed": args.seed,
            "strength": args.strength,
            "max_new_tokens": args.max_new_tokens,
            "directions": {
                k: v[0] for k, v in directions.items()
            },
        },
        "results": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")

    print("\nSUMMARY")
    for checkpoint, conditions in results.items():
        print("\n" + checkpoint)
        for condition, row in conditions.items():
            rates = row["rates"]
            print(
                f"  {condition:24} "
                f"S={rates['safety_refusal']:.3f} "
                f"E={rates['epistemic_abstention']:.3f}"
            )

    print("\nSaved:", out)


if __name__ == "__main__":
    main()
