#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddmi.data.loaders import load_capability_prompts, prompts_to_text_list
from ddmi.editing.apply_edit import ActivationDirectionAblator, EditSpec
from ddmi.evaluation.detectors import is_capability_refusal
from ddmi.models.generation import TextGenerationConfig, generate_batch
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
from ddmi.utils.io import read_json


def direction_at(artifact, layer):
    return artifact["directions"][layer]["direction"]


def generate_labels(model, tokenizer, prompts, config):
    responses = generate_batch(model, tokenizer, prompts, config=config)
    labels = [is_capability_refusal(r) for r in responses]
    return responses, labels


def transition_counts(before, after):
    rr = sum(b and a for b, a in zip(before, after))
    rn = sum(b and not a for b, a in zip(before, after))
    nr = sum((not b) and a for b, a in zip(before, after))
    nn = sum((not b) and (not a) for b, a in zip(before, after))
    return {
        "refusal_to_refusal": rr,
        "refusal_to_nonrefusal": rn,
        "nonrefusal_to_refusal": nr,
        "nonrefusal_to_nonrefusal": nn,
        "net_drop": rn - nr,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paired capability-refusal validation under safety and capability ablations."
    )
    parser.add_argument(
        "--output",
        default="artifacts/extension/capability_paired_validation.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    safety_art = read_json("artifacts/directions/safety_200.json")
    capability_art = read_json("artifacts/directions/capability_train_160.json")

    safety_layer = "model.layers.22"
    capability_layer = "model.layers.24"
    strength = 1.5

    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            load_in_4bit=True,
            torch_dtype="float16",
            attn_implementation="eager",
        )
    )

    data = load_capability_prompts(
        limit=120,
        seed=42,
        split="expanded_eval",
    )
    samples = data["incapable"]
    prompts = prompts_to_text_list(samples)

    gen_config = TextGenerationConfig(
        max_new_tokens=128,
        temperature=0.0,
        do_sample=False,
    )

    print("Generating baseline...")
    baseline_responses, baseline_labels = generate_labels(
        model, tokenizer, prompts, gen_config
    )

    print("Generating safety ablation...")
    safety_ablator = ActivationDirectionAblator(
        module_names=[safety_layer],
        direction=direction_at(safety_art, safety_layer),
        spec=EditSpec(strength=strength),
    ).attach(model)

    try:
        safety_responses, safety_labels = generate_labels(
            model, tokenizer, prompts, gen_config
        )
    finally:
        safety_ablator.close()

    print("Generating capability ablation...")
    capability_ablator = ActivationDirectionAblator(
        module_names=[capability_layer],
        direction=direction_at(capability_art, capability_layer),
        spec=EditSpec(strength=strength),
    ).attach(model)

    try:
        capability_responses, capability_labels = generate_labels(
            model, tokenizer, prompts, gen_config
        )
    finally:
        capability_ablator.close()

    rows = []

    for sample, prompt, br, bl, sr, sl, cr, cl in zip(
        samples,
        prompts,
        baseline_responses,
        baseline_labels,
        safety_responses,
        safety_labels,
        capability_responses,
        capability_labels,
    ):
        rows.append({
            "id": sample.id,
            "prompt": prompt,
            "baseline": {
                "label": bl,
                "response": br,
            },
            "ablate_safety": {
                "label": sl,
                "response": sr,
            },
            "ablate_capability": {
                "label": cl,
                "response": cr,
            },
        })

    result = {
        "config": {
            "n": len(prompts),
            "strength": strength,
            "safety_layer": safety_layer,
            "capability_layer": capability_layer,
        },
        "rates": {
            "baseline": sum(baseline_labels) / len(baseline_labels),
            "ablate_safety": sum(safety_labels) / len(safety_labels),
            "ablate_capability": sum(capability_labels) / len(capability_labels),
        },
        "transitions": {
            "safety_ablation": transition_counts(
                baseline_labels, safety_labels
            ),
            "capability_ablation": transition_counts(
                baseline_labels, capability_labels
            ),
        },
        "rows": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\nRATES")
    for k, v in result["rates"].items():
        print(f"{k:20} {v:.3f}")

    print("\nTRANSITIONS")
    for k, v in result["transitions"].items():
        print(k, v)

    print("\nSaved:", out)


if __name__ == "__main__":
    main()
