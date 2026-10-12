#!/usr/bin/env python3

import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from ddmi.data.loaders import load_capability_prompts, prompts_to_text_list
from ddmi.editing.apply_edit import ActivationDirectionAblator, EditSpec
from ddmi.evaluation.detectors import is_capability_refusal
from ddmi.models.generation import TextGenerationConfig, generate_batch
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer


ROOT = Path("artifacts/safety_finetuning")
B_CAP = Path("artifacts/directions/capability_train_160.json")
C_CAP = ROOT / "C_capability.json"
ADAPTER = ROOT / "llama31_8b_instruct_safety_lora"
OUT = ROOT / "capability_cross_checkpoint_ablation.json"

STRENGTH = 1.5


def read_json(path):
    return json.loads(Path(path).read_text())


def top_direction(path):
    d = read_json(path)
    layer = d["ranked_layers"][0]["name"]
    direction = d["directions"][layer]["direction"]
    return layer, direction


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


def evaluate(model, tokenizer, prompts, config):
    responses = generate_batch(
        model,
        tokenizer,
        prompts,
        config=config,
    )
    labels = [is_capability_refusal(x) for x in responses]

    return {
        "rate": sum(labels) / len(labels),
        "labels": labels,
        "responses": responses,
    }


def ablate_and_evaluate(
    model,
    tokenizer,
    prompts,
    config,
    layer,
    direction,
):
    ablator = ActivationDirectionAblator(
        module_names=[layer],
        direction=direction,
        spec=EditSpec(strength=STRENGTH),
    ).attach(model)

    try:
        return evaluate(model, tokenizer, prompts, config)
    finally:
        ablator.close()


def release(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    data = load_capability_prompts(
        limit=120,
        seed=42,
        split="expanded_eval",
    )

    samples = data["incapable"]
    prompts = prompts_to_text_list(samples)

    assert len(prompts) == 60

    b_layer, b_direction = top_direction(B_CAP)
    c_layer, c_direction = top_direction(C_CAP)

    print("B capability:", b_layer)
    print("C capability:", c_layer)
    print("Prompts:", len(prompts))

    config = TextGenerationConfig(
        max_new_tokens=128,
        temperature=0.0,
        do_sample=False,
    )

    results = {}

    for checkpoint, adapter in [
        ("B_instruct", None),
        ("C_safety_lora", str(ADAPTER)),
    ]:
        print(f"\nLoading {checkpoint}...")
        model, tokenizer = load_checkpoint(adapter)

        row = {}

        print("  baseline")
        row["baseline"] = evaluate(
            model,
            tokenizer,
            prompts,
            config,
        )

        print(f"  ablate B capability @ {b_layer}")
        row["ablate_B_capability"] = ablate_and_evaluate(
            model,
            tokenizer,
            prompts,
            config,
            b_layer,
            b_direction,
        )

        print(f"  ablate C capability @ {c_layer}")
        row["ablate_C_capability"] = ablate_and_evaluate(
            model,
            tokenizer,
            prompts,
            config,
            c_layer,
            c_direction,
        )

        results[checkpoint] = row

        release(model)
        del tokenizer

    output = {
        "config": {
            "n": len(prompts),
            "split": "expanded_eval",
            "seed": 42,
            "strength": STRENGTH,
            "B_layer": b_layer,
            "C_layer": c_layer,
        },
        "results": results,
    }

    OUT.write_text(json.dumps(output, indent=2) + "\n")

    print("\nSUMMARY")

    for checkpoint, conditions in results.items():
        print("\n" + checkpoint)

        for condition, row in conditions.items():
            print(
                f"  {condition:24} "
                f"capability_refusal={row['rate']:.3f}"
            )

    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
