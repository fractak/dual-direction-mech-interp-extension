#!/usr/bin/env python3

import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from ddmi.editing.apply_edit import (
    ActivationDirectionAblator,
    EditSpec,
)
from ddmi.evaluation.detectors import (
    is_abstention,
    is_capability_refusal,
    is_refusal,
)
from ddmi.models.generation import (
    TextGenerationConfig,
    generate_batch,
)
from ddmi.models.loader import (
    ModelLoadConfig,
    load_model_and_tokenizer,
)

ROOT = Path("artifacts/safety_finetuning")
DATA = Path(
    "data/refusal_overlap/"
    "refusal_overlap_matrix.json"
)
OUT = Path(
    "artifacts/refusal_overlap/"
    "causal_ablation.json"
)

ADAPTER = ROOT / "llama31_8b_instruct_safety_lora"
B_CAP = Path(
    "artifacts/directions/"
    "capability_train_160.json"
)

STRENGTH = 1.5


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
        peft = PeftModel.from_pretrained(
            model,
            adapter,
        )
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

    rows = []

    for response in responses:
        rows.append({
            "response": response,
            "refusal": is_refusal(response),
            "capability_refusal":
                is_capability_refusal(response),
            "epistemic_abstention":
                is_abstention(response),
        })

    return rows


def evaluate_with_ablation(
    model,
    tokenizer,
    prompts,
    config,
    directions,
    names,
):
    ablators = []

    try:
        for name in names:
            layer, vector = directions[name]

            ablator = ActivationDirectionAblator(
                module_names=[layer],
                direction=vector,
                spec=EditSpec(
                    strength=STRENGTH
                ),
            ).attach(model)

            ablators.append(ablator)

        return evaluate(
            model,
            tokenizer,
            prompts,
            config,
        )

    finally:
        for ablator in reversed(ablators):
            ablator.close()


CONDITIONS = {
    "safety_only": [
        (),
        ("safety",),
    ],
    "capability_only": [
        (),
        ("capability",),
    ],
    "epistemic_only": [
        (),
        ("epistemic",),
    ],
    "safety_capability": [
        (),
        ("safety",),
        ("capability",),
        ("safety", "capability"),
    ],
    "safety_epistemic": [
        (),
        ("safety",),
        ("epistemic",),
        ("safety", "epistemic"),
    ],
    "capability_epistemic": [
        (),
        ("capability",),
        ("epistemic",),
        ("capability", "epistemic"),
    ],
    "triple_overlap": [
        (),
        ("safety",),
        ("capability",),
        ("epistemic",),
        ("safety", "capability"),
        ("safety", "epistemic"),
        ("capability", "epistemic"),
        ("safety", "capability", "epistemic"),
    ],
}


def condition_name(names):
    if not names:
        return "baseline"

    return "ablate_" + "_".join(names)


def release(model):
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    data = read_json(DATA)
    rows = data["rows"]

    prompts_by_type = {}

    for prompt_type in CONDITIONS:
        xs = [
            x for x in rows
            if x["prompt_type"] == prompt_type
        ]

        assert len(xs) == 20

        prompts_by_type[prompt_type] = [
            x["prompt"] for x in xs
        ]

    all_directions = {
        "B": {
            "safety":
                top_direction(ROOT / "B_safety.json"),
            "epistemic":
                top_direction(ROOT / "B_epistemic.json"),
            "capability":
                top_direction(B_CAP),
        },
        "C": {
            "safety":
                top_direction(ROOT / "C_safety.json"),
            "epistemic":
                top_direction(ROOT / "C_epistemic.json"),
            "capability":
                top_direction(ROOT / "C_capability.json"),
        },
    }

    for prefix, dirs in all_directions.items():
        print("\n" + prefix + " directions")

        for name, (layer, _) in dirs.items():
            print(f"  {name:12s} {layer}")

    config = TextGenerationConfig(
        max_new_tokens=128,
        temperature=0.0,
        do_sample=False,
    )

    results = {}

    for checkpoint, adapter, prefix in [
        ("B_instruct", None, "B"),
        ("C_safety_lora", str(ADAPTER), "C"),
    ]:
        print("\nLoading", checkpoint)

        model, tokenizer = load_checkpoint(adapter)
        directions = all_directions[prefix]

        checkpoint_results = {}

        for prompt_type, conditions in CONDITIONS.items():
            print("\n ", prompt_type)

            prompts = prompts_by_type[prompt_type]
            type_results = {}

            for names in conditions:
                cname = condition_name(names)
                print("   ", cname)

                if names:
                    out = evaluate_with_ablation(
                        model,
                        tokenizer,
                        prompts,
                        config,
                        directions,
                        names,
                    )
                else:
                    out = evaluate(
                        model,
                        tokenizer,
                        prompts,
                        config,
                    )

                type_results[cname] = out

            checkpoint_results[prompt_type] = (
                type_results
            )

        results[checkpoint] = checkpoint_results

        release(model)
        del tokenizer

    output = {
        "config": {
            "strength": STRENGTH,
            "n_per_category": 20,
            "max_new_tokens": 128,
            "direction_layers": {
                prefix: {
                    name: value[0]
                    for name, value
                    in dirs.items()
                }
                for prefix, dirs
                in all_directions.items()
            },
        },
        "results": results,
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        json.dumps(output, indent=2) + "\n"
    )

    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
