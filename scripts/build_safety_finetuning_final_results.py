#!/usr/bin/env python3

import json
from pathlib import Path

ROOT = Path("artifacts/safety_finetuning")


def load(name):
    return json.loads((ROOT / name).read_text())


behavior = load("behavior_B_vs_C_stats.json")
geometry = load("geometry_B_vs_C.json")
cross = load("cross_checkpoint_ablation_stats.json")
cap_geom = load("capability_geometry_B_vs_C.json")
cap_human = load("capability_behavior_human_stats.json")
cap_causal = load("capability_cross_checkpoint_ablation_stats.json")

result = {
    "experiment": (
        "Safety fine-tuning: representational and causal reorganization "
        "of safety, epistemic, and capability refusal"
    ),
    "checkpoints": {
        "B": "meta-llama/Llama-3.1-8B-Instruct",
        "C": (
            "meta-llama/Llama-3.1-8B-Instruct + "
            "controlled safety QLoRA"
        ),
    },
    "behavior": {
        "safety_refusal": behavior["harmful_refusal"],
        "benign_refusal": behavior["benign_refusal"],
        "epistemic_abstention": behavior["unanswerable_abstention"],
        "answerable_abstention": behavior["answerable_abstention"],
        "capability_refusal_human": cap_human,
    },
    "geometry": {
        "safety_epistemic": geometry,
        "capability": cap_geom,
    },
    "causal_transfer": {
        "safety_epistemic": cross,
        "capability": cap_causal,
    },
}

out = ROOT / "safety_finetuning_final_results.json"
out.write_text(json.dumps(result, indent=2) + "\n")

print("Saved:", out)
