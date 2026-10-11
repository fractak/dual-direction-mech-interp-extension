#!/usr/bin/env python3

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

cross = json.load(open(
    ROOT / "artifacts/cross_model_capability/combined_results.json"
))
probe = json.load(open(
    ROOT / "artifacts/probes/capability_probe.json"
))
stats = json.load(open(
    ROOT / "artifacts/extension/capability_paired_stats.json"
))
quant = json.load(open(
    ROOT / "artifacts/quantization/capability_sweep.json"
))
direction = json.load(open(
    ROOT / "artifacts/directions/capability_train_160.json"
))

out_path = (
    ROOT / "artifacts/extension/"
    "capability_final_results.json"
)

# Primary capability direction.
cap_layer = direction["ranked_layers"][0]

cross_models = []

for r in cross["results"]:
    cross_models.append({
        "model_id": r["model_id"],
        "top_capability_layer": r["top_capability_layer"],
        "top_capability_separability": r["top_capability_sep"],
        "mean_safety_epistemic_cosine":
            r["mean_safety_epistemic_cosine"],
        "mean_safety_capability_cosine":
            r["mean_safety_capability_cosine"],
        "mean_epistemic_capability_cosine":
            r["mean_epistemic_capability_cosine"],
        "safety_capability_at_top_capability":
            r["safety_capability_at_top_capability"],
        "epistemic_capability_at_top_capability":
            r["epistemic_capability_at_top_capability"],
    })

results = {
    "artifact_type": "capability_extension_final_results",

    "experiment": {
        "primary_model":
            "meta-llama/Llama-3.1-8B-Instruct",
        "seed": 42,
        "capability_train_prompts": 160,
        "capability_train_families": 16,
        "capability_eval_prompts": 60,
        "capability_eval_families": 60,
        "capability_layer": 24,
        "safety_layer": 22,
        "epistemic_layer": 9,
        "top_k": 1,
        "ablation_strength": 1.5,
        "manual_evaluation":
            "180 responses blindly annotated using fixed rubric",
    },

    "capability_direction": {
        "top_layer": cap_layer["name"],
        "top_separability": cap_layer["score"],
    },

    "probe": {
        "selected_layer":
            probe["selected_layer"],
        "selected_layer_train_accuracy":
            probe["summary"]["selected_layer_train_accuracy"],
        "selected_layer_eval_accuracy":
            probe["summary"]["selected_layer_eval_accuracy"],
        "best_layer":
            probe["summary"]["best_layer"],
        "best_eval_accuracy":
            probe["summary"]["best_eval_accuracy"],
    },

    "causal": stats,

    "quantization": quant["drift_table"],

    "cross_model_geometry": cross_models,
}

out_path.write_text(json.dumps(results, indent=2))

print("Saved:", out_path)

print("\n=== FINAL EXTENSION SUMMARY ===")
print(
    "Capability separability:",
    f'{results["capability_direction"]["top_separability"]:.3f}',
)

print(
    "Held-out probe @ layer 24:",
    f'{results["probe"]["selected_layer_eval_accuracy"]:.3f}',
)

print(
    "Safety ablation capability drop:",
    f'{stats["safety_ablation"]["net_drop"]:.3f}',
    "p=",
    f'{stats["safety_ablation"]["exact_mcnemar_p"]:.4f}',
)

print(
    "Capability ablation capability drop:",
    f'{stats["capability_ablation"]["net_drop"]:.3f}',
    "p=",
    f'{stats["capability_ablation"]["exact_mcnemar_p"]:.4f}',
)

print("\nCross-model S-C vs E-C:")
for r in cross_models:
    print(
        r["model_id"],
        f'S-C={r["mean_safety_capability_cosine"]:.3f}',
        f'E-C={r["mean_epistemic_capability_cosine"]:.3f}',
    )

print("\nQuantization:")
for r in quant["drift_table"]:
    print(
        r["precision"],
        f'cos={r["capability_cosine_vs_fp16"]:.3f}',
        f'sep={r["capability_separability"]:.3f}',
    )
