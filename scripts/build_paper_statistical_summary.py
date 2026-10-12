#!/usr/bin/env python3

import json
from pathlib import Path

ROOT = Path("artifacts/safety_finetuning")

BEHAVIOR = json.loads(
    (ROOT / "behavior_B_vs_C_stats.json").read_text()
)

SAFETY_CAUSAL = json.loads(
    (ROOT / "cross_checkpoint_ablation_stats.json").read_text()
)

CAP_CAUSAL = json.loads(
    (
        ROOT
        / "capability_cross_checkpoint_ablation_independent_blind_stats.json"
    ).read_text()
)


def holm_adjust(rows):
    ordered = sorted(
        enumerate(rows),
        key=lambda x: x[1]["p_raw"],
    )

    m = len(rows)
    adjusted = [None] * m
    running = 0.0

    for rank, (original_i, row) in enumerate(ordered):
        value = min(
            1.0,
            row["p_raw"] * (m - rank),
        )
        running = max(running, value)
        adjusted[original_i] = running

    for row, adj in zip(rows, adjusted):
        row["p_holm"] = adj

    return rows


# ------------------------------------------------------------
# Pull the four primary comparisons directly from frozen stats.
# ------------------------------------------------------------

harm = BEHAVIOR["harmful_refusal"]
epi = BEHAVIOR["unanswerable_abstention"]

safety_causal = (
    SAFETY_CAUSAL["C_safety_lora"]
    ["ablate_C_safety"]
    ["safety_refusal"]
)

cap_causal = (
    CAP_CAUSAL["results"]
    ["C_safety_lora"]
    ["ablate_C_capability"]
)

primary = [
    {
        "id": "P1",
        "comparison": "B→C harmful-request refusal",
        "source": "detector",
        "n": harm["n"],
        "before": harm["before_rate"],
        "after": harm["after_rate"],
        "effect": harm["change"],
        "ci95": harm["bootstrap_95ci_change"],
        "p_raw": harm["exact_mcnemar_p"],
    },
    {
        "id": "P2",
        "comparison": "B→C unanswerable abstention",
        "source": "detector",
        "n": epi["n"],
        "before": epi["before_rate"],
        "after": epi["after_rate"],
        "effect": epi["change"],
        "ci95": epi["bootstrap_95ci_change"],
        "p_raw": epi["exact_mcnemar_p"],
    },
    {
        "id": "P3",
        "comparison": (
            "C safety refusal under C-safety-direction ablation"
        ),
        "source": "detector",
        "n": 200,
        "before": safety_causal["before"],
        "after": safety_causal["after"],
        "effect": safety_causal["change"],
        "ci95": safety_causal["bootstrap_ci_95"],
        "p_raw": safety_causal["p_exact"],
    },
    {
        "id": "P4",
        "comparison": (
            "C capability refusal under C-capability-direction ablation"
        ),
        "source": "independent blind annotation",
        "n": cap_causal["n"],
        "before": cap_causal["baseline_rate"],
        "after": cap_causal["ablation_rate"],
        "effect": cap_causal["change"],
        "ci95": cap_causal["bootstrap_95_ci_change"],
        "p_raw": cap_causal["paired_exact_p"],
    },
]

primary = holm_adjust(primary)

# ------------------------------------------------------------
# Secondary behavioral effects.
# These are reported with raw paired tests, not placed in the
# primary Holm family.
# ------------------------------------------------------------

secondary = []

for key, label in [
    ("benign_refusal", "B→C benign refusal"),
    (
        "answerable_abstention",
        "B→C answerable-question abstention",
    ),
]:
    x = BEHAVIOR[key]

    secondary.append({
        "comparison": label,
        "source": "detector",
        "n": x["n"],
        "before": x["before_rate"],
        "after": x["after_rate"],
        "effect": x["change"],
        "ci95": x["bootstrap_95ci_change"],
        "p_raw": x["exact_mcnemar_p"],
    })

# Independent-blind capability baseline behavior.
cap_behavior = json.loads(
    (ROOT / "capability_behavior_human_stats.json").read_text()
)

secondary.append({
    "comparison": "B→C capability refusal",
    "source": "independent blind annotation",
    "n": cap_behavior["n"],
    "before": cap_behavior["B_rate"],
    "after": cap_behavior["C_rate"],
    "effect": cap_behavior["change"],
    "ci95": cap_behavior["bootstrap_95_ci_change"],
    "p_raw": cap_behavior["paired_exact_p"],
})

out = {
    "multiple_comparison_policy": {
        "primary_family": [
            x["id"] for x in primary
        ],
        "correction": "Holm family-wise error correction",
        "alpha": 0.05,
        "note": (
            "Secondary and descriptive analyses are reported "
            "separately and are not included in the primary "
            "confirmatory correction family."
        ),
    },
    "primary": primary,
    "secondary_behavior": secondary,
    "geometry_policy": (
        "Representational cosine similarities and separability "
        "are treated as descriptive effect-size analyses; "
        "prompt-resampling stability is used as the robustness control."
    ),
    "annotation_terminology": {
        "capability": "independent blind annotation",
        "safety_epistemic": "detector-based evaluation",
    },
}

out_path = ROOT / "paper_statistical_summary.json"
out_path.write_text(json.dumps(out, indent=2) + "\n")

print(json.dumps(out, indent=2))
print("\nSaved:", out_path)
