#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("artifacts/safety_finetuning")


def load(name):
    return json.loads((ROOT / name).read_text())


behavior = load("behavior_B_vs_C_stats.json")
geometry = load("geometry_B_vs_C.json")
cap_human = load("capability_behavior_human_stats.json")
cross = load("cross_checkpoint_ablation_stats.json")
cap_cross = load("capability_cross_checkpoint_ablation_stats.json")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

# ------------------------------------------------------------------
# A. Behavioral change
# ------------------------------------------------------------------
ax = axes[0]

labels = [
    "Safety\nrefusal",
    "Benign\nrefusal",
    "Epistemic\nabstention",
    "Capability\nrefusal",
]

B = [
    behavior["harmful_refusal"]["before_rate"],
    behavior["benign_refusal"]["before_rate"],
    behavior["unanswerable_abstention"]["before_rate"],
    cap_human["B_rate"],
]

C = [
    behavior["harmful_refusal"]["after_rate"],
    behavior["benign_refusal"]["after_rate"],
    behavior["unanswerable_abstention"]["after_rate"],
    cap_human["C_rate"],
]

x = np.arange(len(labels))
w = 0.36

ax.bar(x - w / 2, B, w, label="B: Instruct")
ax.bar(x + w / 2, C, w, label="C: Safety LoRA")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Behavior rate")
ax.set_title("A. Behavioral effects")
ax.legend(frameon=False, fontsize=8)

for i, (b, c) in enumerate(zip(B, C)):
    ax.text(
        i,
        max(b, c) + 0.035,
        f"{c-b:+.3f}",
        ha="center",
        fontsize=8,
    )

# ------------------------------------------------------------------
# B. Representational rotation
# ------------------------------------------------------------------
ax = axes[1]

geom_labels = ["Safety", "Epistemic", "Capability"]
cosines = [
    geometry["summary"]["mean_cos_S_B_C"],
    geometry["summary"]["mean_cos_E_B_C"],
    load("capability_geometry_B_vs_C.json")["summary"]["cap_B_C"],
]

x = np.arange(len(geom_labels))
bars = ax.bar(x, cosines)

ax.set_xticks(x)
ax.set_xticklabels(geom_labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Cosine similarity, B → C")
ax.set_title("B. Representational stability")
ax.axhline(1.0, linewidth=0.8, linestyle="--")

for bar, value in zip(bars, cosines):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.025,
        f"{value:.3f}",
        ha="center",
        fontsize=9,
    )

ax.text(
    0.5,
    0.05,
    "Lower cosine = greater rotation",
    transform=ax.transAxes,
    ha="center",
    fontsize=8,
)

# ------------------------------------------------------------------
# C. Cross-checkpoint causal transfer
# ------------------------------------------------------------------
ax = axes[2]

def drop(stats, checkpoint, condition, metric):
    return -stats[checkpoint][condition][metric]["change"]


causal_labels = [
    "B←B\nSafety",
    "B←C\nSafety",
    "C←B\nSafety",
    "C←C\nSafety",
    "B←B\nCapability",
    "B←C\nCapability",
    "C←B\nCapability",
    "C←C\nCapability",
]

drops = [
    drop(cross, "B_instruct", "ablate_B_safety", "safety_refusal"),
    drop(cross, "B_instruct", "ablate_C_safety", "safety_refusal"),
    drop(cross, "C_safety_lora", "ablate_B_safety", "safety_refusal"),
    drop(cross, "C_safety_lora", "ablate_C_safety", "safety_refusal"),

    -cap_cross["B_instruct"]["ablate_B_capability"]["change"],
    -cap_cross["B_instruct"]["ablate_C_capability"]["change"],
    -cap_cross["C_safety_lora"]["ablate_B_capability"]["change"],
    -cap_cross["C_safety_lora"]["ablate_C_capability"]["change"],
]

x = np.arange(len(causal_labels))
bars = ax.bar(x, drops)

ax.set_xticks(x)
ax.set_xticklabels(causal_labels, fontsize=7)
ax.set_ylim(-0.08, 1.08)
ax.set_ylabel("Behavior drop under ablation")
ax.set_title("C. Causal transfer")
ax.axhline(0, linewidth=0.8)
ax.axvline(3.5, linewidth=0.8, linestyle="--")

for bar, value in zip(bars, drops):
    y = value + 0.025 if value >= 0 else value - 0.045
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y,
        f"{value:.3f}",
        ha="center",
        fontsize=7,
    )

fig.suptitle(
    "Safety fine-tuning reorganizes refusal-related representations",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

png = ROOT / "safety_finetuning_final_figure.png"
pdf = ROOT / "safety_finetuning_final_figure.pdf"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")

print("Saved:", png)
print("Saved:", pdf)
