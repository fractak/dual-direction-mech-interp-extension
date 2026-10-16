#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("artifacts/refusal_overlap")

SRC = ROOT / "ablation_reviewed_stats_final.json"
PNG = ROOT / "refusal_overlap_rerouting.png"
PDF = ROOT / "refusal_overlap_rerouting.pdf"

d = json.load(open(SRC))

order = [
    (
        "B_safety_epistemic_refusal_safety_ablation",
        "B S+E\nrefusal",
    ),
    (
        "B_safety_epistemic_epi_reason_safety_ablation",
        "B S+E\nepistemic rationale",
    ),
    (
        "C_capability_epistemic_refusal_cap_ablation",
        "C Cap+E\nrefusal",
    ),
    (
        "C_triple_refusal_safety_cap_ablation",
        "C triple\nrefusal",
    ),
    (
        "C_triple_refusal_add_epistemic",
        "C triple\n+ epistemic",
    ),
]

labels = []
before = []
after = []

for key, label in order:
    row = d[key]

    labels.append(label)
    before.append(row["baseline_rate"])
    after.append(row["comparison_rate"])

x = np.arange(len(labels))
width = 0.36

fig, ax = plt.subplots(
    figsize=(9.0, 4.8)
)

ax.bar(
    x - width / 2,
    before,
    width,
    label="Before intervention",
)

ax.bar(
    x + width / 2,
    after,
    width,
    label="After intervention",
)

ax.set_ylim(0, 1.08)
ax.set_ylabel("Annotated response rate")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(frameon=False)

for i, (key, _) in enumerate(order):
    row = d[key]

    delta = row["delta"]
    holm = row["holm_p"]

    if holm < 0.001:
        sig = "***"
    elif holm < 0.01:
        sig = "**"
    elif holm < 0.05:
        sig = "*"
    else:
        sig = "n.s."

    y = max(
        row["baseline_rate"],
        row["comparison_rate"],
    ) + 0.035

    ax.text(
        i,
        y,
        f"{delta:+.2f} {sig}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)

print("Saved:", PNG)
print("Saved:", PDF)
