#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RESULTS = (
    ROOT / "artifacts/extension/"
    "capability_final_results.json"
)

OUT_DIR = ROOT / "artifacts/extension"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PNG = OUT_DIR / "capability_extension_final_figure.png"
FIG_PDF = OUT_DIR / "capability_extension_final_figure.pdf"
TABLE_MD = OUT_DIR / "capability_extension_final_table.md"

d = json.loads(RESULTS.read_text())


# ------------------------------------------------------------
# Final results table
# ------------------------------------------------------------

causal = d["causal"]
probe = d["probe"]
quant = {
    x["precision"]: x
    for x in d["quantization"]
}
geom = d["cross_model_geometry"]

lines = []

lines.append("# Capability Refusal Extension — Final Results")
lines.append("")

lines.append("## Primary Llama results")
lines.append("")
lines.append(
    "| Measure | Result |"
)
lines.append(
    "|---|---:|"
)
lines.append(
    f'| Peak capability separability '
    f'({d["capability_direction"]["top_layer"]}) | '
    f'{d["capability_direction"]["top_separability"]:.3f} |'
)
lines.append(
    f'| Held-out probe accuracy '
    f'({probe["selected_layer"]}) | '
    f'{probe["selected_layer_eval_accuracy"]:.3f} |'
)
lines.append(
    f'| Capability ablation: net capability-refusal drop | '
    f'{causal["capability_ablation"]["net_drop"]:.3f} |'
)
lines.append(
    f'| Capability ablation: exact paired p-value | '
    f'{causal["capability_ablation"]["exact_mcnemar_p"]:.4f} |'
)
lines.append(
    f'| Capability ablation: bootstrap 95% CI | '
    f'[{causal["capability_ablation"]["bootstrap_95_ci"][0]:.3f}, '
    f'{causal["capability_ablation"]["bootstrap_95_ci"][1]:.3f}] |'
)
lines.append(
    f'| Safety ablation: net capability-refusal drop | '
    f'{causal["safety_ablation"]["net_drop"]:.3f} |'
)
lines.append(
    f'| Safety ablation: exact paired p-value | '
    f'{causal["safety_ablation"]["exact_mcnemar_p"]:.4f} |'
)
lines.append(
    f'| Safety ablation: bootstrap 95% CI | '
    f'[{causal["safety_ablation"]["bootstrap_95_ci"][0]:.3f}, '
    f'{causal["safety_ablation"]["bootstrap_95_ci"][1]:.3f}] |'
)

lines.append("")
lines.append("## Cross-model geometry")
lines.append("")
lines.append(
    "| Model | Capability peak | Cap. sep. | "
    "Mean S-E | Mean S-C | Mean E-C |"
)
lines.append(
    "|---|---:|---:|---:|---:|---:|"
)

for r in geom:
    name = r["model_id"]
    name = name.replace(
        "meta-llama/Llama-3.1-8B-Instruct",
        "Llama-3.1-8B",
    )
    name = name.replace(
        "Qwen/Qwen3-8B",
        "Qwen3-8B",
    )
    name = name.replace(
        "unsloth/gemma-2-9b-it-bnb-4bit",
        "Gemma-2-9B",
    )

    layer = r["top_capability_layer"].split(".")[-1]

    lines.append(
        f'| {name} | L{layer} | '
        f'{r["top_capability_separability"]:.3f} | '
        f'{r["mean_safety_epistemic_cosine"]:.3f} | '
        f'{r["mean_safety_capability_cosine"]:.3f} | '
        f'{r["mean_epistemic_capability_cosine"]:.3f} |'
    )

lines.append("")
lines.append("## Quantization robustness")
lines.append("")
lines.append(
    "| Precision | Direction cosine vs FP16 | "
    "Capability separability |"
)
lines.append(
    "|---|---:|---:|"
)

for name in ["fp16", "bnb_nf4", "bnb_int8"]:
    r = quant[name]
    lines.append(
        f'| {name} | '
        f'{r["capability_cosine_vs_fp16"]:.3f} | '
        f'{r["capability_separability"]:.3f} |'
    )

TABLE_MD.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------
# Final multi-panel figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4.8),
)

# ---- Panel A: cross-model geometry ----

ax = axes[0]

model_names = []
se = []
sc = []
ec = []

for r in geom:
    name = r["model_id"]

    if "Llama" in name:
        name = "Llama"
    elif "Qwen" in name:
        name = "Qwen"
    elif "gemma" in name.lower():
        name = "Gemma"

    model_names.append(name)
    se.append(r["mean_safety_epistemic_cosine"])
    sc.append(r["mean_safety_capability_cosine"])
    ec.append(r["mean_epistemic_capability_cosine"])

x = np.arange(len(model_names))
width = 0.24

ax.bar(x - width, se, width, label="Safety–Epistemic")
ax.bar(x, sc, width, label="Safety–Capability")
ax.bar(x + width, ec, width, label="Epistemic–Capability")

ax.axhline(0, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel("Mean cosine similarity")
ax.set_title("A. Cross-model geometry")
ax.legend(fontsize=8)


# ---- Panel B: causal effects ----

ax = axes[1]

conditions = [
    "Safety\ndirection",
    "Capability\ndirection",
]

drops = np.array([
    causal["safety_ablation"]["net_drop"],
    causal["capability_ablation"]["net_drop"],
])

cis = np.array([
    causal["safety_ablation"]["bootstrap_95_ci"],
    causal["capability_ablation"]["bootstrap_95_ci"],
])

lower = drops - cis[:, 0]
upper = cis[:, 1] - drops

ax.errorbar(
    np.arange(2),
    drops,
    yerr=np.vstack([lower, upper]),
    fmt="o",
    capsize=5,
)

ax.axhline(0, linewidth=0.8)
ax.set_xticks(np.arange(2))
ax.set_xticklabels(conditions)
ax.set_ylabel("Drop in capability-refusal rate")
ax.set_title("B. Paired causal effects")

pvals = [
    causal["safety_ablation"]["exact_mcnemar_p"],
    causal["capability_ablation"]["exact_mcnemar_p"],
]

for i, (drop, p) in enumerate(zip(drops, pvals)):
    ax.text(
        i,
        cis[i, 1] + 0.012,
        f"p={p:.4f}",
        ha="center",
        fontsize=9,
    )


# ---- Panel C: robustness / validation ----

ax = axes[2]

labels = [
    "Capability\nseparability",
    "Held-out\nprobe",
    "NF4\ncosine",
    "INT8\ncosine",
]

values = [
    d["capability_direction"]["top_separability"],
    probe["selected_layer_eval_accuracy"],
    quant["bnb_nf4"]["capability_cosine_vs_fp16"],
    quant["bnb_int8"]["capability_cosine_vs_fp16"],
]

bars = ax.bar(
    np.arange(len(values)),
    values,
)

ax.set_xticks(np.arange(len(values)))
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("C. Validation and robustness")

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.025,
        f"{value:.3f}",
        ha="center",
        fontsize=9,
    )


fig.suptitle(
    "Capability refusal is distinct but partially aligned with safety refusal",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

fig.savefig(
    FIG_PNG,
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    FIG_PDF,
    bbox_inches="tight",
)

print("Saved:", TABLE_MD)
print("Saved:", FIG_PNG)
print("Saved:", FIG_PDF)
