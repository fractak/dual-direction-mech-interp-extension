"""Publication figures from recomputed results; no hard-coded outcome counts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddmi.checkpoints import CHECKPOINTS
from ddmi.paper.evidence import OUTCOMES

COLORS = ["#3274A1", "#E6A23C", "#C75450", "#92999F"]
LABELS = ["Acknowledgment", "Instructions", "Unsupported", "Other"]
CHECKPOINT_COLORS = ["#3274A1", "#E58B36"]


def plot_figures(result, output_dir, numbers=(1, 2, 3, 4), dpi=400):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    functions = {1: fixed_intervention, 2: controls_and_strength, 3: overlap, 4: geometry}
    files = []
    for number in dict.fromkeys(numbers):
        fig, name = functions[number](result)
        try:
            for extension in ("png", "pdf"):
                path = output_dir / f"figure{number}_{name}.{extension}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                files.append(path)
        finally:
            plt.close(fig)
    return files


def fixed_intervention(result):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    conditions = result["capability"]["conditions"]
    x = np.arange(2)
    for ax, outcome, title in zip(
        axes,
        ("acknowledgment", "unsupported"),
        ("Acknowledgment of access limits", "Unsupported assertions"),
    ):
        for i, (checkpoint, metadata) in enumerate(CHECKPOINTS.items()):
            counts = [
                conditions[checkpoint][condition]["counts"][outcome]
                for condition in ("baseline", "capability_from_safety_lora")
            ]
            bars = ax.bar(
                x + (i - 0.5) * 0.34,
                np.array(counts) / 60 * 100,
                width=0.34,
                color=CHECKPOINT_COLORS[i],
                label=metadata["label"],
            )
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{count}/60",
                    ha="center",
                    fontsize=9,
                )
        ax.set(
            title=title,
            xticks=x,
            xticklabels=["No edit", "Same activation edit"],
            ylim=(0, 112),
            yticks=[0, 25, 50, 75, 100],
            ylabel="Responses (%)",
        )
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Llama 3.1 8B Instruct: before and after the additional safety LoRA", fontsize=13)
    return fig, "fixed_intervention"


def controls_and_strength(result):
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.35, 1]}, layout="constrained"
    )
    conditions = result["capability"]["conditions"]["safety_lora"]
    random_counts = [conditions[f"random_seed{seed}"]["counts"] for seed in range(2026, 2032)]
    if any(counts != random_counts[0] for counts in random_counts):
        raise ValueError(
            "Random controls differ; a single collapsed figure row would be misleading"
        )
    names = [
        "baseline",
        "random_seed2026",
        "epistemic",
        "safety",
        "safety_orthogonal",
        "capability_from_safety_lora",
    ]
    labels = [
        "No edit",
        "Random directions\n(each of 6)",
        "Epistemic direction",
        "Safety direction",
        "Safety orthogonal\nto capability",
        "Capability direction",
    ]
    values = np.array([[conditions[name]["counts"][key] for key in OUTCOMES] for name in names])
    ax = axes[0]
    left = np.zeros(len(names))
    for column, (label, color) in enumerate(zip(LABELS, COLORS)):
        ax.barh(
            np.arange(len(names)),
            values[:, column],
            left=left,
            color=color,
            label=label,
            height=0.7,
        )
        for row, count in enumerate(values[:, column]):
            if count >= 4:
                ax.text(
                    left[row] + count / 2,
                    row,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if column == 1 else "white",
                )
        left += values[:, column]
    ax.set(
        yticks=np.arange(len(names)),
        yticklabels=labels,
        xlim=(0, 60),
        xlabel="Responses (60 prompts)",
        title="Direction controls",
    )
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9)
    dose = ["baseline", "capability_low", "capability_medium", "capability_from_safety_lora"]
    strengths = [conditions[name]["strength"] for name in dose]
    for key, label, color in zip(OUTCOMES, LABELS, COLORS):
        axes[1].plot(
            strengths,
            [conditions[name]["counts"][key] for name in dose],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axes[1].set(
        xlim=(-0.04, 1.54),
        ylim=(0, 63),
        xlabel="Intervention strength",
        ylabel="Responses (60 prompts)",
        title="Capability-direction strength",
    )
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Llama 3.1 8B Instruct + safety LoRA: controls and strength sweep", fontsize=13)
    return fig, "controls_and_strength"


def overlap(result):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    joint = result["overlap"]["joint_outcomes_original"]
    before, after = joint["baseline"], joint["ablate_safety"]
    y = np.arange(4)
    ax = axes[0]
    for row, (start, end) in enumerate(zip(before, after)):
        ax.plot([start, end], [row, row], color="#B8BEC4", lw=3)
    ax.scatter(before, y, s=65, color=CHECKPOINT_COLORS[0], label="Before edit", zorder=3)
    ax.scatter(after, y, s=65, color=CHECKPOINT_COLORS[1], label="After safety edit", zorder=3)
    ax.set(
        yticks=y,
        yticklabels=[
            "Refusal + epistemic language",
            "Refusal without epistemic language",
            "No refusal + epistemic language",
            "Neither",
        ],
        xlim=(-1, 21),
        xticks=[0, 5, 10, 15, 20],
        xlabel="Prompts (20 total)",
        title="Original Instruct: joint outcomes",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    counts = result["overlap"]["adapted_capability_epistemic_refusals"]
    labels = ["No edit", "Capability edit", "Epistemic edit"]
    values = [counts[key] for key in ("baseline", "ablate_capability", "ablate_epistemic")]
    bars = axes[1].bar(labels, values, color=[*CHECKPOINT_COLORS, "#739B73"], width=0.65)
    for bar, count in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, count + 0.4, f"{count}/20", ha="center", fontsize=9
        )
    axes[1].set(
        ylim=(0, 23),
        yticks=[0, 5, 10, 15, 20],
        ylabel="Refusals (20 prompts)",
        title="Safety LoRA: capability/epistemic overlap",
    )
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].set_axisbelow(True)
    fig.suptitle("Epistemic language and refusal are separate response properties", fontsize=13)
    return fig, "overlap_joint_outcomes"


def geometry(result):
    data = result["geometry"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    layers = data["layers"]
    for name, color in zip(("safety", "epistemic", "capability"), COLORS):
        axes[0].plot(
            layers, data["between_checkpoints"][name], label=name.capitalize(), color=color
        )
    for checkpoint, color in zip(CHECKPOINTS, CHECKPOINT_COLORS):
        for pair, style in (("safety_capability", "-"), ("safety_epistemic", "--")):
            axes[1].plot(
                layers,
                data["within_checkpoint"][checkpoint][pair],
                style,
                color=color,
                label=f"{CHECKPOINTS[checkpoint]['label']}: {pair.replace('_', '/')}",
            )
    axes[0].set(title="Direction similarity across checkpoints", ylim=(-0.1, 1.05))
    axes[1].set(title="Direction similarity within each checkpoint", ylim=(-0.2, 1.05))
    for ax in axes:
        ax.set(xlabel="Layer", ylabel="Cosine similarity")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
    return fig, "supporting_geometry"
