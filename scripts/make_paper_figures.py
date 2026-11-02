#!/usr/bin/env python3
"""Plot the paper's reported results.

Inputs are frozen aggregate counts from the paper, not newly scored responses.
Creates PNG and vector PDF figures. No GPU, models, or GUI required.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "Acknowledgment": "#3274A1",
    "Instructions": "#E6A23C",
    "Unsupported": "#C75450",
    "Other": "#92999F",
}
BLUE = "#3274A1"
ORANGE = "#E58B36"

# Category order: acknowledgment, instructions, unsupported, other.
# Every capability condition contains the same 60 evaluation prompts.
BASELINE = [58, 2, 0, 0]
B_EDITED = [49, 10, 1, 0]
C_EDITED = [6, 8, 40, 6]

CONTROLS = {
    "No edit": BASELINE,
    "Random directions\n(each of 6)": [58, 2, 0, 0],
    "Epistemic direction": [52, 4, 3, 1],
    "Safety direction": [43, 5, 11, 1],
    "Safety orthogonal\nto capability": [59, 1, 0, 0],
    "Capability direction": C_EDITED,
}

STRENGTHS = np.array([
    0.0,
    0.41651599666722117,
    0.7904264640058756,
    1.5,
])
DOSE_COUNTS = np.array([
    BASELINE,
    [49, 3, 8, 0],
    [25, 7, 24, 4],
    C_EDITED,
])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def validate():
    rows = [
        BASELINE, B_EDITED, C_EDITED,
        *CONTROLS.values(), *DOSE_COUNTS.tolist(),
    ]
    for row in rows:
        if len(row) != 4 or sum(row) != 60:
            raise ValueError(f"Invalid capability counts: {row}")


def finish(fig, name, output_dir, dpi):
    for extension in ("png", "pdf"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(path)
    plt.close(fig)


def figure1(output_dir, dpi):
    """Same capability operator applied to both checkpoints."""
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained"
    )
    x = np.arange(2)
    width = 0.34

    panels = [
        ("Acknowledgment of access limits", 0),
        ("Unsupported assertions", 2),
    ]

    for ax, (title, category) in zip(axes, panels):
        for offset, label, edited, color in [
            (-width / 2, "B: original", B_EDITED, BLUE),
            (width / 2, "C: fine-tuned", C_EDITED, ORANGE),
        ]:
            values = np.array([
                BASELINE[category], edited[category]
            ])
            bars = ax.bar(
                x + offset, values / 60 * 100,
                width, color=color, label=label,
            )
            for bar, count in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{count}/60",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_title(title)
        ax.set_xticks(x, ["No edit", "Same activation edit"])
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Responses (%)")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "The same activation operator has different effects after fine-tuning",
        fontsize=13,
    )
    finish(fig, "figure1_fixed_intervention", output_dir, dpi)


def figure2(output_dir, dpi):
    """Direction controls and capability-edit strength in checkpoint C."""
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.8),
        gridspec_kw={"width_ratios": [1.35, 1]},
        layout="constrained",
    )

    ax = axes[0]
    values = np.array(list(CONTROLS.values()))
    y = np.arange(len(values))
    left = np.zeros(len(values))

    for column, (category, color) in enumerate(COLORS.items()):
        counts = values[:, column]
        ax.barh(
            y, counts, left=left, color=color,
            label=category, height=0.7,
        )
        for row, count in enumerate(counts):
            if count >= 4:
                ax.text(
                    left[row] + count / 2, row, str(count),
                    ha="center", va="center",
                    color="white" if category != "Instructions" else "black",
                    fontsize=9,
                )
        left += counts

    ax.set_yticks(y, list(CONTROLS))
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xlabel("Responses (60 prompts)")
    ax.set_title("A. Direction controls in C")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.17),
        ncol=2, frameon=False, fontsize=9,
    )

    ax = axes[1]
    for column, (category, color) in enumerate(COLORS.items()):
        ax.plot(
            STRENGTHS, DOSE_COUNTS[:, column],
            marker="o", linewidth=2,
            color=color, label=category,
        )

    ax.set_xlim(-0.04, 1.54)
    ax.set_ylim(0, 63)
    ax.set_xticks([0, 0.42, 0.79, 1.5])
    ax.set_xlabel("Intervention strength")
    ax.set_ylabel("Responses (60 prompts)")
    ax.set_title("B. Capability-direction strength in C")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Controls and strength sweep",
        fontsize=13,
    )
    finish(fig, "figure2_controls_and_strength", output_dir, dpi)


def figure3(output_dir, dpi):
    """Secondary overlap study; these outcomes use a separate rubric."""
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), layout="constrained"
    )

    # Joint outcomes for 20 B safety/epistemic prompts.
    labels = [
        "Refusal + epistemic language",
        "Refusal without epistemic language",
        "No refusal + epistemic language",
        "Neither",
    ]
    before = np.array([4, 16, 0, 0])
    after = np.array([3, 5, 10, 2])
    y = np.arange(4)

    ax = axes[0]
    for row, (start, end) in enumerate(zip(before, after)):
        ax.plot([start, end], [row, row], color="#B8BEC4", lw=3)

    ax.scatter(before, y, color=BLUE, s=65, label="Before edit", zorder=3)
    ax.scatter(
        after, y, color=ORANGE, s=65,
        label="After safety edit", zorder=3,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-1, 21)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Prompts (20 total)")
    ax.set_title("A. Joint outcomes in B")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # C capability/epistemic overlap group: 20 prompts.
    ax = axes[1]
    conditions = ["No edit", "Capability edit", "Epistemic edit"]
    counts = [20, 12, 20]
    bars = ax.bar(
        conditions, counts,
        color=[BLUE, ORANGE, "#739B73"], width=0.65,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.4, f"{count}/20",
            ha="center", fontsize=9,
        )

    ax.set_ylim(0, 23)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("Refusals (20 prompts)")
    ax.set_title("B. Capability/epistemic overlap in C")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)

    fig.suptitle(
        "More epistemic language does not necessarily mean more refusal",
        fontsize=13,
    )
    finish(fig, "figure3_overlap_joint_outcomes", output_dir, dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts" / "figures" / "revised_paper",
    )
    parser.add_argument(
        "--figures", nargs="+", type=int,
        choices=[1, 2, 3], default=[1, 2, 3],
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    if not 72 <= args.dpi <= 2400:
        parser.error("--dpi must be between 72 and 2400")

    validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    functions = {1: figure1, 2: figure2, 3: figure3}
    for number in dict.fromkeys(args.figures):
        functions[number](args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
