"""Rebuild the main figure from the adjudicated aggregate results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

generated = Path("artifacts/reproduced/results")
fresh = json.loads((generated / "fresh_absolute_results.json").read_text())
followup = json.loads((generated / "learned_controls_results.json").read_text())
a = fresh["analyses"]["all/acknowledgment/outcome"]
b = followup["analyses"]["all/acknowledgment/outcome"]["contrasts"]
estimates = [
    ("Evidence relevance", a["evidence"]["pooled"]),
    ("Prose vs. table", a["format"]["pooled"]),
    ("Field order", [b["field_order"]["pooled"]["pp"], *b["field_order"]["pooled"]["ci95"]]),
    ("Batch code", [b["batch_code"]["pooled"]["pp"], *b["batch_code"]["pooled"]["ci95"]]),
]
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.1, 2.35))
fig.subplots_adjust(left=.24, right=.97, top=.95, bottom=.24)
ax.axvline(0, color="#68747c", linewidth=.9, linestyle="--")
ax.axhline(1.5, color="#c8cfd4", linewidth=.8)
for i, (label, (point, low, high)) in enumerate(estimates):
    color, marker = ("#175982", "o") if i < 2 else ("#8f4e21", "D")
    ax.errorbar(point, i, xerr=[[point-low], [high-point]], color=color,
                fmt=marker, markersize=5, capsize=3, linewidth=1.4)
    ax.text(high + 1.5, i, f"{point:+.1f}", ha="left", va="center", color=color)
ax.set(yticks=range(4), yticklabels=[row[0] for row in estimates],
       xlim=(-47, 45), ylim=(3.55, -.55),
       xlabel="Adapted minus original change in acknowledgment (percentage points)")
ax.tick_params(axis="y", length=0, pad=7)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#68747c")
fig.savefig(Path("artifacts/reproduced/figures") / "learned_direction_contrasts.pdf")
