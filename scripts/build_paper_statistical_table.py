#!/usr/bin/env python3

import json
from pathlib import Path

ROOT = Path("artifacts/safety_finetuning")

d = json.loads(
    (ROOT / "paper_statistical_summary.json").read_text()
)


def pfmt(p):
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def cifmt(ci):
    if ci is None:
        return "—"
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


lines = [
    "# Paper Statistical Summary",
    "",
    "## Primary confirmatory comparisons",
    "",
    "| ID | Comparison | n | Before | After | Effect | 95% CI | Raw p | Holm p | Source |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
]

for x in d["primary"]:
    lines.append(
        "| "
        + " | ".join([
            x["id"],
            x["comparison"],
            str(x["n"]),
            f'{x["before"]:.3f}',
            f'{x["after"]:.3f}',
            f'{x["effect"]:+.3f}',
            cifmt(x["ci95"]),
            pfmt(x["p_raw"]),
            pfmt(x["p_holm"]),
            x["source"],
        ])
        + " |"
    )

lines += [
    "",
    "Holm correction is applied only to the four pre-designated "
    "primary confirmatory comparisons.",
    "",
    "## Secondary behavioral comparisons",
    "",
    "| Comparison | n | Before | After | Effect | 95% CI | Raw p | Source |",
    "|---|---:|---:|---:|---:|---:|---:|---|",
]

for x in d["secondary_behavior"]:
    lines.append(
        "| "
        + " | ".join([
            x["comparison"],
            str(x["n"]),
            f'{x["before"]:.3f}',
            f'{x["after"]:.3f}',
            f'{x["effect"]:+.3f}',
            cifmt(x["ci95"]),
            pfmt(x["p_raw"]),
            x["source"],
        ])
        + " |"
    )

lines += [
    "",
    "## Statistical interpretation",
    "",
    "- Paired binary behavioral changes use exact McNemar/binomial "
    "tests over discordant pairs.",
    "- Effect sizes are absolute changes in behavioral rate.",
    "- 95% confidence intervals use paired bootstrap resampling.",
    "- Geometry analyses are descriptive effect-size analyses with "
    "prompt-resampling stability controls.",
    "- Capability labels are described as independent blind "
    "annotations.",
    "",
]

out = ROOT / "paper_statistical_summary.md"
out.write_text("\n".join(lines))

print(out.read_text())
print("Saved:", out)
