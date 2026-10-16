#!/usr/bin/env python3

import json
from pathlib import Path

ROOT = Path("artifacts/refusal_overlap")

SRC = ROOT / "ablation_reviewed_stats.json"
OUT = ROOT / "ablation_reviewed_stats_final.json"

d = json.load(open(SRC))

items = [
    (name, row["exact_p"])
    for name, row in d.items()
]

items.sort(key=lambda x: x[1])

m = len(items)
running = 0.0
adjusted = {}

for rank, (name, p) in enumerate(items):
    holm = min(
        1.0,
        (m - rank) * p,
    )

    running = max(running, holm)
    adjusted[name] = running

for name, row in d.items():
    row["holm_p"] = adjusted[name]

OUT.write_text(
    json.dumps(d, indent=2) + "\n"
)

print("Saved:", OUT)

for name, row in d.items():
    print(
        f"{name}\n"
        f"  raw={row['exact_p']:.10g} "
        f"holm={row['holm_p']:.10g}"
    )
