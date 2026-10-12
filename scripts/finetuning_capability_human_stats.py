#!/usr/bin/env python3

import json
import math
from pathlib import Path

BLIND = Path(
    "artifacts/safety_finetuning/capability_B_vs_C_blind.json"
)
KEY = Path(
    "artifacts/safety_finetuning/capability_B_vs_C_blind_key.json"
)
OUT = Path(
    "artifacts/safety_finetuning/capability_behavior_human_stats.json"
)


def exact_binomial_two_sided(x, n):
    if n == 0:
        return 1.0
    k = min(x, n - x)
    return min(
        1.0,
        2 * sum(
            math.comb(n, i) * (0.5 ** n)
            for i in range(k + 1)
        ),
    )


blind = json.loads(BLIND.read_text())
key = json.loads(KEY.read_text())

by_sample = {}

for x in blind:
    assert x["capability_refusal"] is not None

    info = key[x["blind_id"]]
    sid = info["sample_id"]
    cond = info["condition"]

    by_sample.setdefault(sid, {})[cond] = x["capability_refusal"]

rows = list(by_sample.values())

assert len(rows) == 60
assert all(set(x) == {"B", "C"} for x in rows)

b = [x["B"] for x in rows]
c = [x["C"] for x in rows]

rr = sum(x and y for x, y in zip(b, c))
rn = sum(x and not y for x, y in zip(b, c))
nr = sum((not x) and y for x, y in zip(b, c))
nn = sum((not x) and (not y) for x, y in zip(b, c))

discordant = rn + nr

p = exact_binomial_two_sided(
    min(rn, nr),
    discordant,
)

result = {
    "n": len(rows),
    "B_rate": sum(b) / len(b),
    "C_rate": sum(c) / len(c),
    "change": (sum(c) - sum(b)) / len(b),
    "transitions": {
        "refusal_to_refusal": rr,
        "refusal_to_nonrefusal": rn,
        "nonrefusal_to_refusal": nr,
        "nonrefusal_to_nonrefusal": nn,
    },
    "paired_exact_p": p,
    "annotation": "blind_human_capability_refusal",
}

OUT.write_text(json.dumps(result, indent=2) + "\n")

print(json.dumps(result, indent=2))
print("\nSaved:", OUT)
