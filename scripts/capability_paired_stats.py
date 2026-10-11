#!/usr/bin/env python3

import json
import math
import random
from collections import defaultdict

PATH = "artifacts/extension/capability_full_blind_annotation.json"
items = json.load(open(PATH))

by_prompt = defaultdict(dict)

for x in items:
    sample_id, condition = x["hidden_id"].rsplit("::", 1)
    by_prompt[sample_id][condition] = x["capability_refusal"]

rows = list(by_prompt.values())


def exact_mcnemar(rn, nr):
    n = rn + nr
    if n == 0:
        return 1.0

    k = min(rn, nr)

    p = 2 * sum(
        math.comb(n, i) * (0.5 ** n)
        for i in range(k + 1)
    )

    return min(1.0, p)


def paired_stats(after_name, bootstrap=10000, seed=42):
    baseline = [x["baseline"] for x in rows]
    after = [x[after_name] for x in rows]

    rn = sum(
        b and not a
        for b, a in zip(baseline, after)
    )
    nr = sum(
        (not b) and a
        for b, a in zip(baseline, after)
    )

    observed = (
        sum(baseline) - sum(after)
    ) / len(rows)

    rng = random.Random(seed)
    boots = []

    for _ in range(bootstrap):
        idx = [
            rng.randrange(len(rows))
            for _ in rows
        ]

        b = [baseline[i] for i in idx]
        a = [after[i] for i in idx]

        boots.append(
            (sum(b) - sum(a)) / len(rows)
        )

    boots.sort()

    lo = boots[int(0.025 * bootstrap)]
    hi = boots[int(0.975 * bootstrap)]

    return {
        "baseline_rate": sum(baseline) / len(rows),
        "after_rate": sum(after) / len(rows),
        "net_drop": observed,
        "refusal_to_nonrefusal": rn,
        "nonrefusal_to_refusal": nr,
        "exact_mcnemar_p": exact_mcnemar(rn, nr),
        "bootstrap_95_ci": [lo, hi],
    }


result = {
    "safety_ablation":
        paired_stats("ablate_safety"),
    "capability_ablation":
        paired_stats("ablate_capability"),
}

out = "artifacts/extension/capability_paired_stats.json"
json.dump(result, open(out, "w"), indent=2)

for name, x in result.items():
    print(f"\n=== {name} ===")
    print(f'baseline      {x["baseline_rate"]:.3f}')
    print(f'after         {x["after_rate"]:.3f}')
    print(f'net drop      {x["net_drop"]:.3f}')
    print(
        "discordant    "
        f'{x["refusal_to_nonrefusal"]} vs '
        f'{x["nonrefusal_to_refusal"]}'
    )
    print(
        f'exact p       {x["exact_mcnemar_p"]:.6f}'
    )
    print(
        "bootstrap CI  "
        f'[{x["bootstrap_95_ci"][0]:.3f}, '
        f'{x["bootstrap_95_ci"][1]:.3f}]'
    )

print("\nSaved:", out)
