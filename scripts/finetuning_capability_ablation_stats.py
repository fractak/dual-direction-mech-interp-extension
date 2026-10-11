#!/usr/bin/env python3

import json
import math
import random
from pathlib import Path

PATH = Path(
    "artifacts/safety_finetuning/capability_cross_checkpoint_ablation.json"
)
OUT = Path(
    "artifacts/safety_finetuning/capability_cross_checkpoint_ablation_stats.json"
)

BOOT = 10000
SEED = 42


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


def paired(before, after):
    n = len(before)

    lost = sum(a and not b for a, b in zip(before, after))
    gained = sum((not a) and b for a, b in zip(before, after))
    discordant = lost + gained

    before_rate = sum(before) / n
    after_rate = sum(after) / n

    rng = random.Random(SEED)
    deltas = []

    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        b0 = sum(before[i] for i in idx) / n
        b1 = sum(after[i] for i in idx) / n
        deltas.append(b1 - b0)

    deltas.sort()

    return {
        "before": before_rate,
        "after": after_rate,
        "change": after_rate - before_rate,
        "lost": lost,
        "gained": gained,
        "discordant": discordant,
        "p_exact": exact_binomial_two_sided(
            min(lost, gained),
            discordant,
        ),
        "bootstrap_ci_95": [
            deltas[int(0.025 * BOOT)],
            deltas[int(0.975 * BOOT)],
        ],
    }


def main():
    d = json.loads(PATH.read_text())
    out = {}

    for checkpoint, conditions in d["results"].items():
        baseline = conditions["baseline"]["labels"]

        out[checkpoint] = {}

        for condition in [
            "ablate_B_capability",
            "ablate_C_capability",
        ]:
            out[checkpoint][condition] = paired(
                baseline,
                conditions[condition]["labels"],
            )

    OUT.write_text(json.dumps(out, indent=2) + "\n")

    for checkpoint, conditions in out.items():
        print("\n" + checkpoint)

        for condition, x in conditions.items():
            print(
                f"  {condition:24} "
                f"{x['before']:.3f} -> {x['after']:.3f} "
                f"delta={x['change']:+.3f} "
                f"lost/gained={x['lost']}/{x['gained']} "
                f"p={x['p_exact']:.8g} "
                f"CI=[{x['bootstrap_ci_95'][0]:+.3f}, "
                f"{x['bootstrap_ci_95'][1]:+.3f}]"
            )

    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
