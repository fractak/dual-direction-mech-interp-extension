#!/usr/bin/env python3

import json
import math
import random
from pathlib import Path

PATH = Path("artifacts/safety_finetuning/cross_checkpoint_ablation_final.json")
OUT = Path("artifacts/safety_finetuning/cross_checkpoint_ablation_stats.json")

BOOT = 10000
SEED = 42


def exact_binomial_two_sided(x, n):
    if n == 0:
        return 1.0
    k = min(x, n - x)
    p = 2 * sum(
        math.comb(n, i) * (0.5 ** n)
        for i in range(k + 1)
    )
    return min(1.0, p)


def paired(before, after):
    n = len(before)

    before_rate = sum(before) / n
    after_rate = sum(after) / n

    lost = sum(a and not b for a, b in zip(before, after))
    gained = sum((not a) and b for a, b in zip(before, after))
    discordant = lost + gained

    rng = random.Random(SEED)
    deltas = []

    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        b0 = sum(before[i] for i in idx) / n
        b1 = sum(after[i] for i in idx) / n
        deltas.append(b1 - b0)

    deltas.sort()
    lo = deltas[int(0.025 * BOOT)]
    hi = deltas[int(0.975 * BOOT)]

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
        "bootstrap_ci_95": [lo, hi],
    }


def main():
    d = json.loads(PATH.read_text())
    out = {}

    for checkpoint, conditions in d["results"].items():
        baseline = conditions["baseline"]["labels"]
        out[checkpoint] = {}

        for condition, row in conditions.items():
            if condition == "baseline":
                continue

            out[checkpoint][condition] = {}

            for metric in [
                "safety_refusal",
                "epistemic_abstention",
            ]:
                out[checkpoint][condition][metric] = paired(
                    baseline[metric],
                    row["labels"][metric],
                )

    OUT.write_text(json.dumps(out, indent=2) + "\n")

    for checkpoint, conditions in out.items():
        print("\n" + checkpoint)

        for condition, metrics in conditions.items():
            print("\n ", condition)

            for metric, x in metrics.items():
                print(
                    f"    {metric:23} "
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
