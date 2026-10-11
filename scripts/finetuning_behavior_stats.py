#!/usr/bin/env python3

import json
import math
import random
from pathlib import Path

from ddmi.evaluation.detectors import is_abstention, is_refusal


PATH = Path("artifacts/safety_finetuning/behavior_B_vs_C_full.json")
OUT = Path("artifacts/safety_finetuning/behavior_B_vs_C_stats.json")
SEED = 42
BOOTSTRAPS = 10000


def exact_binomial_two_sided(x, n):
    if n == 0:
        return 1.0
    k = min(x, n - x)
    return min(
        1.0,
        2.0 * sum(
            math.comb(n, i) * (0.5 ** n)
            for i in range(k + 1)
        ),
    )


def paired_stats(before, after):
    n = len(before)

    yy = sum(b and a for b, a in zip(before, after))
    yn = sum(b and not a for b, a in zip(before, after))
    ny = sum((not b) and a for b, a in zip(before, after))
    nn = n - yy - yn - ny

    rng = random.Random(SEED)
    diffs = []

    for _ in range(BOOTSTRAPS):
        idx = [rng.randrange(n) for _ in range(n)]
        b_rate = sum(before[i] for i in idx) / n
        a_rate = sum(after[i] for i in idx) / n
        diffs.append(a_rate - b_rate)

    diffs.sort()
    lo = diffs[int(0.025 * BOOTSTRAPS)]
    hi = diffs[int(0.975 * BOOTSTRAPS)]

    discordant = yn + ny

    return {
        "n": n,
        "before_rate": sum(before) / n,
        "after_rate": sum(after) / n,
        "change": (sum(after) - sum(before)) / n,
        "yes_yes": yy,
        "yes_to_no": yn,
        "no_to_yes": ny,
        "no_no": nn,
        "exact_mcnemar_p": exact_binomial_two_sided(
            min(yn, ny), discordant
        ),
        "bootstrap_95ci_change": [lo, hi],
    }


def main():
    d = json.loads(PATH.read_text())

    b = d["responses"]["B_instruct"]
    c = d["responses"]["C_safety_lora"]

    result = {
        "harmful_refusal": paired_stats(
            [is_refusal(x) for x in b["harmful"]],
            [is_refusal(x) for x in c["harmful"]],
        ),
        "benign_refusal": paired_stats(
            [is_refusal(x) for x in b["benign"]],
            [is_refusal(x) for x in c["benign"]],
        ),
        "unanswerable_abstention": paired_stats(
            [is_abstention(x) for x in b["unanswerable"]],
            [is_abstention(x) for x in c["unanswerable"]],
        ),
        "answerable_abstention": paired_stats(
            [is_abstention(x) for x in b["answerable"]],
            [is_abstention(x) for x in c["answerable"]],
        ),
    }

    OUT.write_text(json.dumps(result, indent=2) + "\n")

    for name, stats in result.items():
        print("\n" + name)
        print(f"  before: {stats['before_rate']:.3f}")
        print(f"  after:  {stats['after_rate']:.3f}")
        print(f"  change: {stats['change']:+.3f}")
        print(
            "  discordant:",
            stats["yes_to_no"],
            "vs",
            stats["no_to_yes"],
        )
        print(f"  p: {stats['exact_mcnemar_p']:.6f}")
        print(
            "  bootstrap 95% CI:",
            stats["bootstrap_95ci_change"],
        )


if __name__ == "__main__":
    main()
