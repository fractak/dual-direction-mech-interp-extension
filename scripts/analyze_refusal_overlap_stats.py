#!/usr/bin/env python3

import json
import math
import random
from pathlib import Path

ROOT = Path("artifacts/refusal_overlap")

ANN = ROOT / "ablation_all_blind_annotated.json"
KEY = ROOT / "ablation_all_blind_key.json"
OUT = ROOT / "ablation_reviewed_stats.json"

BOOTSTRAP_N = 10000
SEED = 42


def exact_pvalue(lost, gained):
    n = lost + gained

    if n == 0:
        return 1.0

    k = min(lost, gained)

    p = 2 * sum(
        math.comb(n, i) * (0.5 ** n)
        for i in range(k + 1)
    )

    return min(1.0, p)


def bootstrap_ci(a, b):
    rng = random.Random(SEED)

    diffs = []

    n = len(a)

    for _ in range(BOOTSTRAP_N):
        idx = [
            rng.randrange(n)
            for _ in range(n)
        ]

        da = sum(a[i] for i in idx) / n
        db = sum(b[i] for i in idx) / n

        diffs.append(db - da)

    diffs.sort()

    lo = diffs[int(0.025 * BOOTSTRAP_N)]
    hi = diffs[int(0.975 * BOOTSTRAP_N)]

    return [lo, hi]


def paired_stats(a, b):
    assert len(a) == len(b)

    lost = sum(
        x and not y
        for x, y in zip(a, b)
    )

    gained = sum(
        (not x) and y
        for x, y in zip(a, b)
    )

    return {
        "n": len(a),
        "baseline_rate": sum(a) / len(a),
        "comparison_rate": sum(b) / len(b),
        "delta": (
            sum(b) / len(b)
            - sum(a) / len(a)
        ),
        "lost": lost,
        "gained": gained,
        "exact_p": exact_pvalue(
            lost,
            gained,
        ),
        "bootstrap_ci": bootstrap_ci(
            a,
            b,
        ),
    }


ann = json.load(open(ANN))
key = json.load(open(KEY))

rows = []

for x in ann:
    meta = key[x["blind_id"]]

    rows.append({
        **meta,
        "refusal": x["refusal"],
        "reason_safety": x["reason_safety"],
        "reason_capability": x["reason_capability"],
        "reason_epistemic": x["reason_epistemic"],
    })


def get(
    checkpoint,
    prompt_type,
    condition,
    field,
):
    xs = [
        x for x in rows
        if x["checkpoint"] == checkpoint
        and x["prompt_type"] == prompt_type
        and x["condition"] == condition
    ]

    xs = sorted(
        xs,
        key=lambda x: x["index"],
    )

    assert len(xs) == 20

    return [
        x[field]
        for x in xs
    ]


contrasts = {
    "B_safety_epistemic_refusal_safety_ablation": {
        "a": (
            "B_instruct",
            "safety_epistemic",
            "baseline",
            "refusal",
        ),
        "b": (
            "B_instruct",
            "safety_epistemic",
            "ablate_safety",
            "refusal",
        ),
    },

    "B_safety_epistemic_epi_reason_safety_ablation": {
        "a": (
            "B_instruct",
            "safety_epistemic",
            "baseline",
            "reason_epistemic",
        ),
        "b": (
            "B_instruct",
            "safety_epistemic",
            "ablate_safety",
            "reason_epistemic",
        ),
    },

    "C_capability_epistemic_refusal_cap_ablation": {
        "a": (
            "C_safety_lora",
            "capability_epistemic",
            "baseline",
            "refusal",
        ),
        "b": (
            "C_safety_lora",
            "capability_epistemic",
            "ablate_capability",
            "refusal",
        ),
    },

    "C_triple_refusal_safety_cap_ablation": {
        "a": (
            "C_safety_lora",
            "triple_overlap",
            "baseline",
            "refusal",
        ),
        "b": (
            "C_safety_lora",
            "triple_overlap",
            "ablate_safety_capability",
            "refusal",
        ),
    },

    "C_triple_refusal_add_epistemic": {
        "a": (
            "C_safety_lora",
            "triple_overlap",
            "ablate_safety_capability",
            "refusal",
        ),
        "b": (
            "C_safety_lora",
            "triple_overlap",
            "ablate_safety_capability_epistemic",
            "refusal",
        ),
    },
}


results = {}

for name, spec in contrasts.items():
    a = get(*spec["a"])
    b = get(*spec["b"])

    results[name] = paired_stats(
        a,
        b,
    )

OUT.write_text(
    json.dumps(results, indent=2) + "\n"
)

for name, r in results.items():
    print("\n" + name)

    print(
        f"  {r['baseline_rate']:.3f} "
        f"-> {r['comparison_rate']:.3f} "
        f"delta={r['delta']:+.3f}"
    )

    print(
        f"  lost={r['lost']} "
        f"gained={r['gained']} "
        f"p={r['exact_p']:.8g}"
    )

    print(
        "  CI "
        f"[{r['bootstrap_ci'][0]:+.3f}, "
        f"{r['bootstrap_ci'][1]:+.3f}]"
    )

print("\nSaved:", OUT)
