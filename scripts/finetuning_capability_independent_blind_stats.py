#!/usr/bin/env python3

import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("artifacts/safety_finetuning")

BASE_BLIND = ROOT / "capability_B_vs_C_blind.json"
BASE_KEY = ROOT / "capability_B_vs_C_blind_key.json"

ABL_BLIND = ROOT / "capability_ablation_blind.json"
ABL_KEY = ROOT / "capability_ablation_blind_key.json"

OUT = ROOT / "capability_cross_checkpoint_ablation_independent_blind_stats.json"

BOOTSTRAP_SEED = 42
N_BOOT = 10000


def exact_mcnemar(lost, gained):
    n = lost + gained
    if n == 0:
        return 1.0

    k = min(lost, gained)

    p = 2 * sum(
        math.comb(n, i) * (0.5 ** n)
        for i in range(k + 1)
    )

    return min(1.0, p)


def bootstrap_ci(before, after):
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(before)

    diffs = []

    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]

        b = sum(before[i] for i in idx) / n
        a = sum(after[i] for i in idx) / n

        diffs.append(a - b)

    diffs.sort()

    lo = diffs[int(0.025 * N_BOOT)]
    hi = diffs[int(0.975 * N_BOOT)]

    return [lo, hi]


base_blind = json.loads(BASE_BLIND.read_text())
base_key = json.loads(BASE_KEY.read_text())

abl_blind = json.loads(ABL_BLIND.read_text())
abl_key = json.loads(ABL_KEY.read_text())

assert all(
    x["capability_refusal"] is not None
    for x in base_blind
)

assert all(
    x["capability_refusal"] is not None
    for x in abl_blind
)

# ------------------------------------------------------------
# Baseline labels indexed by checkpoint condition + sample_id
# ------------------------------------------------------------
baseline = {}

for item in base_blind:
    meta = base_key[item["blind_id"]]

    condition = meta["condition"]  # B or C
    sample_id = meta["sample_id"]

    key = (condition, sample_id)

    assert key not in baseline

    baseline[key] = bool(item["capability_refusal"])

assert len(baseline) == 120

# ------------------------------------------------------------
# Ablation labels grouped by checkpoint + ablation condition
# ------------------------------------------------------------
groups = defaultdict(dict)

for item in abl_blind:
    meta = abl_key[item["blind_id"]]

    checkpoint = meta["checkpoint"]
    ablation = meta["ablation"]
    sample_id = meta["sample_id"]

    assert sample_id not in groups[(checkpoint, ablation)]

    groups[(checkpoint, ablation)][sample_id] = bool(
        item["capability_refusal"]
    )

assert len(groups) == 4
assert all(len(x) == 60 for x in groups.values())

checkpoint_to_baseline = {
    "B_instruct": "B",
    "C_safety_lora": "C",
}

result = {
    "annotation": "independent_blind_annotation",
    "n_per_condition": 60,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_samples": N_BOOT,
    "results": {},
}

for checkpoint in [
    "B_instruct",
    "C_safety_lora",
]:
    result["results"][checkpoint] = {}

    baseline_condition = checkpoint_to_baseline[checkpoint]

    for ablation in [
        "ablate_B_capability",
        "ablate_C_capability",
    ]:
        labeled = groups[(checkpoint, ablation)]

        sample_ids = sorted(labeled)

        before = [
            baseline[(baseline_condition, sid)]
            for sid in sample_ids
        ]

        after = [
            labeled[sid]
            for sid in sample_ids
        ]

        n = len(before)
        assert n == 60

        before_rate = sum(before) / n
        after_rate = sum(after) / n
        change = after_rate - before_rate

        rr = sum(
            b and a
            for b, a in zip(before, after)
        )
        rn = sum(
            b and not a
            for b, a in zip(before, after)
        )
        nr = sum(
            (not b) and a
            for b, a in zip(before, after)
        )
        nn = sum(
            (not b) and (not a)
            for b, a in zip(before, after)
        )

        stats = {
            "n": n,
            "baseline_rate": before_rate,
            "ablation_rate": after_rate,
            "change": change,
            "transitions": {
                "refusal_to_refusal": rr,
                "refusal_to_nonrefusal": rn,
                "nonrefusal_to_refusal": nr,
                "nonrefusal_to_nonrefusal": nn,
            },
            "lost_refusals": rn,
            "gained_refusals": nr,
            "paired_exact_p": exact_mcnemar(rn, nr),
            "bootstrap_95_ci_change": bootstrap_ci(
                before,
                after,
            ),
        }

        result["results"][checkpoint][ablation] = stats

OUT.write_text(json.dumps(result, indent=2) + "\n")

print(json.dumps(result, indent=2))
print("\nSaved:", OUT)
