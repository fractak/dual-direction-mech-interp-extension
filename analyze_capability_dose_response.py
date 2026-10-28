import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path("artifacts/safety_finetuning")
N = 60
BOOTSTRAPS = 20000

def load(name):
    return json.loads((ROOT / name).read_text())

conditions = {
    "baseline": (
        "capability_C_baseline_reannotation_v2.json",
        "C-baseline",
        0.0,
    ),
    "low": (
        "capability_C_capability_a2_layer20_reannotation_v2.json",
        "C-capability-a2",
        0.416516,
    ),
    "medium": (
        "capability_C_capability_a_layer20_reannotation_v2.json",
        "C-capability-a",
        0.790426,
    ),
}

labels = {}

for name, (filename, prefix, strength) in conditions.items():
    data = load(filename)
    labels[name] = {
        field: np.array([
            int(data[f"{prefix}-{i:03d}"][field])
            for i in range(1, N + 1)
        ], dtype=int)
        for field in ("explicit_capability", "fabricated_answer")
    }

# Recover the original full-strength capability annotations
# from the verified blinded reannotation.
blind = load("capability_reannotation_v2.json")
key = load("capability_ablation_blind_key.json")

full = {}

for blind_id, meta in key.items():
    if (
        meta["checkpoint"] == "C_safety_lora"
        and meta["ablation"] == "ablate_C_capability"
    ):
        i = int(meta["source_index"]) + 1
        full[i] = blind[blind_id]

assert set(full) == set(range(1, N + 1))

labels["full"] = {
    field: np.array([
        int(full[i][field])
        for i in range(1, N + 1)
    ], dtype=int)
    for field in ("explicit_capability", "fabricated_answer")
}

order = ["baseline", "low", "medium", "full"]

strengths = {
    "baseline": 0.0,
    "low": 0.416516,
    "medium": 0.790426,
    "full": 1.5,
}

print("=== OVERALL DOSE RESPONSE ===")

for name in order:
    ack = labels[name]["explicit_capability"]
    unsupported = labels[name]["fabricated_answer"]

    print(
        f"{name:8s} "
        f"lambda={strengths[name]:.6f} "
        f"ack={ack.sum():2d}/60 "
        f"unsupported={unsupported.sum():2d}/60"
    )

comparisons = [
    ("baseline", "low"),
    ("low", "medium"),
    ("medium", "full"),
    ("baseline", "full"),
]

rng = np.random.default_rng(42)
samples = rng.integers(
    0, N, size=(BOOTSTRAPS, N)
)

print("\n=== PAIRED COMPARISONS ===")

for before, after in comparisons:
    print(f"\n{before} -> {after}")

    for field in (
        "explicit_capability",
        "fabricated_answer",
    ):
        a = labels[before][field]
        b = labels[after][field]

        diff = b - a
        gained = int(((a == 0) & (b == 1)).sum())
        lost = int(((a == 1) & (b == 0)).sum())
        discordant = gained + lost

        pvalue = (
            binomtest(
                min(gained, lost),
                discordant,
            ).pvalue
            if discordant else 1.0
        )

        boot = diff[samples].mean(axis=1)
        ci = np.quantile(boot, [0.025, 0.975])

        print(
            f"  {field}: "
            f"{a.sum()}/60 -> {b.sum()}/60"
        )
        print(
            f"    Difference: "
            f"{100 * diff.mean():+.1f} pp"
        )
        print(
            f"    95% paired bootstrap CI: "
            f"[{100 * ci[0]:+.1f}, "
            f"{100 * ci[1]:+.1f}] pp"
        )
        print(
            f"    Gained={gained}, "
            f"lost={lost}, "
            f"McNemar p={pvalue:.6g}"
        )

print("\n=== PROMPT-LEVEL MONOTONICITY ===")

U = np.stack([
    labels[name]["fabricated_answer"]
    for name in order
], axis=1)

# Monotone: once unsupported assertions appear,
# they never disappear at a higher strength.
monotone = np.all(
    np.diff(U, axis=1) >= 0,
    axis=1,
)

print(
    "Monotone unsupported-assertion trajectories:",
    f"{monotone.sum()}/{N}"
)

violations = np.flatnonzero(~monotone) + 1

print("Non-monotone prompt IDs:", violations.tolist())

print("\nUnsupported-assertion trajectories:")
for pattern in sorted(set(map(tuple, U))):
    count = sum(tuple(row) == pattern for row in U)
    ids = [
        i
        for i, row in enumerate(U, 1)
        if tuple(row) == pattern
    ]
    print(
        "".join(map(str, pattern)),
        f"count={count:2d}",
        "IDs=", ids,
    )

# Also test monotonic loss of acknowledgments.
A = np.stack([
    labels[name]["explicit_capability"]
    for name in order
], axis=1)

ack_monotone = np.all(
    np.diff(A, axis=1) <= 0,
    axis=1,
)

print(
    "\nMonotone acknowledgment-loss trajectories:",
    f"{ack_monotone.sum()}/{N}"
)

print(
    "Non-monotone acknowledgment IDs:",
    (np.flatnonzero(~ack_monotone) + 1).tolist()
)
