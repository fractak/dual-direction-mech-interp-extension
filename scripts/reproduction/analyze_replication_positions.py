"""Reproduce the frozen paired-family estimates from the two resolved exports."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

OUTCOMES = ("acknowledgment", "instructions", "unsupported", "other")
ORIGINAL = "original_instruct"
POSITION_MODES = {
    "position_final_fixed": "final_prefill",
    "position_previous_fixed": "previous_prefill",
    "position_two_back_fixed": "two_back_prefill",
    "position_user_last_fixed": "last_user_content",
}


def read_study(path, study):
    raw = path.read_bytes()
    source = json.loads(raw)
    assert source["complete"] is True
    is_replication = study == "family_replication"
    receivers = (ORIGINAL, "seed42", "seed43", "seed44") if is_replication else (ORIGINAL, "safety_lora")
    modes = ("baseline", "evidence", "format") if is_replication else ("baseline", *POSITION_MODES)
    grid, pairs = {}, {}
    reference = None
    for condition in source["conditions"]:
        receiver = condition["receiver"]
        mode = condition["mode"]
        if mode == "observe":
            mode = "baseline"
        elif is_replication:
            assert mode == "absolute_l2"
            mode = condition["source"]
        assert condition["layer"] == 20 and condition["amount"] == (0 if mode == "baseline" else 4)
        key = receiver, mode
        assert key not in grid
        rows = {row["prompt_id"]: row for row in condition["rows"]}
        assert len(rows) == len(condition["rows"]) == 120
        prompts = {key: (row["prompt"], row["domain"]) for key, row in rows.items()}
        reference = prompts if reference is None else reference
        assert prompts == reference
        for row in rows.values():
            review = row["independent_review"]
            a, b, outcome = (review[k] for k in ("a", "b", "outcome"))
            assert {a, b, outcome} <= set(OUTCOMES)
            assert review["adjudicated"] == (a != b)
            assert a != b or outcome == a
            pair = row["prompt"], row["response"]
            assert pairs.setdefault(pair, review) == review
            assert not row["input_truncated"]
        grid[key] = rows
    assert set(grid) == {(receiver, mode) for receiver in receivers for mode in modes}
    families = collections.Counter(family for _, family in reference.values())
    assert len(families) == 30 and set(families.values()) == {4}
    return grid, reference, pairs, receivers, modes, {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "component_plans": source.get("component_plan_sha256", {"full": source.get("plan_sha256")}),
    }


def summarize(grid, reference, receivers, modes, is_replication, vote, outcome, uncapped=False):
    ids = sorted(pid for pid in reference if not uncapped or all(
        not rows[pid]["hit_token_limit"] for rows in grid.values()
    ))
    families = sorted({reference[pid][1] for pid in ids})
    mask = np.array([[reference[pid][1] == family for pid in ids] for family in families], dtype=int)
    indices = np.random.default_rng(42).integers(0, len(families), (20000, len(families)))
    denominators = mask.sum(axis=1)[indices].sum(axis=1)

    def interval(values, alpha=0.05):
        draws = 100 * (mask @ values)[indices].sum(axis=1) / denominators
        return [float(100 * values.mean()), *np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist()]

    z = {
        key: np.array([rows[pid]["independent_review"][vote] == outcome for pid in ids], dtype=float)
        for key, rows in grid.items()
    }
    contrasts = {
        (receiver, mode): z[receiver, mode] - z[receiver, "baseline"] - z[ORIGINAL, mode] + z[ORIGINAL, "baseline"]
        for receiver in receivers[1:] for mode in modes[1:]
    }
    if is_replication:
        interactions = {r: contrasts[r, "evidence"] - contrasts[r, "format"] for r in receivers[1:]}
        estimates = {"pooled_interaction": interval(sum(interactions.values()) / 3)}
        estimates.update({f"{r}/interaction": interval(values) for r, values in interactions.items()})
        estimates.update({f"{r}/{m}": interval(values) for (r, m), values in contrasts.items()})
        estimates.update({f"pooled/{m}": interval(sum(contrasts[r, m] for r in receivers[1:]) / 3) for m in modes[1:]})
        result = {"contrasts": estimates}
    else:
        effects = {m: contrasts[receivers[1], m] for m in modes[1:]}
        comparisons = {m: effects["position_final_fixed"] - effects[m] for m in modes[2:]}
        result = {
            "receiver_contrasts": {m: interval(v) for m, v in effects.items()},
            "final_minus_other_pointwise95": {m: interval(v) for m, v in comparisons.items()},
            "primary_position_simultaneous95": {
                m: interval(comparisons[m], 0.025)
                for m in ("position_user_last_fixed", "position_previous_fixed")
            },
        }
    return dict(result, prompts=len(ids), families=len(families))


def analyze(path, study):
    grid, reference, pairs, receivers, modes, info = read_study(path, study)
    a = collections.Counter(row["a"] for row in pairs.values())
    b = collections.Counter(row["b"] for row in pairs.values())
    n = len(pairs)
    disagreement = sum(row["a"] != row["b"] for row in pairs.values())
    agreement = (n - disagreement) / n
    chance = sum(a[k] * b[k] for k in OUTCOMES) / n**2
    return {
        "study": study, **info,
        "bootstrap": {
            "draws": 20000, "seed": 42, "unit": "paired prompt family",
            "position_primary_intervals": "97.5% marginal percentile intervals for two Bonferroni comparisons",
        },
        "review": {
            "unique_pairs": n, "disagreements": disagreement, "agreement": agreement,
            "cohen_kappa": None if chance == 1 else (agreement - chance) / (1 - chance),
        },
        "conditions": {
            "/".join(key): {
                "counts": dict(collections.Counter(r["independent_review"]["outcome"] for r in rows.values())),
                "cap_hits": sum(r["hit_token_limit"] for r in rows.values()),
            }
            for key, rows in grid.items()
        },
        "analyses": {
            f"{subset}/{outcome}/{vote}": summarize(grid, reference, receivers, modes,
                study == "family_replication", vote, outcome, subset == "uncapped")
            for subset in ("all", "uncapped") for outcome in ("acknowledgment", "unsupported")
            for vote in ("outcome", "a", "b")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("family_replication", "position_control"))
    parser.add_argument("reviewed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.reviewed, args.study)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
