"""Prepare and analyze a blinded, family-stratified human annotation audit."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from joint_review import COHORTS, EDITS, EXPECTED, LABELS, RECEIVERS, estimate, grid, load, require, write

BASE = Path(__file__).resolve().parent
CONDITIONS = ("baseline", *EDITS)
SAMPLING_SEED = 20261105
MATERIAL_PP = 10.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_rows(path, cohort):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == EXPECTED[cohort], f"Wrong {cohort} resolved source")
    result = {}
    for c in json.loads(raw)["conditions"]:
        if c["mode"] != "observe":
            continue
        receiver = c["receiver"].replace("safety", "seed")
        require(receiver not in result and c["amount"] == 0, "Unexpected baseline cell")
        result[receiver] = {row["prompt_id"]: row for row in c["rows"]}
    require(set(result) == set(RECEIVERS), "Missing baseline receiver")
    return result


def prepare(original, fresh, joint, output):
    reference = load(BASE / "joint_reference.json")
    require(sha(joint) == reference["reviewed_sha256"], "Wrong joint-reviewed source")
    run = load(joint)
    require(run["source_sha256"] == EXPECTED and run["complete"] is True, "Wrong joint source")
    cells, prompts = grid(run["conditions"])
    rng = np.random.default_rng(SAMPLING_SEED)
    conditions = []
    for cohort, path in zip(COHORTS, (original, fresh)):
        families = defaultdict(list)
        for pid, (family, _) in prompts[cohort].items():
            families[family].append(pid)
        sample = [str(rng.choice(sorted(families[family]))) for family in sorted(families)]
        baseline = baseline_rows(path, cohort)
        for receiver in RECEIVERS:
            require(set(baseline[receiver]) == set(prompts[cohort]), "Baseline prompt grid differs")
            for edit in CONDITIONS:
                rows = []
                for pid in sample:
                    family, prompt = prompts[cohort][pid]
                    if edit == "baseline":
                        row = baseline[receiver][pid]
                        require((row["domain"], row["prompt"]) == (family, prompt), "Unpaired baseline")
                        prior, relabel = row["independent_review"]["outcome"], None
                    else:
                        row = cells[cohort, receiver, edit][pid]
                        prior, relabel = row["prior_outcome"], row["independent_review"]["outcome"]
                        require(relabel in LABELS, "Missing joint label")
                    require(prior in LABELS, "Missing initial label")
                    rows.append(dict(prompt_id=pid, family=family, prompt=prompt, response=row["response"],
                                     initial_outcome=prior, joint_outcome=relabel,
                                     hit_token_limit=row["hit_token_limit"]))
                conditions.append(dict(cohort=cohort, receiver=receiver, edit=edit, rows=rows))
    write(output, dict(complete=True, sampling_seed=SAMPLING_SEED,
                       joint_review_sha256=reference["reviewed_sha256"], conditions=conditions))
    print("Selected 60 prompts and 720 response rows before exact-pair deduplication.")


def agreement(pairs):
    counts = Counter(pairs)
    n = sum(counts.values())
    observed = sum(counts[x, x] for x in LABELS) / n
    left = {x: sum(counts[x, y] for y in LABELS) for x in LABELS}
    right = {x: sum(counts[y, x] for y in LABELS) for x in LABELS}
    chance = sum(left[x] * right[x] for x in LABELS) / n**2
    return dict(n=n, agreement=observed, cohen_kappa=(observed-chance)/(1-chance) if chance < 1 else None,
                confusion={x: {y: counts[x, y] for y in LABELS} for x in LABELS})


def interaction(cells, label):
    # The family order is shared across every receiver and edit within each cohort.
    values = np.array([[[[label(row) == "acknowledgment" for row in cells[cohort, receiver, edit]]
                         for edit in EDITS] for receiver in RECEIVERS] for cohort in COHORTS], dtype=float)
    delta = values[:, :, 0] - values[:, :, 1]
    return delta[:, 1:].mean(axis=1) - delta[:, 0]


def assess(interval):
    _, low, high = interval
    if low > MATERIAL_PP or high < -MATERIAL_PP:
        return "material correction supported"
    if low > -MATERIAL_PP and high < MATERIAL_PP:
        return "material correction ruled out at the prespecified margin"
    return "inconclusive at the prespecified margin"


def analyze(source, reviewed, output):
    raw, merged = load(source), load(reviewed)
    require(raw.get("complete") is True and merged.get("complete") is True
            and raw["sampling_seed"] == SAMPLING_SEED, "Unexpected audit source")
    reference = load(BASE / "joint_reference.json")
    require(raw["joint_review_sha256"] == reference["reviewed_sha256"], "Wrong reference for audit")
    cells, pairs, by_cohort, by_condition = {}, {}, defaultdict(dict), defaultdict(dict)
    family_order = {}
    require(len(raw["conditions"]) == len(merged["conditions"]) == 24, "Wrong audit grid")
    for frozen, scored in zip(raw["conditions"], merged["conditions"]):
        keys = ("cohort", "receiver", "edit")
        key = tuple(frozen[k] for k in keys)
        cohort, _, edit = key
        require(key not in cells and key == tuple(scored[k] for k in keys), "Condition mismatch")
        require(len(frozen["rows"]) == len(scored["rows"]) == 30, "Incomplete cell")
        roster = [(row["family"], row["prompt_id"], row["prompt"]) for row in frozen["rows"]]
        require(len({item[0] for item in roster}) == 30 and
                roster == family_order.setdefault(cohort, roster), "Unpaired family grid")
        for old, new in zip(frozen["rows"], scored["rows"]):
            require(old == {k: v for k, v in new.items() if k != "independent_review"},
                    "Review changed more than labels")
            review = new["independent_review"]
            a, b, final = (review[k] for k in ("a", "b", "outcome"))
            require({a, b, final} <= set(LABELS) and (a != b or a == final), "Unresolved or invalid review")
            pair = old["prompt"], old["response"]
            require(pairs.setdefault(pair, (a, b, final)) == (a, b, final), "Inconsistent duplicate labels")
            by_cohort[cohort][pair] = (a, b)
            by_condition[edit][pair] = (a, b)
        cells[key] = scored["rows"]
    require(set(cells) == {(c, r, e) for c in COHORTS for r in RECEIVERS for e in CONDITIONS},
            "Missing audit cell")

    rng = np.random.default_rng(42)
    indexes = rng.integers(0, 30, (2, 20_000, 30))
    humans = {s: interaction(cells, lambda row: row["independent_review"][s])
              for s in ("outcome", "a", "b")}
    projections = {}
    comparisons = {}
    for target in ("initial", "joint"):
        old = interaction(cells, lambda row: row[f"{target}_outcome"])
        anchor = reference["estimates"]["prior" if target == "initial" else "outcome"]
        projections[target] = {}
        for stream, values in humans.items():
            correction = estimate(values - old, indexes)
            projections[target][stream] = {
                "audit_subset_model": estimate(old, indexes),
                "correction_pp": correction,
                "full_estimate_plus_correction_pp": {key: anchor[key][0] + v[0] for key, v in correction.items()},
                "materiality": {key: assess(v) for key, v in correction.items()},
            }
        comparisons[target] = agreement(
            (row[f"{target}_outcome"], row["independent_review"]["outcome"])
            for (_, _, edit), rows in cells.items() for row in rows if target == "initial" or edit != "baseline"
        )

    result = dict(
        reviewed_source_sha256=sha(reviewed), sampling_seed=SAMPLING_SEED, responses=720,
        materiality_margin_pp=MATERIAL_PP, intervals="paired family bootstrap, 20,000 draws, seed 42",
        human_agreement_unique_pairs=agreement((a, b) for a, b, _ in pairs.values()),
        human_agreement_by_cohort_unique_pairs={k: agreement(v.values()) for k, v in by_cohort.items()},
        human_agreement_by_condition_unique_pairs={k: agreement(v.values()) for k, v in by_condition.items()},
        model_vs_human_adjudicated_response_rows=comparisons,
        cap_hits=sum(row["hit_token_limit"] for rows in cells.values() for row in rows),
        audit_subset_human={k: estimate(v, indexes) for k, v in humans.items()},
        correction_analysis=projections,
    )
    write(output, result)
    print(result["human_agreement_unique_pairs"])
    print(result["correction_analysis"]["joint"]["outcome"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, names in (("prepare", ("original", "fresh", "joint", "output")),
                           ("analyze", ("source", "reviewed", "output"))):
        sub = commands.add_parser(command)
        for name in names:
            sub.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.original, args.fresh, args.joint, args.output)
    else:
        analyze(args.source, args.reviewed, args.output)
