"""Pool the two frozen evaluation sets for blinded relabeling and score the shift."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

COHORTS = ("original", "fresh")
RECEIVERS = ("original_instruct", "seed42", "seed43", "seed44")
EDITS = ("evidence", "format")
LABELS = ("acknowledgment", "instructions", "unsupported", "other")
EXPECTED = {
    "original": "fceb0f69cb1614a940fa3e9adf2a30a7acdda1d29c15bec9aa9278f3c6728e4f",
    "fresh": "06bb77b746dedaabdbfd10287634d222b608dc9df446c7f55df2c8982892d646",
}


def load(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def grid(conditions):
    cells = {}
    prompts = {}
    for condition in conditions:
        key = tuple(condition[k] for k in ("cohort", "receiver", "edit"))
        require(key not in cells, f"Duplicate cell {key}")
        by_prompt = {row["prompt_id"]: row for row in condition["rows"]}
        require(len(by_prompt) == len(condition["rows"]) == 120, f"Incomplete cell {key}")
        reference = {pid: (row["family"], row["prompt"]) for pid, row in by_prompt.items()}
        require(prompts.setdefault(key[0], reference) == reference, f"Unpaired cell {key}")
        counts = Counter(family for family, _ in reference.values())
        require(len(counts) == 30 and set(counts.values()) == {4}, f"Wrong family grid {key}")
        cells[key] = by_prompt
    expected = {(cohort, receiver, edit)
                for cohort in COHORTS for receiver in RECEIVERS for edit in EDITS}
    require(cells.keys() == expected, "Wrong cohort/receiver/edit grid")
    return cells, prompts


def prepare(original, fresh, output):
    conditions = []
    for cohort, path in zip(COHORTS, (original, fresh)):
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == EXPECTED[cohort],
                f"{path} differs from the {cohort} resolved source used in the manuscript")
        study = json.loads(raw)
        del raw
        require(study["complete"] is True, f"Incomplete {cohort} run")
        for condition in study["conditions"]:
            if condition["source"] not in EDITS or condition["mode"] == "observe":
                continue
            require((condition["mode"], condition["layer"], condition["amount"])
                    == ("absolute_l2", 20, 4), "Unexpected intervention")
            receiver = condition["receiver"].replace("safety", "seed")
            rows = []
            for row in condition["rows"]:
                old = row["independent_review"]["outcome"]
                require(old in LABELS and row["input_truncated"] is False, "Invalid source row")
                rows.append({
                    "prompt_id": row["prompt_id"],
                    "family": row["domain"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "hit_token_limit": row["hit_token_limit"],
                    "prior_outcome": old,
                })
            conditions.append(dict(cohort=cohort, receiver=receiver,
                                   edit=condition["source"], rows=rows))
        del study
    grid(conditions)
    write(output, dict(complete=True, source_sha256=EXPECTED, conditions=conditions))
    print("Prepared 1,920 evidence/format responses across 60 families.")


def contrast(cells, prompts, stream):
    per_cohort = []
    for cohort in COHORTS:
        ids = sorted(prompts[cohort])
        family = [prompts[cohort][pid][0] for pid in ids]
        values = np.array([
            [[(cells[cohort, r, e][pid]["prior_outcome"] if stream == "prior"
               else cells[cohort, r, e][pid]["independent_review"][stream]) == "acknowledgment"
              for pid in ids] for e in EDITS] for r in RECEIVERS
        ], dtype=float)
        evidence_minus_format = values[:, 0] - values[:, 1]
        interaction = evidence_minus_format[1:].mean(axis=0) - evidence_minus_format[0]
        per_cohort.append([
            interaction[np.array(family) == key].mean() for key in sorted(set(family))
        ])
    return np.array(per_cohort)


def estimate(values, indexes):
    samples = np.stack([cohort[idx].mean(axis=1) for cohort, idx in zip(values, indexes)])
    points = values.mean(axis=1)

    def summary(point, draws):
        return [float(100 * point), *(100 * np.quantile(draws, (.025, .975))).tolist()]

    return {
        "original_pp": summary(points[0], samples[0]),
        "fresh_pp": summary(points[1], samples[1]),
        "fresh_minus_original_pp": summary(points[1] - points[0], samples[1] - samples[0]),
    }


def analyze(source, reviewed, output):
    raw = load(source)
    merged = load(reviewed)
    cells, prompts = grid(merged["conditions"])
    agreed = {}
    for cell in cells.values():
        for row in cell.values():
            review = row["independent_review"]
            labels = tuple(review[stream] for stream in ("outcome", "a", "b"))
            require(all(label in LABELS for label in labels), "Unresolved review")
            pair = row["prompt"], row["response"]
            require(agreed.setdefault(pair, labels) == labels, "Inconsistent duplicate labels")

    raw_rows = [(c, row) for c in raw["conditions"] for row in c["rows"]]
    merged_rows = [(c, row) for c in merged["conditions"] for row in c["rows"]]
    require(len(raw_rows) == len(merged_rows) == 1920, "Unexpected response count")
    for (old_c, old), (new_c, new) in zip(raw_rows, merged_rows):
        require(all(old_c[k] == new_c[k] for k in ("cohort", "receiver", "edit"))
                and old == {k: v for k, v in new.items() if k != "independent_review"},
                "Review changed more than labels")

    rng = np.random.default_rng(42)
    indexes = rng.integers(0, 30, (2, 20_000, 30))
    values = {stream: contrast(cells, prompts, stream) for stream in ("prior", "outcome", "a", "b")}
    estimates = {stream: estimate(v, indexes) for stream, v in values.items()}
    require(np.allclose([estimates["prior"][key][0] for key in ("original_pp", "fresh_pp")],
                        [-45.0, -13.0555555556]), "Prior estimates do not match the manuscript")
    changes = {stream: estimate(values[stream] - values["prior"], indexes)
               for stream in ("outcome", "a", "b")}
    report = {
        "source_sha256": raw["source_sha256"],
        "reviewed_sha256": hashlib.sha256(reviewed.read_bytes()).hexdigest(),
        "unique_pairs": len(agreed),
        "format": "[estimate, 95% interval lower, 95% interval upper]",
        "interval": "20,000 independent cohort draws, paired by family within cohort and across label streams",
        "estimates": estimates,
        "joint_minus_prior": changes,
    }
    write(output, report)
    print(json.dumps(report["estimates"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for name in ("original", "fresh", "output"):
        prep.add_argument(name, type=Path)
    score = sub.add_parser("analyze")
    for name in ("source", "reviewed", "output"):
        score.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.original, args.fresh, args.output)
    else:
        analyze(args.source, args.reviewed, args.output)
