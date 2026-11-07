"""Package prompt-level labels and reproduce the new paper's family-bootstrap results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import zipfile

import numpy as np

SEEDS = (42, 43, 44)
ARMS = ("mixed", "benign", "safety")
DOSES = (2, 4, 8)
PAIRS = ("a", "b")
BASE = (0, "evidence_a")
ORIG = "original_instruct"
SWEEP = (ORIG, *(f"seed{s}" for s in SEEDS))
TRAIN = tuple(f"{arm}{s}" for arm in ARMS for s in SEEDS)
OUTCOMES = ("acknowledgment", "instructions", "unsupported", "other")
STREAMS = ("outcome", "a", "b")
STUDY = Path("artifacts/receiver_robustness")
AUDIT_SEED = 20261107
AUDIT_FAMILIES = 10


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(c):
    return c["label"], c["amount"], c["source"]


def cells_expected(audit):
    doses = (4,) if audit else DOSES
    sweep = [BASE, *((dose, f"{kind}_{pair}") for dose in doses for pair in PAIRS
                     for kind in ("evidence", "format"))]
    train = (BASE, (4, "evidence_a"), (4, "format_a"))
    return {(r, *edit) for r in SWEEP for edit in sweep} | {
        (r, *edit) for r in TRAIN for edit in train}


def compact(source, audit=False):
    """Retain the frozen condition grid and all labels, with IDs in generation order."""
    if source.get("complete") is not True:
        raise ValueError("Expected the complete reviewed export")
    expected = cells_expected(audit)
    conditions = {}
    reference = None
    for condition in source["conditions"]:
        cell = key(condition)
        if cell not in expected or cell in conditions:
            raise ValueError(f"Unexpected or repeated cell: {cell}")
        rows = condition["rows"]
        ids = [(r["prompt_id"], r["domain"]) for r in rows]
        if reference is None:
            reference = ids
        if ids != reference or len({p for p, _ in ids}) != len(ids):
            raise ValueError(f"Prompt grid differs at {cell}")
        result = []
        for row in rows:
            review = row["independent_review"]
            scores = [review[s] for s in STREAMS]
            if not set(scores) <= set(OUTCOMES):
                raise ValueError("All outcomes must be resolved")
            result.append([int(row["hit_token_limit"]), *scores])
        conditions[cell] = result
    sizes = Counter(family for _, family in reference)
    expected_sizes = (AUDIT_FAMILIES, 1) if audit else (30, 4)
    if conditions.keys() != expected or (len(sizes), set(sizes.values())) != (
            expected_sizes[0], {expected_sizes[1]}):
        raise ValueError("Unexpected condition grid or family sizes")
    return {
        "prompt_columns": ["prompt_id", "family_id"],
        "observation_columns": ["hit_token_limit", *STREAMS],
        "prompts": reference,
        "conditions": [{"cell": k, "observations": v} for k, v in conditions.items()],
        "plan_sha256s": source["plan_sha256s"],
    }


def endpoint_vectors(grid, stream, outcome, doses):
    col = 1 + STREAMS.index(stream)
    table = {tuple(c["cell"]): np.array([r[col] == outcome for r in c["observations"]], dtype=int)
             for c in grid["conditions"]}

    def gap(receiver, pair, dose):
        return table[receiver, dose, f"evidence_{pair}"] - table[receiver, dose, f"format_{pair}"]

    result = {}
    for dose in doses:
        values = []
        for pair in PAIRS:
            original = gap(ORIG, pair, dose)
            by_seed = [gap(f"seed{s}", pair, dose) - original for s in SEEDS]
            values.extend(by_seed)
            result[f"sweep_{pair}_{dose}"] = np.mean(by_seed, axis=0)
        result[f"sweep_{dose}"] = np.mean(values, axis=0)
    for left, right in (("mixed", "benign"), ("safety", "benign"), ("mixed", "safety")):
        result[f"{left}_minus_{right}"] = np.mean(
            [gap(f"{left}{s}", "a", 4) - gap(f"{right}{s}", "a", 4) for s in SEEDS], axis=0)
    return result


def intervals(grid, values, use):
    by_family = {}
    for index, (_, family) in enumerate(grid["prompts"]):
        if use[index]:
            by_family.setdefault(family, []).append(index)
    families = list(by_family.values())
    sizes = np.array([len(indices) for indices in families])
    if len(sizes) < 2:
        raise ValueError("Fewer than two retained prompt families")
    draws = np.random.default_rng(42).integers(0, len(sizes), (20_000, len(sizes)))
    denominators = sizes[draws].sum(axis=1)
    results = {}
    for name, vector in values.items():
        totals = np.array([vector[indices].sum() for indices in families])
        samples = 100 * totals[draws].sum(axis=1) / denominators
        results[name] = [float(100 * totals.sum() / sizes.sum()),
                         *np.quantile(samples, (.025, .975)).tolist()]
    return results


def verify(grid, reference):
    if grid["plan_sha256s"] != reference["plan_sha256s"]:
        raise ValueError("Plan hashes differ from the experiment summary")
    n = len(grid["prompts"])
    all_prompts = np.ones(n, dtype=bool)
    uncapped = ~np.array([[r[0] for r in c["observations"]] for c in grid["conditions"]],
                        dtype=bool).any(axis=0)
    cap_hits = {"/".join(map(str, c["cell"])): sum(r[0] for r in c["observations"])
                for c in grid["conditions"]}
    if cap_hits != reference["cap_hits"]:
        raise ValueError("Per-cell cap counts differ from the experiment summary")
    def check(observed, expected):
        if not np.allclose(observed, expected, atol=1e-9, rtol=0):
            raise ValueError(f"Released interval mismatch: {observed} versus {expected}")
    checks = 0
    for outcome in ("acknowledgment", "unsupported"):
        for stream in STREAMS:
            values = endpoint_vectors(grid, stream, outcome, DOSES)
            expected = reference["all_prompts"][outcome][stream]
            observed = intervals(grid, values, all_prompts)
            check(observed["sweep_4"], expected["primary_sweep"])
            check(observed["mixed_minus_benign"], expected["primary_training"])
            for dose in DOSES:
                check(observed[f"sweep_{dose}"], expected["sweep_by_dose"][str(dose)])
                for pair in PAIRS:
                    check(observed[f"sweep_{pair}_{dose}"],
                          expected["sweep_by_pair_and_dose"][f"{pair}/{dose}"]["pooled"])
            for name, estimate in expected["training"].items():
                check(observed[name], estimate["pooled"])
            for condition in grid["conditions"]:
                cell = "/".join(map(str, condition["cell"]))
                count = sum(row[1 + STREAMS.index(stream)] == outcome for row in condition["observations"])
                if count != expected["counts"][cell]:
                    raise ValueError(f"Released outcome counts differ: {cell}/{stream}/{outcome}")
            checks += len(observed)
            if outcome == "acknowledgment":
                expected_uncapped = reference["complete_prompt_uncapped_acknowledgment"][stream]
                if int(uncapped.sum()) != expected_uncapped["n_prompts"]:
                    raise ValueError("Uncapped prompt count differs")
                obs = intervals(grid, values, uncapped)
                check(obs["sweep_4"], expected_uncapped["primary_sweep"])
                check(obs["mixed_minus_benign"], expected_uncapped["primary_training"])
                checks += 2
    print(f"Reproduced {checks} pooled intervals and the reported per-cell counts.")


def make_release(source, source_path, reference_path, output):
    reference = read(reference_path)
    grid = compact(source)
    verify(grid, reference)
    root = output / "release"
    root.mkdir()
    write(root / "reviewed_labels.json", grid)
    shutil.copyfile(reference_path, root / "results.json")
    shutil.copyfile(Path(__file__), root / Path(__file__).name)
    shutil.copyfile(Path(__file__).with_name("RELEASE.md"), root / "README.md")
    write(root / "manifest.json", {
        "version": "1.0.0",
        "upstream_reviewed_source_sha256": sha(source_path),
        "files_sha256": {p.name: sha(p) for p in sorted(root.iterdir())},
    })
    release = output / "receiver-robustness-analysis-v1.0.0.zip"
    with zipfile.ZipFile(release, "x", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(root.iterdir()):
            z.write(file, file.name)
    with zipfile.ZipFile(release) as z:
        if z.testzip() is not None:
            raise ValueError("Release ZIP failed integrity check")
    print(release)


def audit_export(source, output):
    rng = random.Random(AUDIT_SEED)
    groups = {}
    for row in source["conditions"][0]["rows"]:
        groups.setdefault(row["domain"], []).append(row["prompt_id"])
    families = rng.sample(sorted(groups), AUDIT_FAMILIES)
    selected = {rng.choice(sorted(groups[family])) for family in families}
    expected = cells_expected(audit=True)
    model_conditions = []
    blinded_conditions = []
    for c in source["conditions"]:
        if key(c) not in expected:
            continue
        rows = [row for row in c["rows"] if row["prompt_id"] in selected]
        model_conditions.append(dict(c, rows=rows))
        blinded_conditions.append(dict(c, rows=[{k: r[k] for k in
             ("prompt_id", "domain", "prompt", "response", "hit_token_limit")} for r in rows]))
    base = {"complete": True, "plan_sha256s": source["plan_sha256s"]}
    write(output / "audit_model_labels.json", compact(dict(base, conditions=model_conditions), audit=True))
    pooled = output / "audit_for_review.json"
    write(pooled, dict(base, conditions=blinded_conditions))
    subprocess.run([sys.executable, "scripts/review_responses.py", "export", "--source", str(pooled),
                    "--output-dir", str(output / "audit_review"), "--seed", str(AUDIT_SEED)], check=True)


def audit_analyze(model_path, human_path, output):
    model, human = read(model_path), compact(read(human_path), audit=True)
    model_ids = [tuple(c["cell"]) for c in model["conditions"]]
    human_ids = [tuple(c["cell"]) for c in human["conditions"]]
    if (list(map(tuple, model["prompts"])) != human["prompts"] or model_ids != human_ids
            or model["plan_sha256s"] != human["plan_sha256s"]):
        raise ValueError("Audit inputs have different prompts or conditions")
    use = np.ones(len(human["prompts"]), dtype=bool)
    results = {}
    automated = endpoint_vectors(model, "outcome", "acknowledgment", (4,))
    for stream in STREAMS:
        human_scores = endpoint_vectors(human, stream, "acknowledgment", (4,))
        change = {k: human_scores[k] - automated[k] for k in ("sweep_4", "mixed_minus_benign")}
        results[stream] = {
            "model_subset": intervals(model, {k: automated[k] for k in change}, use),
            "human_subset": intervals(human, {k: human_scores[k] for k in change}, use),
            "paired_human_minus_model": intervals(human, change, use),
        }
    by_recipe = {}
    for c, d in zip(model["conditions"], human["conditions"]):
        label, amount, direction = c["cell"]
        recipe = label[:-2] if label in TRAIN else "sweep"
        values = by_recipe.setdefault(recipe, dict(n=0, same=0, acknowledgment_same=0))
        for m, h in zip(c["observations"], d["observations"]):
            values["n"] += 1
            values["same"] += m[1] == h[1]
            values["acknowledgment_same"] += (m[1] == "acknowledgment") == (h[1] == "acknowledgment")
    write(output, dict(n_families=len(human["prompts"]), comparisons=results, agreement=by_recipe,
                       human_source_sha256=sha(human_path), model_labels_sha256=sha(model_path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    commands = {
        "prepare": ("output",),
        "verify": ("labels", "reference"),
        "audit-analyze": ("model", "human", "output"),
    }
    for command, fields in commands.items():
        p = actions.add_parser(command)
        for field in fields:
            p.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        source_path = STUDY / "resolved/reviewed_source.json"
        reference = STUDY / "results.json"
        source = read(source_path)
        args.output.mkdir(parents=True, exist_ok=False)
        make_release(source, source_path, reference, args.output)
        audit_export(source, args.output)
    elif args.action == "verify":
        verify(read(args.labels), read(args.reference))
    else:
        audit_analyze(args.model, args.human, args.output)


if __name__ == "__main__":
    main()
