"""Recompute the revised paper's results from normalized human annotations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ddmi.checkpoints import CHECKPOINTS
from ddmi.paper.evidence import OUTCOMES, counts, load_rows, read_json, select, verify_checksums
from ddmi.paper.stats import bootstrap_indices, delta_summary, holm, paired


def outcome_flags(rows, outcome):
    return np.array([row["outcome"] == outcome for row in rows])


def capability_results(rows):
    conditions = {}
    for checkpoint in CHECKPOINTS:
        conditions[checkpoint] = {}
        for name in sorted({r["condition"] for r in rows if r["checkpoint"] == checkpoint}):
            members = select(rows, checkpoint, name)
            first = members[0]
            metadata = {k: first[k] for k in ("layer", "strength", "direction_checkpoint")}
            if any(any(r[k] != v for k, v in metadata.items()) for r in members):
                raise ValueError(f"Inconsistent intervention settings: {checkpoint}/{name}")
            conditions[checkpoint][name] = {"counts": counts(members), **metadata}

    indices = bootstrap_indices(60)
    main = {}
    changes = {}
    for checkpoint in CHECKPOINTS:
        baseline = select(rows, checkpoint, "baseline")
        edited = select(rows, checkpoint, "capability_from_safety_lora")
        if any(r["layer"] != "model.layers.20" or r["strength"] != 1.5 for r in edited):
            raise ValueError("Main comparison must use the safety-LoRA layer-20 vector at 1.5")
        main[checkpoint], changes[checkpoint] = {}, {}
        for outcome in ("acknowledgment", "unsupported"):
            before, after = outcome_flags(baseline, outcome), outcome_flags(edited, outcome)
            main[checkpoint][outcome] = paired(before, after, indices)
            changes[checkpoint][outcome] = after.astype(int) - before.astype(int)
    difference = {
        outcome: delta_summary(
            changes["safety_lora"][outcome] - changes["original_instruct"][outcome], indices
        )
        for outcome in ("acknowledgment", "unsupported")
    }

    # Sixteen exploratory tests, adjusted as one retrospective family.
    tests = {}
    for checkpoint in CHECKPOINTS:
        before = outcome_flags(select(rows, checkpoint, "baseline"), "acknowledgment")
        for condition in conditions[checkpoint]:
            if condition == "baseline":
                continue
            after = outcome_flags(select(rows, checkpoint, condition), "acknowledgment")
            tests[f"{checkpoint}/{condition}/acknowledgment"] = paired(before, after, indices)
    tests["safety_lora/capability_from_safety_lora/unsupported"] = main["safety_lora"][
        "unsupported"
    ]
    adjusted = holm({k: v["exact_p"] for k, v in tests.items()})
    for key, value in tests.items():
        value["holm_p"] = adjusted[key]

    random_controls = {}
    baseline = select(rows, "safety_lora", "baseline")
    for seed in range(2026, 2032):
        random_rows = select(rows, "safety_lora", f"random_seed{seed}")
        random_controls[str(seed)] = {
            "counts": counts(random_rows),
            "changed_text": sum(
                a["response"] != b["response"] for a, b in zip(baseline, random_rows)
            ),
            "changed_outcome": sum(
                a["outcome"] != b["outcome"] for a, b in zip(baseline, random_rows)
            ),
        }
    return {
        "conditions": conditions,
        "same_operator": main,
        "difference_in_changes": difference,
        "exploratory_tests": tests,
        "random_controls": random_controls,
    }


def overlap_results(rows):
    def get(checkpoint, category, condition, field="refusal"):
        return np.array([r[field] for r in select(rows, checkpoint, condition, category)])

    specs = {
        "original_safety_epistemic_refusal": (
            "original_instruct",
            "safety_epistemic",
            "baseline",
            "ablate_safety",
            "refusal",
        ),
        "original_safety_epistemic_language": (
            "original_instruct",
            "safety_epistemic",
            "baseline",
            "ablate_safety",
            "reason_epistemic",
        ),
        "adapted_capability_epistemic_refusal": (
            "safety_lora",
            "capability_epistemic",
            "baseline",
            "ablate_capability",
            "refusal",
        ),
        "adapted_triple_joint_edit": (
            "safety_lora",
            "triple_overlap",
            "baseline",
            "ablate_safety_capability",
            "refusal",
        ),
        "adapted_triple_add_epistemic": (
            "safety_lora",
            "triple_overlap",
            "ablate_safety_capability",
            "ablate_safety_capability_epistemic",
            "refusal",
        ),
    }
    tests = {}
    for name, (checkpoint, category, before, after, field) in specs.items():
        tests[name] = paired(
            get(checkpoint, category, before, field),
            get(checkpoint, category, after, field),
            bootstrap_indices(20),
        )
    adjusted = holm({k: v["exact_p"] for k, v in tests.items()})
    for key, value in tests.items():
        value["holm_p"] = adjusted[key]
    joint = {}
    for condition in ("baseline", "ablate_safety"):
        refusal = get("original_instruct", "safety_epistemic", condition)
        epistemic = get("original_instruct", "safety_epistemic", condition, "reason_epistemic")
        joint[condition] = [
            int(np.sum(mask))
            for mask in (
                refusal & epistemic,
                refusal & ~epistemic,
                ~refusal & epistemic,
                ~refusal & ~epistemic,
            )
        ]
    new_mentions = ~get(
        "original_instruct", "safety_epistemic", "baseline", "reason_epistemic"
    ) & get("original_instruct", "safety_epistemic", "ablate_safety", "reason_epistemic")
    return {
        "joint_outcomes_original": joint,
        "new_epistemic_mentions": int(new_mentions.sum()),
        "new_mentions_with_refusal": int(
            np.sum(new_mentions & get("original_instruct", "safety_epistemic", "ablate_safety"))
        ),
        "adapted_capability_epistemic_refusals": {
            condition: int(get("safety_lora", "capability_epistemic", condition).sum())
            for condition in ("baseline", "ablate_capability", "ablate_epistemic")
        },
        "exploratory_tests": tests,
    }


def geometry_results(root):
    directions = {}
    for checkpoint in CHECKPOINTS:
        directions[checkpoint] = {}
        for name in ("safety", "epistemic", "capability"):
            artifact = read_json(Path(root) / f"evidence/directions/{checkpoint}/{name}.json")
            if artifact["checkpoint"] != checkpoint or artifact["precision"] != "bnb_nf4":
                raise ValueError("Direction checkpoint or precision mismatch")
            vectors = []
            for layer in range(32):
                vector = np.array(artifact["directions"][f"model.layers.{layer}"]["direction"])
                if (
                    vector.shape != (4096,)
                    or not np.isfinite(vector).all()
                    or np.linalg.norm(vector) == 0
                ):
                    raise ValueError("Invalid archived direction vector")
                vectors.append(vector / np.linalg.norm(vector))
            directions[checkpoint][name] = np.array(vectors)

    def cosine(left, right):
        return np.einsum("ij,ij->i", left, right).tolist()

    original, adapted = directions["original_instruct"], directions["safety_lora"]
    between = {name: cosine(original[name], adapted[name]) for name in original}
    within = {
        checkpoint: {
            f"safety_{name}": cosine(vectors["safety"], vectors[name])
            for name in ("epistemic", "capability")
        }
        for checkpoint, vectors in directions.items()
    }
    return {
        "layers": list(range(32)),
        "between_checkpoints": between,
        "mean_between_checkpoints": {
            name: float(np.mean(values)) for name, values in between.items()
        },
        "within_checkpoint": within,
        "adapted_layer20": {
            "capability_safety": float(np.dot(adapted["capability"][20], adapted["safety"][20])),
            "capability_epistemic": float(
                np.dot(adapted["capability"][20], adapted["epistemic"][20])
            ),
        },
    }


def analyze(root, check=True):
    checked = verify_checksums(root) if check else 0
    return {
        "schema_version": 1,
        "checkpoints": CHECKPOINTS,
        "provenance": {
            "frozen_files_verified": checked,
            "bootstrap_samples": 20000,
            "bootstrap_seed": 42,
            "interval": "paired percentile, linear quantiles",
        },
        "capability": capability_results(load_rows(root)),
        "overlap": overlap_results(load_rows(root, overlap=True)),
        "geometry": geometry_results(root),
    }


def save_results(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Recomputed paper results",
        "",
        "Counts use the final human outcome rubric.",
        "",
        "| Checkpoint | Condition | Acknowledgment | Instructions | Unsupported | Other |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for checkpoint, conditions in result["capability"]["conditions"].items():
        for name, condition in conditions.items():
            values = " | ".join(str(condition["counts"][key]) for key in OUTCOMES)
            lines.append(f"| {CHECKPOINTS[checkpoint]['label']} | {name} | {values} |")
    lines += [
        "",
        "Unsupported means a specific assertion without supplied evidence; "
        "factual falsity was not established.",
        "",
        "All intervals are exploratory and conditional on these prompts and this checkpoint pair.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
