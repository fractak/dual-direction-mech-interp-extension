import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "artifacts/safety_finetuning"

FIELDS = ("explicit_capability", "instructions_only", "fabricated_answer")

CONDITIONS = {
    "baseline": ("capability_C_baseline_reannotation_v2.json", "C-baseline"),
    "safety": ("capability_C_safety_layer20_reannotation_v2.json", "C-safety"),
    "epistemic": ("capability_C_epistemic_layer20_reannotation_v2.json", "C-epistemic"),
    "safety_orthogonal": (
        "capability_C_safety_orthogonal_layer20_reannotation_v2.json",
        "C-safety-orthogonal",
    ),
    "capability_low": (
        "capability_C_capability_a2_layer20_reannotation_v2.json",
        "C-capability-a2",
    ),
    "capability_medium": (
        "capability_C_capability_a_layer20_reannotation_v2.json",
        "C-capability-a",
    ),
}

def load(filename):
    return json.loads((P / filename).read_text())

def validate(name, labels):
    assert len(labels) == 60, (name, len(labels))
    counts = dict.fromkeys(FIELDS, 0)
    other = 0

    for label in labels:
        values = [int(label[f]) for f in FIELDS]
        assert all(v in (0, 1) for v in values), (name, label)
        assert sum(values) <= 1, (name, label)

        if sum(values) == 0:
            other += 1
        for f, v in zip(FIELDS, values):
            counts[f] += v

    assert sum(counts.values()) + other == 60
    return {**counts, "other": other}

summary = {}

for name, (filename, prefix) in CONDITIONS.items():
    data = load(filename)
    expected = {f"{prefix}-{i:03d}" for i in range(1, 61)}
    assert set(data) == expected, f"ID mismatch: {name}"
    labels = [data[f"{prefix}-{i:03d}"] for i in range(1, 61)]
    summary[name] = validate(name, labels)

blind_labels = load("capability_reannotation_v2.json")
blind_key = load("capability_ablation_blind_key.json")

full = {}
for blind_id, meta in blind_key.items():
    if (
        meta["checkpoint"] == "C_safety_lora"
        and meta["ablation"] == "ablate_C_capability"
    ):
        i = int(meta["source_index"]) + 1
        assert i not in full
        full[i] = blind_labels[blind_id]

assert set(full) == set(range(1, 61))
summary["capability_full"] = validate(
    "capability_full", [full[i] for i in range(1, 61)]
)

cross = load("capability_cross_checkpoint_ablation.json")
baseline = cross["results"]["C_safety_lora"]["baseline"]["responses"]
assert len(baseline) == 60

structured = load("capability_C_structured_layer20.json")["results"]
for name in ("safety", "epistemic"):
    rows = structured[name]
    assert len(rows) == 60
    for i, row in enumerate(rows):
        assert row["source_index"] == i
        assert isinstance(row["response"], str)

for name in ("capability_a2", "capability_a"):
    data = load(f"capability_C_{name}_layer20.json")
    rows = data["results"][name]
    assert len(rows) == 60
    for i, row in enumerate(rows):
        assert row["source_index"] == i
        assert isinstance(row["response"], str)

random_summary = load("capability_C_random_six_seed_reannotation_v2.json")
assert len(random_summary["annotations"]) == 360
assert len(random_summary["summary"]) == 6

expected = {
    "baseline": (58, 2, 0, 0),
    "safety": (43, 5, 11, 1),
    "epistemic": (52, 4, 3, 1),
    "safety_orthogonal": (59, 1, 0, 0),
    "capability_low": (49, 3, 8, 0),
    "capability_medium": (25, 7, 24, 4),
    "capability_full": (6, 8, 40, 6),
}

for name, values in expected.items():
    observed = tuple(summary[name][f] for f in (*FIELDS, "other"))
    assert observed == values, (name, observed, values)

manifest = {
    "checkpoint": "C_safety_lora",
    "intervention_layer": "model.layers.20",
    "prompt_count": 60,
    "prompt_split": "expanded_eval",
    "prompt_seed": 42,
    "max_new_tokens": 128,
    "decoding": "greedy",
    "strengths": {
        "capability_low": 0.416516,
        "capability_medium": 0.790426,
        "capability_full": 1.5,
    },
    "direction_cosines": {
        "capability_safety": 0.526951,
        "capability_epistemic": 0.024747,
        "safety_epistemic": 0.103524,
    },
    "annotation_fields": list(FIELDS),
    "annotation_method": "manual semantic review",
    "condition_counts": summary,
    "random_control_seeds": list(range(2026, 2032)),
    "notes": [
        "Counts are based on manually reviewed annotations.",
        "fabricated_answer denotes an unsupported specific assertion, not verified factual falsity.",
        "Confidence intervals and tests must account for prompt pairing.",
        "Orthogonalized safety was renormalized to unit length.",
        "Dose-response monotonicity holds at tested strengths only.",
    ],
}

out = P / "capability_experiment_manifest.json"
out.write_text(json.dumps(manifest, indent=2) + "\n")

print("AUDIT PASSED")
print("\nCondition                 Ack  Instr  Unsupported  Other")
for name, counts in summary.items():
    print(
        f"{name:25s}"
        f"{counts['explicit_capability']:3d}"
        f"{counts['instructions_only']:7d}"
        f"{counts['fabricated_answer']:13d}"
        f"{counts['other']:7d}"
    )
print("\nSaved:", out)
