"""Audit provenance, preservation of original labels, and complete coverage."""

import copy
import csv
import json
from pathlib import Path

import pytest

from ddmi.consistency_audit import apply_audit
from ddmi.paper.evidence import read_json
from ddmi.review import export_review

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    original = ROOT / "evidence/capability/responses.json"
    source = read_json(original)
    export_review(original, tmp_path / "review")
    map_path = tmp_path / "review/PRIVATE_review_map.json"
    mapping = read_json(map_path)["mapping"]
    for row in source["rows"]:
        row["independent_review"] = {
            "a": "acknowledgment",
            "b": "acknowledgment",
            "outcome": "acknowledgment",
            "notes": {"a": "retain"},
        }
    items = []
    for key, outcome in zip(list(mapping)[:3], ["acknowledgment", "unsupported", "other"]):
        for pointer in mapping[key]["pointers"]:
            row = source["rows"][int(pointer.split(":")[1])]
            row["independent_review"]["outcome"] = "other"
        items.append(
            {
                "review_id": key,
                "prompt": row["prompt"],
                "response": row["response"],
                "outcome": outcome,
                "notes": "A recorded audit rationale",
            }
        )
    source_path = tmp_path / "reviewed.json"
    source_path.write_text(json.dumps(source))
    audit_path = tmp_path / "audit.csv"
    write_audit(audit_path, items)
    return source_path, map_path, audit_path, items


def write_audit(path, items):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["review_id", "prompt", "response", "outcome", "notes"]
        )
        writer.writeheader()
        writer.writerows(items)


def test_audit_preserves_original_labels_and_records_separate_outcomes(inputs):
    source, mapping, audit, items = inputs
    before = read_json(source)
    result, stats = apply_audit(ROOT, source, mapping, audit)
    assert before == read_json(source)
    for old, new in zip(before["rows"], result["rows"]):
        assert all(new[k] == v for k, v in old.items())
        assert new["audited_outcome"] in {"acknowledgment", "unsupported", "other"}
    meta = stats["provenance"]
    assert meta["audited_unique_pairs"] == 3
    assert meta["changed_unique_pairs"] == 2
    assert meta["frozen_inputs_verified"] == 15
    assert meta["audit_sha256"]


@pytest.mark.parametrize("problem", ["missing_item", "modified_text", "blank_reason"])
def test_audit_rejects_incomplete_or_modified_inputs(inputs, problem):
    source, mapping, audit, items = inputs
    items = copy.deepcopy(items)
    if problem == "missing_item":
        items.pop()
    elif problem == "modified_text":
        items[0]["response"] += " altered"
    else:
        items[0]["notes"] = ""
    write_audit(audit, items)
    with pytest.raises(ValueError):
        apply_audit(ROOT, source, mapping, audit)


def test_audit_rejects_changed_archival_labels(inputs):
    source, mapping, audit, _ = inputs
    data = read_json(source)
    data["rows"][0]["outcome"] = "other"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="archived"):
        apply_audit(ROOT, source, mapping, audit)
