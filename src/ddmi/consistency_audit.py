"""Apply a recorded consistency audit without altering original judgments."""

from __future__ import annotations

import copy
import csv

from ddmi.paper.analysis import capability_results
from ddmi.paper.evidence import OUTCOMES, load_rows, read_json, validate_rows, verify_checksums
from ddmi.provenance import sha256_file
from ddmi.publication import digest


def apply_audit(root, source_path, map_path, audit_path):
    verified = verify_checksums(root)
    source = read_json(source_path)
    frozen = load_rows(root)
    rows = source["rows"]
    if len(rows) != len(frozen):
        raise ValueError("Reviewed source row count differs from the frozen archive")
    for row, original in zip(rows, frozen):
        if any(row.get(key) != value for key, value in original.items()):
            raise ValueError("Reviewed source changed archived text, labels, order or metadata")
        review = row.get("independent_review", {})
        if any(review.get(key) not in OUTCOMES for key in ("a", "b", "outcome")):
            raise ValueError("Every source response requires resolved independent reviews")
        if "audited_outcome" in row:
            raise ValueError("Source already contains an applied audit")
    private = read_json(map_path)
    if private["source_sha256"] != sha256_file(root / "evidence/capability/responses.json"):
        raise ValueError("Review map does not identify the frozen capability source")
    mapping = private["mapping"]
    represented, expected = set(), set()
    for review_id, entry in mapping.items():
        decisions = set()
        if not entry["pointers"]:
            raise ValueError("Empty review mapping")
        for pointer in entry["pointers"]:
            if not pointer.startswith("row:"):
                raise ValueError("Audit requires an archived-response review map")
            index = int(pointer.split(":")[1])
            if index not in range(len(rows)) or index in represented:
                raise ValueError("Invalid or duplicate source-row pointer")
            represented.add(index)
            row = rows[index]
            if digest([row["prompt"], row["response"]]) != entry["text_sha256"]:
                raise ValueError("Review-map text does not match source")
            decisions.add(row["independent_review"]["outcome"])
        if len(decisions) != 1:
            raise ValueError("Duplicate text has inconsistent resolved labels")
        if decisions == {"other"}:
            expected.add(review_id)
    if represented != set(range(len(rows))):
        raise ValueError("Review map must cover every source row exactly once")
    audit = {}
    with open(audit_path, newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            key = item["review_id"]
            if key in audit or key not in mapping:
                raise ValueError("Unknown or duplicate audit review ID")
            if digest([item["prompt"], item["response"]]) != mapping[key]["text_sha256"]:
                raise ValueError("Audit prompt or response was modified")
            if item["outcome"] not in OUTCOMES or not item["notes"].strip():
                raise ValueError("Every audit item requires a valid outcome and reason")
            audit[key] = item
    if audit.keys() != expected:
        raise ValueError("Audit must cover exactly all previously resolved 'other' items")
    result = copy.deepcopy(source)
    for row in result["rows"]:
        row["audited_outcome"] = row["independent_review"]["outcome"]
    changed_unique, changed_rows = 0, 0
    for review_id, item in audit.items():
        changed_unique += item["outcome"] != "other"
        for pointer in mapping[review_id]["pointers"]:
            index = int(pointer.split(":")[1])
            row = result["rows"][index]
            changed_rows += item["outcome"] != row["audited_outcome"]
            row["audited_outcome"] = item["outcome"]
            row["consistency_review"] = {
                "review_id": review_id,
                "outcome": item["outcome"],
                "notes": item["notes"],
                "reviewer_type": "assistant",
            }
    metadata = {
        "source_sha256": sha256_file(source_path),
        "map_sha256": sha256_file(map_path),
        "audit_sha256": sha256_file(audit_path),
        "frozen_inputs_verified": verified,
        "audited_unique_pairs": len(audit),
        "changed_unique_pairs": changed_unique,
        "changed_source_rows": changed_rows,
        "selection": "All resolved 'other' items; not a full second review of every outcome",
        "clarification": "Explicitly identifying missing information needed to determine "
        "the requested answer counts as acknowledgment; merely requesting "
        "more details does not automatically count.",
        "interpretation": "Agent-review and assistant-adjudication sensitivity analysis of "
        "archived outputs; not human agreement or new model generation.",
    }
    result["consistency_audit"] = metadata

    def analysis_rows(key):
        selected = copy.deepcopy(result["rows"])
        for row in selected:
            row["outcome"] = (
                row["audited_outcome"] if key == "after" else row["independent_review"]["outcome"]
            )
        validate_rows(selected)
        return selected

    statistics = {
        "provenance": metadata,
        "before_consistency_audit": capability_results(analysis_rows("before")),
        "after_consistency_audit": capability_results(analysis_rows("after")),
    }
    return result, statistics
