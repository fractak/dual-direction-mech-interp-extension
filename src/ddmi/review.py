"""Blinded, deduplicated human review with agreement before adjudication."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

from ddmi.paper.evidence import OUTCOMES, read_json
from ddmi.provenance import sha256_file
from ddmi.publication import digest


def flattened(source):
    if "rows" in source:
        return [(f"row:{i}", row) for i, row in enumerate(source["rows"])]
    if source.get("complete") is not True:
        raise ValueError("Only complete prospective runs can be exported")
    return [
        (f"condition:{i}:row:{j}", row)
        for i, condition in enumerate(source["conditions"])
        for j, row in enumerate(condition["rows"])
    ]


def write_csv(path, rows, fields):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_review(source_path, output, seed=42):
    source_path, output = Path(source_path), Path(output)
    source = read_json(source_path)
    entries, unique = [], {}
    for pointer, row in flattened(source):
        if not isinstance(row["prompt"], str) or not isinstance(row["response"], str):
            raise ValueError("Review rows require prompt and response strings")
        key = (row["prompt"], row["response"])
        if key not in unique:
            unique[key] = {"prompt": key[0], "response": key[1], "pointers": []}
            entries.append(unique[key])
        unique[key]["pointers"].append(pointer)
    if not entries:
        raise ValueError("No responses available for review")
    random.Random(seed).shuffle(entries)
    output.mkdir(parents=True, exist_ok=False)
    mapping, public = {}, []
    for i, entry in enumerate(entries):
        review_id = f"r{i + 1:05d}"
        mapping[review_id] = {
            "pointers": entry["pointers"],
            "text_sha256": digest([entry["prompt"], entry["response"]]),
        }
        public.append(
            {
                "review_id": review_id,
                "prompt": entry["prompt"],
                "response": entry["response"],
                "outcome": "",
                "notes": "",
            }
        )
    fields = ["review_id", "prompt", "response", "outcome", "notes"]
    for reviewer in ("A", "B"):
        write_csv(output / f"reviewer_{reviewer}.csv", public, fields)
    private = {
        "source_path": os.path.relpath(source_path.resolve(), output.resolve()),
        "source_sha256": sha256_file(source_path),
        "seed": seed,
        "mapping": mapping,
    }
    (output / "PRIVATE_review_map.json").write_text(json.dumps(private, indent=2) + "\n")
    return {
        "unique_text_pairs": len(entries),
        "source_rows": sum(len(x["pointers"]) for x in entries),
    }


def load_labels(path, mapping, require_all=True):
    labels = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["review_id"]
            if key not in mapping or key in labels:
                raise ValueError("Unknown or duplicate review ID")
            if (
                "prompt" in row
                and digest([row["prompt"], row["response"]]) != mapping[key]["text_sha256"]
            ):
                raise ValueError("Review prompt/response was modified")
            if row["outcome"] not in OUTCOMES:
                raise ValueError(f"Missing or invalid outcome for {key}")
            labels[key] = row["outcome"]
    if require_all and labels.keys() != mapping.keys():
        raise ValueError("Every review ID must have a label")
    return labels


def merge_review(mapping_path, reviewer_a, reviewer_b, output, adjudications=None):
    private = read_json(mapping_path)
    if Path(reviewer_a).resolve() == Path(reviewer_b).resolve():
        raise ValueError("Two separate reviewer files are required")
    source_path = Path(private["source_path"])
    if not source_path.is_absolute():
        source_path = Path(mapping_path).resolve().parent / source_path
    if sha256_file(source_path) != private["source_sha256"]:
        raise ValueError("Source changed after review export")
    source = read_json(source_path)
    rows = dict(flattened(source))
    mapping = private["mapping"]
    a = load_labels(reviewer_a, mapping)
    b = load_labels(reviewer_b, mapping)
    adjudicated = load_labels(adjudications, mapping, False) if adjudications else {}

    def notes(path):
        if path is None:
            return {}
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            return {row["review_id"]: row.get("notes", "") for row in csv.DictReader(handle)}

    notes_a, notes_b, notes_final = notes(reviewer_a), notes(reviewer_b), notes(adjudications)
    conflicts = {k for k in mapping if a[k] != b[k]}
    if not adjudicated.keys() <= conflicts:
        raise ValueError("Adjudication may only resolve reviewer disagreements")
    n = len(mapping)
    observed = sum(a[k] == b[k] for k in mapping) / n
    expected = (
        sum(
            sum(v == outcome for v in a.values()) * sum(v == outcome for v in b.values())
            for outcome in OUTCOMES
        )
        / n**2
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    disagreements = []
    for key, entry in mapping.items():
        final = a[key] if a[key] == b[key] else adjudicated.get(key)
        for pointer in entry["pointers"]:
            row = rows[pointer]
            # Existing archival judgments are retained; this is a separate review.
            row["independent_review"] = {
                "a": a[key],
                "b": b[key],
                "outcome": final,
                "adjudicated": key in adjudicated,
                "notes": {
                    "a": notes_a.get(key, ""),
                    "b": notes_b.get(key, ""),
                    "adjudication": notes_final.get(key, ""),
                },
            }
        if key in conflicts:
            row = rows[entry["pointers"][0]]
            disagreements.append(
                {
                    "review_id": key,
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "reviewer_a": a[key],
                    "reviewer_b": b[key],
                    "outcome": adjudicated.get(key, ""),
                    "notes": notes_final.get(key, ""),
                }
            )
    report = {
        "unique_pairs": n,
        "agreement": observed,
        "cohen_kappa": kappa,
        "kappa_unit": "unique prompt-response pair, before adjudication",
        "confusion": {
            x: {y: sum(a[k] == x and b[k] == y for k in mapping) for y in OUTCOMES}
            for x in OUTCOMES
        },
        "disagreements": len(conflicts),
        "unresolved": len(conflicts - adjudicated.keys()),
        "all_resolved": not (conflicts - adjudicated.keys()),
        "reviewer_files_sha256": [sha256_file(reviewer_a), sha256_file(reviewer_b)],
        "source_sha256": private["source_sha256"],
        "adjudication_file_sha256": sha256_file(adjudications) if adjudications else None,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "reviewed_source.json").write_text(json.dumps(source, indent=2) + "\n")
    write_csv(
        output / "adjudications.csv",
        disagreements,
        ["review_id", "prompt", "response", "reviewer_a", "reviewer_b", "outcome", "notes"],
    )
    return report
