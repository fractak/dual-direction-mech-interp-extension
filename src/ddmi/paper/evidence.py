"""Read normalized evidence, with explicit provenance and pairing checks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ddmi.checkpoints import CHECKPOINTS

OUTCOMES = ("acknowledgment", "instructions", "unsupported", "other")
OVERLAP_FIELDS = ("refusal", "reason_safety", "reason_capability", "reason_epistemic")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing input: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}: {exc}") from exc


def verify_checksums(root):
    root = Path(root)
    manifest = read_json(root / "evidence/manifest.json")
    for filename, expected in manifest["files"].items():
        path = root / filename
        if not path.is_file():
            raise ValueError(f"Missing frozen evidence: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen input changed: {path}")
    return len(manifest["files"])


def validate_rows(rows, overlap=False):
    """Reject duplicate, missing, reordered, or inconsistently labeled contexts."""
    groups = defaultdict(list)
    text_outcomes = {}
    expected_size = 20 if overlap else 60
    for row in rows:
        if row["checkpoint"] not in CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint: {row['checkpoint']}")
        if not isinstance(row["prompt"], str) or not isinstance(row["response"], str):
            raise ValueError("Prompt and response must be strings")
        group = (row["checkpoint"], row["condition"])
        if overlap:
            group += (row["prompt_type"],)
            if any(type(row[field]) is not bool for field in OVERLAP_FIELDS):
                raise ValueError("Overlap annotations must be booleans")
        else:
            if row["outcome"] not in OUTCOMES:
                raise ValueError(f"Unknown final outcome: {row['outcome']}")
            text = (row["prompt"], row["response"])
            if text_outcomes.setdefault(text, row["outcome"]) != row["outcome"]:
                raise ValueError("Identical prompt/response received different final outcomes")
        groups[group].append(row)
    for group, members in groups.items():
        indices = sorted(row["source_index"] for row in members)
        if indices != list(range(expected_size)):
            raise ValueError(f"Missing or duplicate prompt indices in {group}")
    expected_groups = 52 if overlap else 17
    if len(groups) != expected_groups:
        raise ValueError(f"Expected {expected_groups} condition groups; got {len(groups)}")
    # Context pairing must agree across all conditions, including checkpoints.
    contexts = {}
    for row in rows:
        key = ((row["prompt_type"],) if overlap else ()) + (row["source_index"],)
        value = (row["prompt_id"], row["prompt"])
        if contexts.setdefault(key, value) != value:
            raise ValueError(f"Prompt pairing mismatch at {key}")
    return groups


def load_rows(root, overlap=False):
    name = "overlap" if overlap else "capability"
    data = read_json(Path(root) / f"evidence/{name}/responses.json")
    if data["schema_version"] != 1:
        raise ValueError("Unsupported evidence schema")
    rows = data["rows"]
    validate_rows(rows, overlap=overlap)
    return rows


def select(rows, checkpoint, condition, prompt_type=None):
    selected = sorted(
        (
            row
            for row in rows
            if row["checkpoint"] == checkpoint
            and row["condition"] == condition
            and (prompt_type is None or row.get("prompt_type") == prompt_type)
        ),
        key=lambda row: row["source_index"],
    )
    if not selected:
        raise ValueError(f"Missing condition: {checkpoint}/{condition}/{prompt_type}")
    return selected


def counts(rows):
    return {outcome: sum(row["outcome"] == outcome for row in rows) for outcome in OUTCOMES}
