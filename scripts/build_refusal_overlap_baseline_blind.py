#!/usr/bin/env python3

import json
import random
from pathlib import Path

SRC = Path(
    "artifacts/refusal_overlap/"
    "baseline_behavior.json"
)

BLIND = Path(
    "artifacts/refusal_overlap/"
    "baseline_blind.json"
)

KEY = Path(
    "artifacts/refusal_overlap/"
    "baseline_blind_key.json"
)

SEED = 42

d = json.loads(SRC.read_text())

items = []

for checkpoint, rows in d["results"].items():
    for row in rows:
        items.append({
            "checkpoint": checkpoint,
            "id": row["id"],
            "prompt_type": row["prompt_type"],
            "prompt": row["prompt"],
            "response": row["response"],
        })

rng = random.Random(SEED)
rng.shuffle(items)

blind = []
key = {}

for i, x in enumerate(items, start=1):
    blind_id = f"overlap-base-{i:03d}"

    blind.append({
        "blind_id": blind_id,
        "prompt": x["prompt"],
        "response": x["response"],
        "refusal": None,
        "reason_safety": None,
        "reason_capability": None,
        "reason_epistemic": None,
    })

    key[blind_id] = {
        "checkpoint": x["checkpoint"],
        "id": x["id"],
        "prompt_type": x["prompt_type"],
    }

assert len(blind) == 280
assert len(key) == 280

BLIND.parent.mkdir(
    parents=True,
    exist_ok=True,
)

BLIND.write_text(
    json.dumps(blind, indent=2) + "\n"
)

KEY.write_text(
    json.dumps(key, indent=2) + "\n"
)

print("blind items:", len(blind))
print("saved:", BLIND)
print("saved:", KEY)
