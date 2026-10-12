#!/usr/bin/env python3

import json
import random
from pathlib import Path

SRC = Path(
    "artifacts/safety_finetuning/capability_behavior_B_vs_C.json"
)
BLIND_OUT = Path(
    "artifacts/safety_finetuning/capability_B_vs_C_blind.json"
)
KEY_OUT = Path(
    "artifacts/safety_finetuning/capability_B_vs_C_blind_key.json"
)

src = json.loads(SRC.read_text())

raw = []

for row in src["rows"]:
    for condition in ["B", "C"]:
        raw.append({
            "sample_id": row["id"],
            "condition": condition,
            "prompt": row["prompt"],
            "response": row[condition]["response"],
            "detector_label": row[condition]["detector_label"],
        })

assert len(raw) == 120

random.Random(24680).shuffle(raw)

blind = []
key = {}

for i, row in enumerate(raw, 1):
    blind_id = f"cap-ft-{i:03d}"

    blind.append({
        "item": i,
        "blind_id": blind_id,
        "prompt": row["prompt"],
        "response": row["response"],
        "capability_refusal": None,
    })

    key[blind_id] = {
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "detector_label": row["detector_label"],
    }

BLIND_OUT.write_text(json.dumps(blind, indent=2) + "\n")
KEY_OUT.write_text(json.dumps(key, indent=2) + "\n")

print("Created", len(blind), "blind items")
print("Blind file:", BLIND_OUT)
print("Key file:", KEY_OUT)
