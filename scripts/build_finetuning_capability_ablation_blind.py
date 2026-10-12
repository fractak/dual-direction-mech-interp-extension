#!/usr/bin/env python3

import json
import random
from pathlib import Path

from ddmi.data.loaders import (
    load_capability_prompts,
    prompts_to_text_list,
)

ROOT = Path("artifacts/safety_finetuning")

CAUSAL = ROOT / "capability_cross_checkpoint_ablation.json"
OLD_BLIND = ROOT / "capability_B_vs_C_blind.json"
OLD_KEY = ROOT / "capability_B_vs_C_blind_key.json"

OUT_BLIND = ROOT / "capability_ablation_blind.json"
OUT_KEY = ROOT / "capability_ablation_blind_key.json"

SHUFFLE_SEED = 314159

causal = json.loads(CAUSAL.read_text())
old_blind = json.loads(OLD_BLIND.read_text())
old_key = json.loads(OLD_KEY.read_text())

# ------------------------------------------------------------
# Recover prompt -> stable sample_id from previous blind eval.
# Each prompt appears for both B and C, but sample_id must agree.
# ------------------------------------------------------------
prompt_to_sample = {}

for item in old_blind:
    prompt = item["prompt"]
    sample_id = old_key[item["blind_id"]]["sample_id"]

    if prompt in prompt_to_sample:
        assert prompt_to_sample[prompt] == sample_id
    else:
        prompt_to_sample[prompt] = sample_id

assert len(prompt_to_sample) == 60

# ------------------------------------------------------------
# Reconstruct exact prompt order used by the causal experiment.
# ------------------------------------------------------------
data = load_capability_prompts(
    limit=120,
    seed=42,
    split="expanded_eval",
)

samples = data["incapable"]
prompts = prompts_to_text_list(samples)

assert len(prompts) == 60
assert len(set(prompts)) == 60

for prompt in prompts:
    if prompt not in prompt_to_sample:
        raise RuntimeError(
            "Prompt from causal evaluation missing from old blind set:\n"
            + prompt
        )

# ------------------------------------------------------------
# Build only the four ablation conditions:
# 2 checkpoints x 2 directions x 60 prompts = 240
# ------------------------------------------------------------
conditions = [
    ("B_instruct", "ablate_B_capability"),
    ("B_instruct", "ablate_C_capability"),
    ("C_safety_lora", "ablate_B_capability"),
    ("C_safety_lora", "ablate_C_capability"),
]

records = []

for checkpoint, ablation in conditions:
    responses = causal["results"][checkpoint][ablation]["responses"]

    assert len(responses) == len(prompts)

    for idx, (prompt, response) in enumerate(
        zip(prompts, responses)
    ):
        records.append({
            "prompt": prompt,
            "response": response,
            "checkpoint": checkpoint,
            "ablation": ablation,
            "sample_id": prompt_to_sample[prompt],
            "source_index": idx,
        })

assert len(records) == 240

# ------------------------------------------------------------
# Shuffle before annotation.
# ------------------------------------------------------------
rng = random.Random(SHUFFLE_SEED)
rng.shuffle(records)

blind = []
key = {}

for i, row in enumerate(records, start=1):
    blind_id = f"cap-causal-{i:03d}"

    blind.append({
        "blind_id": blind_id,
        "prompt": row["prompt"],
        "response": row["response"],
        "capability_refusal": None,
    })

    key[blind_id] = {
        "checkpoint": row["checkpoint"],
        "ablation": row["ablation"],
        "sample_id": row["sample_id"],
        "source_index": row["source_index"],
    }

OUT_BLIND.write_text(
    json.dumps(blind, indent=2) + "\n"
)

OUT_KEY.write_text(
    json.dumps(key, indent=2) + "\n"
)

print("blind items:", len(blind))
print("key items:", len(key))
print("unique prompts:", len(set(x["prompt"] for x in blind)))
print("saved:", OUT_BLIND)
print("saved:", OUT_KEY)
