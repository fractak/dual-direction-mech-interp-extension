import json
import random
from pathlib import Path

src = json.load(open(
    "artifacts/extension/capability_paired_validation.json"
))

items = []

for row in src["rows"]:
    for condition in [
        "baseline",
        "ablate_safety",
        "ablate_capability",
    ]:
        items.append({
            "hidden_id": f'{row["id"]}::{condition}',
            "prompt": row["prompt"],
            "response": row[condition]["response"],
            "capability_refusal": None,
        })

assert len(items) == 180

random.Random(12345).shuffle(items)

for i, item in enumerate(items, 1):
    item["item"] = i

out = Path(
    "artifacts/extension/capability_full_blind_annotation.json"
)
out.write_text(json.dumps(items, indent=2))

print("Created", len(items), "blind items")
print("Saved:", out)
