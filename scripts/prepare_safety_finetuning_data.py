#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=67)
    p.add_argument(
        "--output",
        default="data/safety_finetuning/train_sft.json",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)

    adv = load_dataset(
        "ivnle/advbench_harmful_behaviors",
        split="train",
    )

    adv_prompts = set()
    for row in adv:
        text = row.get("goal") or row.get("prompt") or row.get("text")
        if text:
            adv_prompts.add(norm(text))

    safety_ds = load_dataset(
        "iNLP-Lab/multilingual-safety",
        "en",
        split="train",
    )

    safety = []
    seen = set()

    for idx, row in enumerate(safety_ds):
        prompt = row["prompt"].strip()
        response = row["output"].strip()
        key = norm(prompt)

        if not prompt or not response:
            continue
        if key in adv_prompts or key in seen:
            continue

        seen.add(key)

        safety.append({
            "kind": "safety",
            "source": "iNLP-Lab/multilingual-safety:en",
            "source_index": idx,
            "prompt": prompt,
            "response": response,
            "prompt_sha256": sha(prompt),
            "response_sha256": sha(response),
        })

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")

    benign = []

    for idx, row in enumerate(alpaca):
        instruction = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        response = row.get("output", "").strip()

        if not instruction or not response:
            continue

        prompt = instruction
        if inp:
            prompt += "\n\n" + inp

        key = norm(prompt)

        if key in adv_prompts or key in seen:
            continue

        seen.add(key)

        benign.append({
            "kind": "benign",
            "source": "tatsu-lab/alpaca",
            "source_index": idx,
            "prompt": prompt,
            "response": response,
            "prompt_sha256": sha(prompt),
            "response_sha256": sha(response),
        })

    rng.shuffle(benign)
    benign = benign[:len(safety)]

    rows = safety + benign
    rng.shuffle(rows)

    result = {
        "config": {
            "seed": args.seed,
            "n_safety": len(safety),
            "n_benign": len(benign),
            "safety_source": "iNLP-Lab/multilingual-safety:en",
            "advbench_exact_overlap_excluded": True,
            "normalized_prompt_deduplication": True,
        },
        "rows": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("safety:", len(safety))
    print("benign:", len(benign))
    print("total:", len(rows))
    print("saved:", out)


if __name__ == "__main__":
    main()
