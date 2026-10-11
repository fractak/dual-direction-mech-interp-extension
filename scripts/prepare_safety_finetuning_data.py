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
    p.add_argument("--n-safety", type=int, default=2000)
    p.add_argument("--n-benign", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        default="data/safety_finetuning/train_sft.json",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)

    print("Loading AdvBench exclusion set...")
    adv = load_dataset(
        "ivnle/advbench_harmful_behaviors",
        split="train",
    )

    adv_prompts = set()
    for row in adv:
        text = row.get("goal") or row.get("prompt") or row.get("text")
        if text:
            adv_prompts.add(norm(text))

    print("Loading PKU-SafeRLHF...")
    pku = load_dataset(
        "PKU-Alignment/PKU-SafeRLHF",
        "default",
        split="train",
    )

    safety = []
    seen_safety = set()

    for idx, row in enumerate(pku):
        prompt = row["prompt"]

        prompt_key = norm(prompt) if prompt else ""
        if not prompt_key or prompt_key in adv_prompts:
            continue
        if prompt_key in seen_safety:
            continue

        safe0 = bool(row["is_response_0_safe"])
        safe1 = bool(row["is_response_1_safe"])

        # Keep only clean safety-preference pairs:
        # exactly one response safe and one unsafe.
        if safe0 == safe1:
            continue

        safer = int(row["safer_response_id"])

        if safer == 0 and safe0:
            response = row["response_0"]
            response_hash = row.get("response_0_sha256") or sha(response)
        elif safer == 1 and safe1:
            response = row["response_1"]
            response_hash = row.get("response_1_sha256") or sha(response)
        else:
            continue

        seen_safety.add(prompt_key)

        safety.append({
            "kind": "safety",
            "source": "PKU-Alignment/PKU-SafeRLHF",
            "source_index": idx,
            "prompt": prompt,
            "response": response,
            "prompt_sha256": sha(prompt),
            "response_sha256": response_hash,
        })

    if len(safety) < args.n_safety:
        raise RuntimeError(
            f"Only {len(safety)} eligible safety examples found."
        )

    rng.shuffle(safety)
    safety = safety[:args.n_safety]

    print("Loading Alpaca benign examples...")
    alpaca = load_dataset(
        "tatsu-lab/alpaca",
        split="train",
    )

    benign = []
    seen_benign = {norm(r["prompt"]) for r in safety}

    for idx, row in enumerate(alpaca):
        instruction = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        response = row.get("output", "").strip()

        if not instruction or not response:
            continue

        prompt = instruction
        if inp:
            prompt += "\n\n" + inp

        prompt_key = norm(prompt)
        if prompt_key in adv_prompts:
            continue
        if prompt_key in seen_benign:
            continue

        seen_benign.add(prompt_key)

        benign.append({
            "kind": "benign",
            "source": "tatsu-lab/alpaca",
            "source_index": idx,
            "prompt": prompt,
            "response": response,
            "prompt_sha256": sha(prompt),
            "response_sha256": sha(response),
        })

    if len(benign) < args.n_benign:
        raise RuntimeError(
            f"Only {len(benign)} eligible benign examples found."
        )

    rng.shuffle(benign)
    benign = benign[:args.n_benign]

    rows = safety + benign
    rng.shuffle(rows)

    result = {
        "config": {
            "seed": args.seed,
            "n_safety": len(safety),
            "n_benign": len(benign),
            "advbench_exact_overlap_excluded": True,
            "safety_filter": "exactly_one_response_safe",
        },
        "rows": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"Safety examples: {len(safety)}")
    print(f"Benign examples: {len(benign)}")
    print(f"Total: {len(rows)}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
