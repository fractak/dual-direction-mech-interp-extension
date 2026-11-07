"""Compare interleaved acknowledgment shifts with a paired family bootstrap."""
import csv
import json
from pathlib import Path

import numpy as np

from interleaved_llama.family_stats import COHORTS, summarize

ROOT = Path(__file__).resolve().parent
DRAWS = 20_000
SEED = 42


def family_interactions(rows, receivers):
    _, families = summarize(rows, receivers)
    return {
        (row["cohort"], row["family"]): (row["n"], row["interaction_pp"])
        for row in families
    }


def main():
    llama_pair = ("original_instruct", "seed42")
    qwen_pair = ("original_instruct", "safety_lora")
    with (ROOT / "interleaved_llama/family_counts.csv").open(newline="") as source:
        llama_rows = [row for row in csv.DictReader(source) if row["receiver"] in llama_pair]
    with (ROOT / "qwen_family_counts.csv").open(newline="") as source:
        qwen_rows = list(csv.DictReader(source))
    llama = family_interactions(llama_rows, llama_pair)
    qwen = family_interactions(qwen_rows, qwen_pair)
    if llama.keys() != qwen.keys():
        raise ValueError("Llama and Qwen family IDs differ")

    rng = np.random.default_rng(SEED)
    means = {}
    samples = {}
    for cohort in COHORTS:
        keys = sorted(key for key in llama if key[0] == cohort)
        if len(keys) != 30 or any(llama[key][0] != 4 or qwen[key][0] != 4 for key in keys):
            raise ValueError(f"{cohort}: expected 30 shared families of four prompts each")
        values = np.array([[llama[key][1], qwen[key][1]] for key in keys])
        indices = rng.integers(0, len(keys), (DRAWS, len(keys)))
        means[cohort] = values.mean(axis=0)
        samples[cohort] = values[indices].mean(axis=1)

    shift = means["fresh"] - means["original"]
    shift_draws = samples["fresh"] - samples["original"]

    def interval(point, draws):
        return [float(point), *np.quantile(draws, (0.025, 0.975)).tolist()]

    report = {
        "format": "[estimate, lower, upper], percentage points",
        "bootstrap": {"draws": DRAWS, "seed": SEED,
                      "unit": "shared prompt family, sampled separately within each bank"},
    }
    for col, model in enumerate(("llama_seed42", "qwen_seed42")):
        report[model] = {
            cohort: interval(means[cohort][col], samples[cohort][:, col]) for cohort in COHORTS
        }
        report[model]["fresh_minus_original"] = interval(shift[col], shift_draws[:, col])
    report["shift_gap_llama_minus_qwen"] = interval(
        shift[0] - shift[1], shift_draws[:, 0] - shift_draws[:, 1]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
