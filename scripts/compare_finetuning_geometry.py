#!/usr/bin/env python3

import json
import math
from pathlib import Path


ROOT = Path("artifacts/safety_finetuning")

FILES = {
    "B_S": ROOT / "B_safety.json",
    "B_E": ROOT / "B_epistemic.json",
    "C_S": ROOT / "C_safety.json",
    "C_E": ROOT / "C_epistemic.json",
}


def load(path):
    return json.loads(path.read_text())


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def vec(artifact, layer):
    return artifact["directions"][layer]["direction"]


def sep(artifact, layer):
    return artifact["directions"][layer]["separability_score"]


def mean(xs):
    return sum(xs) / len(xs)


def main():
    d = {k: load(v) for k, v in FILES.items()}

    layers = list(d["B_S"]["directions"].keys())

    rows = []

    for layer in layers:
        row = {
            "layer": layer,

            # Within-checkpoint safety vs epistemic geometry
            "cos_B_S_E": cosine(
                vec(d["B_S"], layer),
                vec(d["B_E"], layer),
            ),
            "cos_C_S_E": cosine(
                vec(d["C_S"], layer),
                vec(d["C_E"], layer),
            ),

            # Same-behavior direction stability across fine-tuning
            "cos_S_B_C": cosine(
                vec(d["B_S"], layer),
                vec(d["C_S"], layer),
            ),
            "cos_E_B_C": cosine(
                vec(d["B_E"], layer),
                vec(d["C_E"], layer),
            ),

            # Cross-checkpoint / cross-behavior
            "cos_B_S_C_E": cosine(
                vec(d["B_S"], layer),
                vec(d["C_E"], layer),
            ),
            "cos_B_E_C_S": cosine(
                vec(d["B_E"], layer),
                vec(d["C_S"], layer),
            ),

            "sep_B_S": sep(d["B_S"], layer),
            "sep_C_S": sep(d["C_S"], layer),
            "sep_B_E": sep(d["B_E"], layer),
            "sep_C_E": sep(d["C_E"], layer),
        }

        rows.append(row)

    result = {
        "summary": {
            "mean_cos_B_S_E": mean([r["cos_B_S_E"] for r in rows]),
            "mean_cos_C_S_E": mean([r["cos_C_S_E"] for r in rows]),
            "mean_cos_S_B_C": mean([r["cos_S_B_C"] for r in rows]),
            "mean_cos_E_B_C": mean([r["cos_E_B_C"] for r in rows]),
            "mean_cos_B_S_C_E": mean(
                [r["cos_B_S_C_E"] for r in rows]
            ),
            "mean_cos_B_E_C_S": mean(
                [r["cos_B_E_C_S"] for r in rows]
            ),
        },
        "layers": rows,
    }

    out = ROOT / "geometry_B_vs_C.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("MEAN COSINES")
    for k, v in result["summary"].items():
        print(f"{k:24} {v:+.4f}")

    print("\nKEY LAYERS")
    for target in ["model.layers.9", "model.layers.22"]:
        r = next(x for x in rows if x["layer"] == target)

        print("\n" + target)
        print(f"  B S-E:             {r['cos_B_S_E']:+.4f}")
        print(f"  C S-E:             {r['cos_C_S_E']:+.4f}")
        print(f"  S stability B->C:  {r['cos_S_B_C']:+.4f}")
        print(f"  E stability B->C:  {r['cos_E_B_C']:+.4f}")
        print(f"  B-S vs C-E:        {r['cos_B_S_C_E']:+.4f}")
        print(f"  B-E vs C-S:        {r['cos_B_E_C_S']:+.4f}")
        print(
            f"  safety sep:        "
            f"{r['sep_B_S']:.4f} -> {r['sep_C_S']:.4f}"
        )
        print(
            f"  epistemic sep:     "
            f"{r['sep_B_E']:.4f} -> {r['sep_C_E']:.4f}"
        )

    print("\nSaved:", out)


if __name__ == "__main__":
    main()
