#!/usr/bin/env python3

import json
import math
from pathlib import Path

ROOT = Path("artifacts/safety_finetuning")

FILES = {
    "B_S_42": ROOT / "B_safety.json",
    "B_S_43": ROOT / "B_safety_seed43.json",
    "B_E_42": ROOT / "B_epistemic.json",
    "B_E_43": ROOT / "B_epistemic_seed43.json",
    "C_S_42": ROOT / "C_safety.json",
    "C_S_43": ROOT / "C_safety_seed43.json",
    "C_E_42": ROOT / "C_epistemic.json",
    "C_E_43": ROOT / "C_epistemic_seed43.json",
}


def load(path):
    return json.loads(path.read_text())


def vec(d, layer):
    return d["directions"][layer]["direction"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def mean(xs):
    return sum(xs) / len(xs)


def main():
    d = {k: load(v) for k, v in FILES.items()}
    layers = list(d["B_S_42"]["directions"])

    rows = []

    for layer in layers:
        rows.append({
            "layer": layer,
            "B_S_seed_stability": cosine(
                vec(d["B_S_42"], layer),
                vec(d["B_S_43"], layer),
            ),
            "B_E_seed_stability": cosine(
                vec(d["B_E_42"], layer),
                vec(d["B_E_43"], layer),
            ),
            "C_S_seed_stability": cosine(
                vec(d["C_S_42"], layer),
                vec(d["C_S_43"], layer),
            ),
            "C_E_seed_stability": cosine(
                vec(d["C_E_42"], layer),
                vec(d["C_E_43"], layer),
            ),
            "S_B_to_C_seed42": cosine(
                vec(d["B_S_42"], layer),
                vec(d["C_S_42"], layer),
            ),
            "S_B_to_C_seed43": cosine(
                vec(d["B_S_43"], layer),
                vec(d["C_S_43"], layer),
            ),
            "E_B_to_C_seed42": cosine(
                vec(d["B_E_42"], layer),
                vec(d["C_E_42"], layer),
            ),
            "E_B_to_C_seed43": cosine(
                vec(d["B_E_43"], layer),
                vec(d["C_E_43"], layer),
            ),
        })

    keys = [k for k in rows[0] if k != "layer"]

    print("MEAN COSINES")
    for key in keys:
        print(f"{key:24} {mean([r[key] for r in rows]):+.4f}")

    print("\nKEY LAYERS")
    for target in ["model.layers.9", "model.layers.22"]:
        r = next(x for x in rows if x["layer"] == target)
        print("\n" + target)
        for key in keys:
            print(f"  {key:22} {r[key]:+.4f}")

    out = ROOT / "seed_stability_B_vs_C.json"
    out.write_text(json.dumps({"layers": rows}, indent=2) + "\n")
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
