#!/usr/bin/env python3

import json
import math
from pathlib import Path

ROOT = Path("artifacts/safety_finetuning")

PATHS = {
    "B_S": ROOT / "B_safety.json",
    "B_E": ROOT / "B_epistemic.json",
    "B_C": Path("artifacts/directions/capability_train_160.json"),
    "C_S": ROOT / "C_safety.json",
    "C_E": ROOT / "C_epistemic.json",
    "C_C": ROOT / "C_capability.json",
}


def load(p):
    return json.loads(p.read_text())


def vec(d, layer):
    return d["directions"][layer]["direction"]


def sep(d, layer):
    return d["directions"][layer]["separability_score"]


def cos(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb)


def mean(xs):
    return sum(xs) / len(xs)


def main():
    d = {k: load(v) for k, v in PATHS.items()}
    layers = list(d["B_C"]["directions"])

    rows = []
    for layer in layers:
        rows.append({
            "layer": layer,
            "cap_B_C": cos(vec(d["B_C"], layer), vec(d["C_C"], layer)),
            "B_S_Cap": cos(vec(d["B_S"], layer), vec(d["B_C"], layer)),
            "C_S_Cap": cos(vec(d["C_S"], layer), vec(d["C_C"], layer)),
            "B_E_Cap": cos(vec(d["B_E"], layer), vec(d["B_C"], layer)),
            "C_E_Cap": cos(vec(d["C_E"], layer), vec(d["C_C"], layer)),
            "sep_B_cap": sep(d["B_C"], layer),
            "sep_C_cap": sep(d["C_C"], layer),
        })

    summary = {
        k: mean([r[k] for r in rows])
        for k in ["cap_B_C", "B_S_Cap", "C_S_Cap", "B_E_Cap", "C_E_Cap"]
    }

    print("MEAN COSINES")
    for k, v in summary.items():
        print(f"{k:18} {v:+.4f}")

    b_peak = d["B_C"]["ranked_layers"][0]["name"]
    c_peak = d["C_C"]["ranked_layers"][0]["name"]

    print("\nPEAKS")
    print("B capability:", b_peak,
          f"sep={sep(d['B_C'], b_peak):.4f}")
    print("C capability:", c_peak,
          f"sep={sep(d['C_C'], c_peak):.4f}")

    print("\nKEY LAYERS")
    for target in sorted(
        {b_peak, c_peak, "model.layers.9", "model.layers.22"},
        key=lambda x: int(x.split(".")[-1]),
    ):
        r = next(x for x in rows if x["layer"] == target)
        print(f"\n{target}")
        print(f"  capability B->C: {r['cap_B_C']:+.4f}")
        print(f"  B S-Cap:         {r['B_S_Cap']:+.4f}")
        print(f"  C S-Cap:         {r['C_S_Cap']:+.4f}")
        print(f"  B E-Cap:         {r['B_E_Cap']:+.4f}")
        print(f"  C E-Cap:         {r['C_E_Cap']:+.4f}")
        print(
            f"  capability sep:  "
            f"{r['sep_B_cap']:.4f} -> {r['sep_C_cap']:.4f}"
        )

    out = ROOT / "capability_geometry_B_vs_C.json"
    out.write_text(
        json.dumps({"summary": summary, "layers": rows}, indent=2) + "\n"
    )
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
