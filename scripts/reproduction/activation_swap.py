"""Summarize the donor-by-downstream factorial with paired family resampling."""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("results", type=Path)
p.add_argument("summary", type=Path)
a = p.parse_args()
data = json.loads(a.results.read_text())
rows = data["rows"]
if data["manifest"]["smoke"] or len(rows) != 120 or len({r["prompt_id"] for r in rows}) != 120:
    p.error("Expected the full 120-prompt result.")
families = sorted({r["family"] for r in rows})
membership = np.array([[r["family"] == f for r in rows] for f in families], dtype=int)
assert len(families) == 30 and (membership.sum(axis=1) == 4).all()
ix = np.random.default_rng(42).integers(0, 30, (20000, 30))

def summary(values):
    boots = (membership @ values)[ix].mean(axis=1) / 4
    return dict(estimate=float(values.mean()), ci95=np.quantile(boots, [.025, .975]).tolist())

cells = {(d, r, e): [] for d in ("original","adapted") for r in ("original","adapted")
         for e in ("none","adaptive","add","subtract")}
for row in rows:
    scores = {(x["donor"], x["receiver"], x["edit"], x["pair"], x["category"]): x["mean_logprob"]
              for x in row["scores"]}
    assert len(scores) == len(row["scores"]) == 64 and np.isfinite(list(scores.values())).all()
    for cell, values in cells.items():
        values.append(np.mean([scores[(*cell, i, "acknowledgment")] -
                               scores[(*cell, i, "instructions")] for i in (0, 1)]))
cells = {k:np.array(v) for k,v in cells.items()}
output = dict(unit="difference in mean log probability per continuation token",
              manifest=data["manifest"], self_swap_max_error=max(r["self_swap_max_error"] for r in rows),
              signed_or_norm_max_error=max(r["signed_or_norm_max_error"] for r in rows),
              cells={}, contrasts={})
for edit in ("adaptive","add","subtract"):
    oo,oa,ao,aa = (cells[d,r,edit]-cells[d,r,"none"] for d,r in
                   (("original","original"),("original","adapted"),
                    ("adapted","original"),("adapted","adapted")))
    output["cells"][edit] = {k:summary(v) for k,v in zip(("OO","OA","AO","AA"),(oo,oa,ao,aa))}
    output["contrasts"][edit] = {
        k:summary(v) for k,v in {
            "native_receiver_gap": aa-oo,
            "donor_main_effect": ((ao-oo)+(aa-oa))/2,
            "downstream_main_effect": ((oa-oo)+(aa-ao))/2,
            "interaction": aa-ao-oa+oo,
        }.items()
    }
with a.summary.open("x") as f:
    json.dump(output,f,indent=2,allow_nan=False)
print(json.dumps(output["contrasts"],indent=2))
