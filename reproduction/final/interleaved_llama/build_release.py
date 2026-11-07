"""Extract verified Llama family counts and reproduce Figure 1, Table 1, and intervals."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import platform
import shutil
from pathlib import Path

import numpy as np

from family_stats import COHORTS, EDITS, FIELDS, STREAMS, aggregate, summarize

HERE = Path(__file__).resolve().parent
RECEIVERS = ("original_instruct", "seed42", "seed43", "seed44")
DISPLAY = ("Original", "Mixed 42", "Mixed 43", "Mixed 44")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def verify(estimates, rows, reference):
    for stream in STREAMS:
        for label in ("original", "fresh", "fresh_minus_original"):
            if not np.allclose(estimates[stream]["interaction"][label],
                               reference["estimates"][stream][label+"_pp"], atol=1e-8, rtol=0):
                raise ValueError(f"Reported interval mismatch: {stream}/{label}")
    caps = {"/".join((c,r,e)):sum(int(x["cap_hits"]) for x in rows
            if (x["cohort"],x["receiver"],x["edit"])==(c,r,e))
            for c in COHORTS for r in RECEIVERS for e in EDITS}
    if caps != reference["cap_hits"]:
        raise ValueError("Cap counts do not match frozen summary")


def build(family_csv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with family_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 720 or set(int(r["n"]) for r in rows) != {4}:
        raise ValueError("Expected 720 four-phrasing family-condition cells")
    estimates, per_family = summarize(rows, RECEIVERS)
    verify(estimates, rows, json.loads((HERE / "reference.json").read_text()))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "family_interactions.csv", per_family, tuple(per_family[0]))
    write_json(out / "interleaved_contrasts.json", estimates)
    table = [
        r"\begin{table}[t]", r"\centering\footnotesize",
        r"\caption{Interleaved Llama acknowledgments out of 120 prompts per receiver and condition. The original and fresh prompt sets share the generation and review workflows.}",
        r"\label{tab:interleaved-counts}", r"\begin{tabular}{llrrr}", r"\toprule",
        r"Families & Receiver & Base & Evidence & Format\\", r"\midrule",
    ]
    for cohort in COHORTS:
        for i, receiver in enumerate(RECEIVERS):
            counts = [sum(int(r["outcome_acknowledgment"]) for r in rows
                          if (r["cohort"],r["receiver"],r["edit"])==(cohort,receiver,edit))
                      for edit in EDITS]
            table.append(" & ".join([cohort.title() if i == 0 else "", DISPLAY[i],
                                     *map(str,counts)]) + r"\\")
        if cohort == "original":
            table.append(r"\midrule")
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "interleaved_counts.tex").write_text("\n".join(table)+"\n")

    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    lo = min(r["interaction_pp"] for r in per_family) - 5
    hi = max(r["interaction_pp"] for r in per_family) + 5
    for cohort, color, line in zip(COHORTS, ("#275786", "#a64720"), ("-", "--")):
        values = sorted(r["interaction_pp"] for r in per_family if r["cohort"] == cohort)
        ax.step([lo,*values,hi], [0,*np.arange(1,len(values)+1)/len(values),1],
                where="post", color=color, ls=line, lw=1.5, label=f"{cohort.title()} families")
    ax.axvline(0, color=".65", lw=.7)
    ax.set(xlabel="Family interaction (percentage points)", ylabel="Fraction of families",
           xlim=(lo,hi), ylim=(-.025,1.025), yticks=[0,.25,.5,.75,1])
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout(pad=.5)
    fig.savefig(out / "interleaved_family_ecdf.pdf", metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)
    return estimates


def extract(reviewed, agreement, out, provenance):
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    reference = json.loads((HERE / "reference.json").read_text())
    if sha(reviewed) != reference["reviewed_sha256"]:
        raise ValueError("Reviewed source does not match the manuscript")
    scored = json.loads(reviewed.read_text())
    summary = json.loads(agreement.read_text())
    if (scored["plan_sha256"] != reference["plan_sha256"] or scored["complete"] is not True
            or summary["unique_pairs"] != 2206 or summary["disagreements"] != 258
            or not summary["all_resolved"]):
        raise ValueError("Unexpected plans or agreement report")
    family = aggregate([(c["cohort"],c["receiver"],c["edit"],c["rows"])
                        for c in scored["conditions"]], RECEIVERS)
    out.mkdir(parents=True)
    write_csv(out / "family_counts.csv", family, FIELDS)
    for name in ("reference.json","family_stats.py","build_release.py","RELEASE.md"):
        shutil.copy2(HERE / name, out / name)
    shutil.copy2(agreement, out / "agreement.json")
    if provenance is not None:
        shutil.copy2(provenance, out / "reviewer_provenance.md")
    else:
        (out / "reviewer_provenance.md").write_text(
            "Exact evaluator model versions, prompts, settings, and adjudication instructions were not supplied to this packer. Record each from contemporaneous logs where available.\n")
    estimates = build(out / "family_counts.csv", out)
    (out / "requirements.txt").write_text(f"numpy=={np.__version__}\nmatplotlib=={version('matplotlib')}\n")
    write_json(out / "manifest.json", dict(
        reviewed_source_sha256=reference["reviewed_sha256"], plan_sha256=reference["plan_sha256"],
        python=platform.python_version(), numpy=np.__version__,
        file_sha256={path.name:sha(path) for path in sorted(out.iterdir()) if path.is_file()}))
    print(json.dumps(estimates["outcome"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    export = commands.add_parser("extract")
    for arg in ("reviewed","agreement","output"):
        export.add_argument(arg, type=Path)
    export.add_argument("--reviewer-provenance", type=Path)
    rerun = commands.add_parser("build")
    rerun.add_argument("family_csv", type=Path)
    rerun.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "extract":
        extract(args.reviewed,args.agreement,args.output,args.reviewer_provenance)
    else:
        build(args.family_csv,args.output)
