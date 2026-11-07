"""Recompute the final paper's portable analyses and optionally saved-input follow-ups."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reproduction/final"
REF = DATA / "reference"
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(DATA / "interleaved_llama"))
sys.path.insert(0, str(REF))
from analyze_replication_positions import read_study, summarize as summarize_positions
from family_stats import aggregate, summarize
from server_inputs import expected
from token_localization import analyze as analyze_localization


def read(path):
    return json.loads(path.read_text())


def run(script, *args, capture=False):
    return subprocess.run([sys.executable, str(script), *map(str, args)], cwd=ROOT,
                          env=dict(os.environ, MPLBACKEND="Agg"), check=True,
                          capture_output=capture, text=True).stdout


def compare(wanted, actual, key):
    if isinstance(wanted, dict):
        if not isinstance(actual, dict) or wanted.keys() != actual.keys():
            raise ValueError(f"Different keys at {key}")
        for k, v in wanted.items():
            compare(v, actual[k], f"{key}/{k}")
    elif isinstance(wanted, list):
        if not isinstance(actual, list) or len(wanted) != len(actual):
            raise ValueError(f"Different array at {key}")
        for i, (left, right) in enumerate(zip(wanted, actual)):
            compare(left, right, f"{key}/{i}")
    elif isinstance(wanted, float):
        if not math.isclose(wanted, actual, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{key}: {wanted} != {actual}")
    elif wanted != actual:
        raise ValueError(f"{key}: {wanted} != {actual}")


def unpack(archive, target, required):
    with zipfile.ZipFile(archive) as z:
        if sorted(z.namelist()) != sorted(required):
            raise ValueError(f"Unexpected members in {archive}")
        target.mkdir()
        for name, digest in required.items():
            sha = hashlib.sha256()
            with z.open(name) as source, (target / name).open("xb") as dest:
                while block := source.read(1024 * 1024):
                    sha.update(block)
                    dest.write(block)
            if digest is not None and sha.hexdigest() != digest:
                raise ValueError(f"Digest mismatch for {name} in {archive}")


def portable(out):
    fig = out / "figures"
    fig.mkdir()
    run(DATA / "interleaved_llama/build_release.py", "build",
        DATA / "interleaved_llama/family_counts.csv", out / "interleaved_llama")
    shutil.copy2(out / "interleaved_llama/interleaved_family_ecdf.pdf", fig)
    # The existing generator also emits a LaTeX table fragment as derived output.
    pair = run(DATA / "paired_model_comparison.py", capture=True)
    compare(read(DATA / "paired_model_comparison.json"), json.loads(pair), "paired comparison")
    (out / "paired_model_comparison.json").write_text(pair)
    qwen = read(REF / "qwen_interleaved_results.json")
    for outcome, saved in qwen["estimates"].items():
        observed, _ = summarize(qwen["family_counts"], ("original_instruct", "safety_lora"), outcome)
        compare(saved, observed, f"Qwen/{outcome}")
    release = DATA / "releases/receiver-robustness-analysis-v1.0.0.zip"
    with zipfile.ZipFile(release) as z:
        hashes = json.loads(z.read("manifest.json"))["files_sha256"]
    unpack(release, out / "robustness_release", {**hashes, "manifest.json": None})
    release_dir = out / "robustness_release"
    compare(read(REF / "receiver_robustness_results.json"), read(release_dir / "results.json"),
            "robustness release reference")
    run(release_dir / "robustness_release.py", "verify",
        release_dir / "reviewed_labels.json", release_dir / "results.json")
    print("Verified interleaved Llama and Qwen, the paired gap, and the held-out and matched-training studies.")


def remove_review(source, dest):
    original = read(source)
    for condition in original["conditions"]:
        for row in condition["rows"]:
            del row["independent_review"]
    dest.write_text(json.dumps(original) + "\n")


def position_sensitivity(inputs, out):
    """Recreate the earlier-label substitution for exact shared text pairs."""
    old = read(inputs / "original.json")
    earlier = {}
    for condition in old["conditions"]:
        for row in condition["rows"]:
            pair = row["prompt"], row["response"]
            label = row["independent_review"]["outcome"]
            if earlier.setdefault(pair, label) != label:
                raise ValueError("Earlier primary study has inconsistent duplicate labels")
    grid, reference, pairs, receivers, modes, _ = read_study(inputs / "position.json", "position_control")
    reference_result = read(REF / "position_control_results.json")
    shared = pairs.keys() & earlier.keys()
    conflicts = sum(pairs[pair]["outcome"] != earlier[pair] for pair in shared)
    if (len(shared), conflicts) != (
            reference_result["prior_primary_overlap"]["exact_prompt_response_pairs"],
            reference_result["prior_primary_overlap"]["conflicting_outcomes"]):
        raise ValueError("Earlier primary overlap does not match the paper")
    changed = 0
    for rows in grid.values():
        for row in rows.values():
            pair = row["prompt"], row["response"]
            previous = row["independent_review"]["outcome"]
            label = earlier.get(pair, previous)
            changed += label != previous
            row["independent_review"]["outcome"] = label
    result = {label: summarize_positions(grid, reference, receivers, modes, False, "outcome", label)
              for label in ("acknowledgment", "unsupported")}
    result["changed_rows"] = changed
    saved = reference_result["earlier_primary_label_sensitivity"]
    compare({key: saved[key] for key in result}, result, "position label substitution")
    (out / "position_label_substitution.json").write_text(json.dumps(result, indent=2) + "\n")


def reviewed(out, archive):
    inputs = out / "reviewed_inputs"
    unpack(archive, inputs, {**expected(), "token_localization.json": None})

    def recompute(script, arguments, reference, output_name, subset=False):
        target = out / output_name
        run(script, *arguments, target)
        saved, actual = read(reference), read(target)
        compare({key: saved[key] for key in actual} if subset else saved, actual, output_name)

    for study, source in (("family_replication", "fresh.json"), ("position_control", "position.json")):
        recompute(REF / "analyze_replication_positions.py", (study, inputs / source, "--output"),
                  REF / f"{study}_results.json", f"{study}_results.json", subset=True)
    position_sensitivity(inputs, out)
    run(DATA / "attenuation/analyze_attenuation.py",
        inputs / "original.json", inputs / "fresh.json", out / "attenuation")
    compare(read(DATA / "attenuation/attenuation_results.json"),
            read(out / "attenuation/attenuation_results.json"), "attenuation")
    for source, script, reference, name in (
        ("joint", DATA / "joint_review.py", REF / "joint_relabel_results.json", "joint_relabel_results.json"),
        ("human_earlier", DATA / "human_validation/human_validation.py",
         DATA / "human_validation/results.json", "earlier_human_audit.json"),
    ):
        raw = inputs / f"{source}_unreviewed.json"
        remove_review(inputs / f"{source}.json", raw)
        recompute(script, ("analyze", raw, inputs / f"{source}.json"), reference, name)
    recompute(DATA / "robustness_release.py",
              ("audit-analyze", inputs / "audit_model_labels.json", inputs / "human_robustness.json"),
              REF / "receiver_robustness_human_audit.json", "receiver_robustness_human_audit.json")
    qwen = read(inputs / "qwen.json")
    cells = [(c["cohort"], c["receiver"], c["edit"], c["rows"]) for c in qwen["conditions"]]
    receivers = ("original_instruct", "safety_lora")
    uncapped, _ = summarize(aggregate(cells, receivers, uncapped=True), receivers)
    compare(read(REF / "qwen_interleaved_results.json")["uncapped_acknowledgment"],
            uncapped, "Qwen uncapped")
    (out / "qwen_uncapped_acknowledgment.json").write_text(json.dumps(uncapped, indent=2) + "\n")
    localized = analyze_localization(ROOT, inputs / "token_localization.json")
    (out / "token_localization_results.json").write_text(json.dumps(localized, indent=2) + "\n")
    print("Verified the additional reviewed-source analyses.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/analysis/reproduced_final")
    parser.add_argument("--legacy", action="store_true", help="Also run the earlier CPU rebuild and regenerate its six figures")
    parser.add_argument("--reviewed-archive", type=Path, help="Also rerun analyses that require late reviewed exports")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.legacy and (ROOT / "artifacts/reproduced").exists():
        parser.error("The legacy output artifacts/reproduced already exists. Move it aside first.")
    out.mkdir(parents=True, exist_ok=False)
    portable(out)
    if args.legacy:
        run(ROOT / "scripts/reproduce_release.py")
        for reference in sorted((DATA / "figures").glob("*.pdf")):
            if reference.name != "interleaved_family_ecdf.pdf":
                shutil.copy2(ROOT / "artifacts/reproduced/figures" / reference.name, out / "figures")
    if args.reviewed_archive:
        reviewed(out, args.reviewed_archive.resolve())
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
