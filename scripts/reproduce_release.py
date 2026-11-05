"""Rebuild the six figures and reported analyses from saved inputs."""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--check-inputs", choices=("cpu", "gpu"))
a = p.parse_args()
inputs = json.loads(Path("reproduction/inputs.json").read_text())

def check(kind):
    missing = [name for name, size in inputs[kind].items()
               if not Path(name).is_file() or Path(name).stat().st_size != size]
    if missing:
        p.error("Missing or size-mismatched inputs:\n" + "\n".join(missing))
    print(f"Verified presence and size of {len(inputs[kind])} {kind} inputs.", flush=True)

def run(script, *args):
    subprocess.run([sys.executable, script, *map(str, args)], check=True,
                   env=dict(os.environ, MPLBACKEND="Agg"))

def compare(expected, observed, location):
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or expected.keys() != observed.keys():
            raise ValueError(f"{location}: keys differ")
        for key, value in expected.items():
            compare(value, observed[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(observed):
            raise ValueError(f"{location}: lengths differ")
        for i, (x, y) in enumerate(zip(expected, observed)):
            compare(x, y, f"{location}[{i}]")
    elif isinstance(expected, float):
        if not math.isclose(expected, observed, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{location}: {expected} != {observed}")
    elif expected != observed:
        raise ValueError(f"{location}: {expected} != {observed}")

def main():
    if a.check_inputs:
        check(a.check_inputs)
        return
    check("cpu")
    out = Path("artifacts/reproduced")
    results = out / "results"
    figures = out / "figures"
    if out.exists():
        p.error(f"{out} already exists. Move it aside before rebuilding.")
    results.mkdir(parents=True)
    figures.mkdir()
    run("scripts/make_paper_figures.py", "--check-paper", "--output-dir", figures)
    run("scripts/analyze_reviewed_run.py", "--plan", "artifacts/plans/measurement_pilot.json",
        "--reviewed-run", "artifacts/review/measurement_pilot_resolved/reviewed_source.json",
        "--output", results / "pilot_results.json")
    for name in ("receiver_replication", "llama_absolute_and_qwen", "length_sensitivity_512",
                 "fresh_absolute", "learned_controls", "reextract", "fixed_sign", "phase_crossover"):
        run(f"scripts/reproduction/{name}.py")
    run("scripts/reproduction/activation_swap.py", "artifacts/runs/activation_swap/full/results.json",
        results / "activation_swap_results.json")
    for produced in sorted(results.glob("*.json")):
        ref = Path("reproduction/reference") / produced.name
        compare(json.loads(ref.read_text()), json.loads(produced.read_text()), produced.name)
    run("scripts/reproduction/plot_learned_directions.py")
    run("scripts/reproduction/plot_measured_dose.py")
    print(f"Verified aggregate results and rebuilt six figures under {out}.")

if __name__ == "__main__":
    main()
