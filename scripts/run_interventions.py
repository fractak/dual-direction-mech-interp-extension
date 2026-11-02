#!/usr/bin/env python3
"""Run new checkpoint interventions; use --dry-run to inspect without a GPU."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        choices=("original_instruct", "safety_lora"),
        default=["original_instruct", "safety_lora"],
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="Required for inference with safety_lora; weights are not in this repository",
    )
    parser.add_argument("--study", choices=("capability", "overlap"), default="capability")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("baseline", "capability", "safety", "epistemic", "safety_orthogonal", "random"),
        default=["baseline", "capability"],
    )
    parser.add_argument(
        "--direction-checkpoint",
        choices=("original_instruct", "safety_lora"),
        default="safety_lora",
    )
    parser.add_argument("--layer", default="model.layers.20")
    parser.add_argument("--strengths", type=float, nargs="+", default=[1.5])
    parser.add_argument("--random-seeds", type=int, nargs="+", default=list(range(2026, 2032)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/interventions.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from ddmi.experiments import build_plan, execute

    if args.study == "overlap" and (
        args.conditions != ["baseline", "capability"]
        or args.direction_checkpoint != "safety_lora"
        or args.layer != "model.layers.20"
    ):
        parser.error(
            "--conditions, --direction-checkpoint and --layer are capability-study options; "
            "overlap uses the documented category-specific plan"
        )
    if args.output.resolve().is_relative_to((ROOT / "evidence").resolve()):
        parser.error("New runs cannot overwrite frozen evidence")
    try:
        plan = build_plan(
            ROOT,
            args.checkpoint,
            args.conditions,
            args.strengths,
            args.random_seeds,
            args.layer,
            args.direction_checkpoint,
            args.study,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {"study": args.study, "adapter_required_for_safety_lora": True, "plan": plan},
                    indent=2,
                )
            )
            return
        if args.output.exists():
            parser.error(f"Output already exists; choose a new --output: {args.output}")
        if "safety_lora" in args.checkpoint and (
            not args.adapter_path or not (args.adapter_path / "adapter_config.json").is_file()
        ):
            parser.error("--adapter-path must contain adapter_config.json for safety_lora")
        execute(plan, ROOT, args.output, args.adapter_path, args.seed, args.study)
    except (ValueError, KeyError) as exc:
        parser.exit(2, f"Experiment error: {exc}\n")


if __name__ == "__main__":
    main()
