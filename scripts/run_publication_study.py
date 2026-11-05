#!/usr/bin/env python3
"""Freeze a prospective experiment plan or execute one with measured perturbations."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser(
        "plan", help="CPU only; freezes prompts, vectors, settings and condition grid"
    )
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--prompt-file", type=Path)
    plan.add_argument("--direction-root", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="GPU inference; requires model access and saved adapter")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--adapter-path", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--resume", action="store_true", help="Recover and skip saved responses")
    run.add_argument(
        "--check-only",
        action="store_true",
        help="With --resume, validate progress without generating",
    )
    args = parser.parse_args()
    from ddmi.publication import create_plan, execute_plan

    try:
        target = args.output if args.command == "plan" else args.output_dir
        if target.resolve().is_relative_to((ROOT / "evidence").resolve()):
            raise ValueError("Prospective outputs cannot replace frozen evidence")
        if args.command == "plan":
            value = create_plan(
                ROOT, json.loads(args.config.read_text()), args.prompt_file, args.direction_root
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x") as handle:
                json.dump(value, handle, indent=2, allow_nan=False)
            print(
                f"Frozen {len(value['payload']['conditions'])} conditions; "
                f"{value['payload']['planned_responses']} planned responses; "
                f"plan SHA-256 {value['plan_sha256']}"
            )
        else:
            execute_plan(
                json.loads(args.plan.read_text()),
                args.output_dir,
                args.adapter_path,
                resume=args.resume,
                check_only=args.check_only,
            )
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Study error: {exc}\n")


if __name__ == "__main__":
    main()
