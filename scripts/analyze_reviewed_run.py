#!/usr/bin/env python3
"""Compute paired outcomes and displacement diagnostics for a reviewed new run."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reviewed-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from ddmi.prospective_analysis import analyze_reviewed

    try:
        if args.output.resolve().is_relative_to((ROOT / "evidence").resolve()):
            raise ValueError("New analysis cannot overwrite frozen evidence")
        result = analyze_reviewed(
            json.loads(args.plan.read_text()), json.loads(args.reviewed_run.read_text())
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        print(f"Analyzed {len(result['conditions'])} conditions; saved {args.output}")
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Analysis error: {exc}\n")


if __name__ == "__main__":
    main()
