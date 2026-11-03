#!/usr/bin/env python3
"""Export blinded capability labels or merge two independent human reviews."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--seed", type=int, default=42)
    merge = sub.add_parser("merge")
    merge.add_argument("--map", type=Path, required=True)
    merge.add_argument("--reviewer-a", type=Path, required=True)
    merge.add_argument("--reviewer-b", type=Path, required=True)
    merge.add_argument("--adjudications", type=Path)
    for p in (export, merge):
        p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from ddmi.review import export_review, merge_review

    try:
        if args.output_dir.resolve().is_relative_to((ROOT / "evidence").resolve()):
            raise ValueError("Review outputs must remain separate from frozen evidence")
        if args.command == "export":
            result = export_review(args.source, args.output_dir, args.seed)
        else:
            result = merge_review(
                args.map, args.reviewer_a, args.reviewer_b, args.output_dir, args.adjudications
            )
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Review error: {exc}\n")


if __name__ == "__main__":
    main()
