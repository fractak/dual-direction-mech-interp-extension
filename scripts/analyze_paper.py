#!/usr/bin/env python3
"""Recompute paper counts, paired statistics, and geometry on CPU."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/paper")
    args = parser.parse_args()
    from ddmi.paper.analysis import analyze, save_results

    try:
        result = analyze(args.repo_root)
        save_results(result, args.output_dir)
    except (ValueError, KeyError) as exc:
        parser.exit(2, f"Evidence error: {exc}\n")
    print(f"Verified {result['provenance']['frozen_files_verified']} frozen input files.")
    for checkpoint, values in result["capability"]["same_operator"].items():
        print(
            f"{checkpoint}: acknowledgment "
            f"{values['acknowledgment']['before_count']}/60 -> "
            f"{values['acknowledgment']['after_count']}/60"
        )
    print(f"Saved: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
