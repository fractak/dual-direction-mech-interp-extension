#!/usr/bin/env python3
"""Regenerate four paper figures from verified evidence (PNG and vector PDF)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/figures/revised_paper")
    parser.add_argument(
        "--figures", type=int, nargs="+", choices=(1, 2, 3, 4), default=(1, 2, 3, 4)
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--check-paper",
        action="store_true",
        help="Explicit verification flag; input hashes are always checked",
    )
    args = parser.parse_args()
    if not 72 <= args.dpi <= 2400:
        parser.error("--dpi must be between 72 and 2400")
    from ddmi.paper.analysis import analyze, save_results
    from ddmi.paper.figures import plot_figures

    try:
        result = analyze(args.repo_root)
        save_results(result, args.output_dir / "data")
        for path in plot_figures(result, args.output_dir, args.figures, args.dpi):
            print(path)
    except (ValueError, KeyError) as exc:
        parser.exit(2, f"Evidence error: {exc}\n")


if __name__ == "__main__":
    main()
