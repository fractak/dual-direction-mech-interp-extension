#!/usr/bin/env python3
"""Apply the completed all-other consistency audit and recompute archived-review results."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--figures", action="store_true", help="Save a separately labeled main figure"
    )
    args = parser.parse_args()
    from ddmi.consistency_audit import apply_audit

    try:
        if args.output_dir.resolve().is_relative_to((ROOT / "evidence").resolve()):
            raise ValueError("Audit outputs must remain outside frozen evidence")
        if args.output_dir.exists():
            raise ValueError("Output directory exists; choose a new one")
        reviewed, result = apply_audit(ROOT, args.source, args.map, args.audit)
        args.output_dir.mkdir(parents=True)
        for name, value in (("reviewed_source.json", reviewed), ("results.json", result)):
            (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        capability = result["after_consistency_audit"]
        lines = [
            "# Agent-reviewed archived results",
            "",
            result["provenance"]["interpretation"],
            "",
            "| Checkpoint | Condition | Acknowledgment | Instructions | Unsupported | Other |",
            "|---|---|---:|---:|---:|---:|",
        ]
        from ddmi.paper.evidence import OUTCOMES

        for checkpoint in ("original_instruct", "safety_lora"):
            for condition in ("baseline", "capability_from_safety_lora"):
                counts = capability["conditions"][checkpoint][condition]["counts"]
                line = " | ".join([checkpoint, condition] + [str(counts[k]) for k in OUTCOMES])
                lines.append("| " + line + " |")
        lines += [
            "",
            "Receiver differences in changes (percentage points; paired bootstrap 95% CI):",
        ]
        for outcome, values in capability["difference_in_changes"].items():
            lo, hi = values["bootstrap_95_ci"]
            lines.append(
                f"- {outcome}: {100 * values['change']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
            )
        lines += [
            "",
            "Intervals are exploratory, conditional on these contexts and one checkpoint pair.",
        ]
        (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
        if args.figures:
            import matplotlib.pyplot as plt

            from ddmi.paper.figures import fixed_intervention

            fig, _ = fixed_intervention({"capability": capability})
            try:
                fig.suptitle("Agent-reviewed archived responses: shared activation edit")
                for extension in ("png", "pdf"):
                    fig.savefig(
                        args.output_dir / f"figure1_agent_reviewed.{extension}",
                        dpi=400,
                        bbox_inches="tight",
                    )
            finally:
                plt.close(fig)
        print("\n".join(lines))
        print(json.dumps(result["provenance"], indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Audit error: {exc}\n")


if __name__ == "__main__":
    main()
