"""Move only inventoried smoke runs and recovery copies outside the checkout."""
import argparse
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("archive", type=Path)
p.add_argument("--apply", action="store_true")
args = p.parse_args()
archive = args.archive.resolve()
if archive.is_relative_to(root) or root.is_relative_to(archive):
    p.error("Choose an archive directory outside the checkout.")
if archive.exists():
    p.error("Choose a new archive directory.")
paths = json.loads((root / "reproduction/cleanup_candidates.json").read_text())
sources = []
for rel in paths:
    source = root / rel
    if not source.resolve().is_relative_to(root / "artifacts/runs"):
        p.error(f"Unexpected candidate: {source}")
    if source.is_dir():
        sources.append((source, archive / rel))
for source, target in sources:
    print(f"{source} -> {target}")
if args.apply:
    for source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    print(f"Archived {len(sources)} directories. Paths can be restored from {archive}.")
else:
    print(f"Preview of {len(sources)} moves.")
