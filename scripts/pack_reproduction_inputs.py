"""Export inventoried saved inputs and a checksum list for a clean checkout."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path, help="New .tar.gz outside the checkout")
p.add_argument("--include-gpu", action="store_true")
a = p.parse_args()
out = a.output.resolve()
manifest = out.with_name(out.name + ".sha256")
if out.is_relative_to(root) or out.exists() or manifest.exists():
    p.error("Choose a new output archive outside the checkout.")
inventory = json.loads((root / "reproduction/inputs.json").read_text())
files = inventory["cpu"] | (inventory["gpu"] if a.include_gpu else {})
for name, size in files.items():
    path = root / name
    if not path.is_file() or path.stat().st_size != size:
        p.error(f"Missing or size-mismatched input: {name}")
with out.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz", compresslevel=3) as archive:
    for name in sorted(files):
        print(name, flush=True)
        archive.add(root / name, arcname=name, recursive=False)
digest = hashlib.sha256()
with out.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)
with manifest.open("x") as f:
    f.write(f"{digest.hexdigest()}  {out.name}\n")
print(out, manifest)
