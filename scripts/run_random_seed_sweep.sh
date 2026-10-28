#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for seed in 2027 2028 2029 2030 2031; do
  python - "$seed" "$tmp/run.py" <<'PY'
import sys
from pathlib import Path

seed = int(sys.argv[1])
destination = Path(sys.argv[2])
source = Path("run_C_random_control.py").read_text()

old_seed = "SEED = 2026"
old_out = 'OUT = ROOT / "capability_C_random_layer20.json"'

assert old_seed in source and old_out in source

source = source.replace(old_seed, f"SEED = {seed}", 1)
source = source.replace(
    old_out,
    f'OUT = ROOT / "capability_C_random_layer20_seed{seed}.json"',
    1,
)

destination.write_text(source)
PY

  echo "===== Seed $seed ====="
  PYTHONPATH=src python "$tmp/run.py"
done
