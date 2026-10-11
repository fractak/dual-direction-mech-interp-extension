#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_XET=1

MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo "=== Capability direction ==="

python scripts/extract_directions.py \
  --model-id "$MODEL" \
  --direction-type capability \
  --capability-split train \
  --prompt-limit 160 \
  --seed 42 \
  --output artifacts/directions/capability_train_160.json

echo
echo "=== Capability probe ==="

python scripts/capability_probe.py \
  --model-id "$MODEL" \
  --direction-artifact artifacts/directions/capability_train_160.json \
  --output artifacts/probes/capability_probe.json \
  --seed 42

echo
echo "=== Cross-model geometry ==="

python scripts/cross_model_capability_replication.py \
  --output-dir artifacts/cross_model_capability \
  --load-in-4bit \
  --prompt-limit 200 \
  --capability-prompt-limit 160 \
  --capability-split train \
  --seed 42

echo
echo "=== Quantization robustness ==="

python scripts/capability_quantization_sweep.py \
  --model-id "$MODEL" \
  --fp16-capability-artifact artifacts/directions/capability_train_160.json \
  --target-layer model.layers.24 \
  --prompt-limit 160 \
  --capability-split train \
  --output artifacts/quantization/capability_sweep.json \
  --seed 42

echo
echo "=== Paired human-label statistics ==="

python scripts/capability_paired_stats.py

echo
echo "=== Final result bundle ==="

python scripts/build_capability_final_results.py

echo
echo "=== Final table and figure ==="

python scripts/make_capability_extension_outputs.py

echo
echo "Capability-refusal extension reproduced successfully."
