# Supported experiment commands

All examples assume a checkout, `python -m pip install -e '.[gpu]'`, and the necessary model access. Inspect `--help` before running. The GPU workflow has been tested with synthetic models on CPU; a full model or training rerun was not performed in the recorded validation.

## Fixed operator

```bash
python scripts/run_interventions.py \
  --checkpoint original_instruct safety_lora \
  --adapter-path /path/to/study_adapter \
  --conditions baseline capability \
  --direction-checkpoint safety_lora \
  --layer model.layers.20 --strengths 1.5 \
  --output artifacts/runs/same_operator.json
```

The receiver checkpoint and the checkpoint supplying the vector are separate arguments. The same archived vector is used for both receivers. Model loading uses NF4 with float16 compute and eager attention; generation is greedy with at most 128 new tokens and 512 input tokens. The LoRA remains injected and unmerged.

At strength 1, the projected component is removed. At 1.5, its sign is reversed and its magnitude halved. Equal layer, strength, and vector do not guarantee equal realized displacement.

## Controls and doses

```bash
python scripts/run_interventions.py \
  --checkpoint safety_lora --adapter-path /path/to/study_adapter \
  --conditions baseline safety epistemic safety_orthogonal random \
  --strengths 1.5 --random-seeds 2026 2027 2028 2029 2030 2031 \
  --output artifacts/runs/controls.json

python scripts/run_interventions.py \
  --checkpoint safety_lora --adapter-path /path/to/study_adapter \
  --conditions capability \
  --strengths 0 0.41651599666722117 0.7904264640058756 1.5 \
  --output artifacts/runs/capability_doses.json
```

`--conditions baseline` emits one unedited condition regardless of the strengths list. `safety_orthogonal` removes safety's component along capability and renormalizes the remaining vector. Random directions are independent seeded unit vectors in the same hidden space. These controls do not match the actual activation displacement.

## Secondary overlap study

```bash
python scripts/run_interventions.py --study overlap --dry-run
python scripts/run_interventions.py --study overlap \
  --adapter-path /path/to/study_adapter --strengths 1.5 \
  --output artifacts/runs/overlap.json
```

The seven prompt categories each contain 20 contexts. There are 26 condition groups per checkpoint (1,040 total responses for both checkpoints at one strength). Safety uses layer 22, epistemic uses layer 9, and capability uses layer 24 for Original Instruct or 20 for Safety LoRA. Multiple edits run in the stated order; their effects are not assumed independent. Overlap uses checkpoint-specific vectors rather than the main comparison's shared operator. Capability-only flags `--conditions`, `--layer`, and `--direction-checkpoint` do not configure this protocol.

## Extract new directions

```bash
python scripts/extract_directions.py \
  --checkpoint original_instruct --direction-type capability \
  --capability-split train --prompt-limit 160 --load-in-4bit \
  --output artifacts/directions/original_instruct_capability.json

python scripts/extract_directions.py \
  --checkpoint safety_lora --adapter-path /path/to/study_adapter \
  --direction-type capability --capability-split train \
  --prompt-limit 160 --load-in-4bit \
  --output artifacts/directions/safety_lora_capability.json
```

Use `--direction-type safety` or `epistemic` with `--prompt-limit 200` for those contrasts. Directions are normalized differences of mean last-input-token activations. Layer ranking uses centroid distance divided by one plus the two average within-group radii. New extractions record the capability split and exact prompt hashes. The historical capability extraction split is not recovered from those additions.

## Training

The checked-in `data/safety_finetuning/train_sft.json` is the historical 1,998-example dataset. Do not replace it when downloading newer upstream data.

```bash
python -m pip install -e '.[gpu,train]'
python scripts/train_safety_lora.py \
  --dataset data/safety_finetuning/train_sft.json \
  --output-dir artifacts/adapters/new_safety_lora
```

The preserved recipe uses rank 16, alpha 32, dropout 0.05, attention and MLP targets, NF4 double quantization with bfloat16 compute, completion-only loss, sequence length 1,024, batch size 1, and accumulation 16. The script defaults to one epoch, learning rate 1e-4, and seed 42. These defaults are not a recovered log of the archived checkpoint's effective run.

To build a **new** training sample from current dataset releases:

```bash
python scripts/prepare_training_data.py \
  --seed 67 --output artifacts/datasets/new_train_sft.json
```

The new preparer reads AdvBench's `instruction` field before alternative schemas and refuses an empty overlap guard. The historical preparer missed that field; its saved `advbench_exact_overlap_excluded` flag is a historical claim, not proof of a functioning guard. Preserve the frozen dataset and report new data separately.

## Human review

New runs contain `annotation: null`. The runner does not apply a regex detector or inherit paper labels. Human judgments require the appropriate rubric and prompt pairing. `analyze_paper.py` evaluates the frozen evidence only; incorporating a new reviewed run requires an explicitly versioned evidence update. New run files cannot overwrite `evidence/`, and existing output paths are refused.
