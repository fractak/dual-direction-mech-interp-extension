# Acknowledgment of access limits before and after LoRA adaptation

This repository supports a directional-intervention study of **Llama 3.1 8B Instruct** before and after an additional safety-oriented LoRA adaptation.

The main question is whether two checkpoints that acknowledge their access limits equally often without an edit can respond differently to the same activation operator.

## Checkpoints

| Identifier used in commands and results | What it is |
|---|---|
| `original_instruct` | `meta-llama/Llama-3.1-8B-Instruct`, without the additional adapter |
| `safety_lora` | The same Instruct checkpoint with the study's additional LoRA adapter |

The original checkpoint is already instruction-tuned. The adapter's training data contains **999 English safety examples and 999 Alpaca instruction examples**. This mixed recipe does not isolate the effect of safety data.

## Reproduce the paper's analysis and figures on CPU

From the repository directory:

```bash
python3 -m venv .venv-figures
source .venv-figures/bin/activate
python -m pip install -r requirements-paper-figures.txt
python scripts/analyze_paper.py
python scripts/make_paper_figures.py --check-paper
```

No model download, GPU, LaTeX installation, or graphical desktop is needed for these commands. Scripts resolve frozen inputs relative to the checkout, so they also work when launched from another directory.

Outputs:

- `artifacts/paper/results.json`: counts, paired statistics, and direction geometry.
- `artifacts/paper/summary.md`: final-outcome counts by checkpoint and condition.
- `artifacts/figures/revised_paper/`: four PNG figures and four vector PDFs, with the numerical results in `data/`.

For one figure, use `python scripts/make_paper_figures.py --figures 1`.
Input checksums are always verified; `--check-paper` remains accepted for existing commands.

The plotting script reads human annotations and archived direction vectors. It does not contain hard-coded response counts or regenerate human judgments.

## Main comparison

Both checkpoints receive the **same capability vector extracted from the safety-LoRA checkpoint**, at layer 20 and strength 1.5. The edit is applied to the layer output during prefill and decoding:

```text
edited_activation = activation - strength * dot(activation, direction) * direction
```

| Checkpoint | Condition | Acknowledgment | Instructions | Unsupported assertion | Other |
|---|---|---:|---:|---:|---:|
| Original Instruct | No edit | 58 | 2 | 0 | 0 |
| Instruct + safety LoRA | No edit | 58 | 2 | 0 | 0 |
| Original Instruct | Same capability edit | 49 | 10 | 1 | 0 |
| Instruct + safety LoRA | Same capability edit | 6 | 8 | 40 | 6 |

Each row contains the same 60 prompt contexts. The difference between acknowledgment changes is -71.7 percentage points; the paired bootstrap interval is [-83.3, -60.0].

“Unsupported” means a specific assertion without evidence supplied in the prompt. It does not establish that the assertion is factually false.

This result shows changed sensitivity to a specified operator in one represented checkpoint pair. The same operator can move activations by different amounts because its effect depends on their projection onto the direction. It does not establish three independent refusal mechanisms or general vulnerability to ordinary prompts.

## Run new experiments on a GPU

The supported runner replaces the separate control and dose scripts:

```bash
# Inspect the main experiment plan; no GPU or adapter needed.
python scripts/run_interventions.py --dry-run

# Execute with your copy of the study adapter.
python -m pip install -e '.[gpu]'
python scripts/run_interventions.py \
  --checkpoint original_instruct safety_lora \
  --adapter-path /path/to/study_adapter \
  --conditions baseline capability \
  --direction-checkpoint safety_lora \
  --layer model.layers.20 \
  --strengths 1.5 \
  --output artifacts/runs/same_operator.json
```

**Adapter weights and the original effective training log are not included.** A newly trained adapter is a new checkpoint, not a recovery of the archived one. New generation outputs remain unannotated and are stored separately from frozen evidence.

See [experiment commands](docs/EXPERIMENTS.md) for controls, dose sweeps, overlap experiments, extraction, and training.

## Repository layout

```text
scripts/                 Six supported command-line entry points
src/ddmi/                Shared editing, model, data, and analysis code
evidence/capability/     1,020 responses with final outcome annotations
evidence/overlap/        1,040 responses with refusal and expressed-reason annotations
evidence/directions/     Three direction collections per checkpoint
evidence/context/        Clearly marked historical summary-only results
evidence/manifest.json  Frozen-input hashes and source provenance
data/                    Capability prompts, overlap prompts, and frozen training data
tests/                   Numerical, provenance, CLI, and intervention tests
docs/                    Protocol details and cleanup record
```

Generated outputs live under `artifacts/` and are ignored by Git. They can be regenerated with the CPU commands above.

## Evidence and interpretation

The final capability rubric assigns exactly one of acknowledgment, instructions, unsupported assertion, or other. The secondary overlap study uses a separate rubric: refusal plus nonexclusive expressed safety, capability, and epistemic reasons. An epistemic phrase can accompany a non-refusal.

Intervals use 20,000 paired prompt resamples with seed 42. Tests are exploratory; retrospective Holm adjustments do not remove condition-selection effects. The 60 capability contexts and 20-context overlap categories are not training replications.

See [provenance and cleanup](docs/PROVENANCE.md) for preserved annotation decisions, limitations, source hashes, and the disposition of historical files.

## Upstream attribution

This project builds on Muhammad Aaliyan's *Two Refusals or One? Disentangling Safety and Epistemic Abstention Directions in Language Model Activations* and retains the `ddmi` package name and upstream authorship metadata.

- Upstream OpenReview reference: https://openreview.net/forum?id=dQJuu7jlZJ
- Upstream archive reference: https://zenodo.org/records/20009857

The old cross-family, probe, quantization, and triple-direction workflows are historical context for this revised study. They are preserved in the pre-cleanup snapshot rather than presented as the current reproduction path.
