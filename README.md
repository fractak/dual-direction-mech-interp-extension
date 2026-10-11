# Dual-Direction Mechanistic Interpretability

**Disentangling safety-refusal and epistemic-abstention directions in LLM activation space, with an extension to capability refusal.**

Do LLMs refuse harmful requests and abstain on unknown factual questions using the *same* internal mechanism — or are these mechanistically independent behaviors?

> **Extension:** This fork adds **capability refusal** as a third behavior: cases where a model declines because it lacks access, perception, tools, data, or another required capability. See [Capability-refusal extension](#capability-refusal-extension) for the new experiments, results, and reproduction instructions.

## Original paper / upstream work

The upstream work was accepted as an in-person poster at the ICML 2026 Mechanistic Interpretability Workshop.

- OpenReview: https://openreview.net/forum?id=dQJuu7jlZJ
- Preprint: https://zenodo.org/records/20009857

The upstream repository provides a reproducible pipeline to:

1. **Extract** safety-refusal and epistemic-abstention directions via difference-in-means on residual stream activations
2. **Measure** their geometric relationship using cosine similarity and angular separation across layers
3. **Cross-ablate** each direction and measure behavioral effects on safety refusal and epistemic abstention
4. **Probe** whether extracted activations linearly separate the underlying task labels
5. **Quantify** how deployment-time quantization perturbs these directions
6. **Estimate** bootstrap confidence intervals for refusal and abstention rates
7. **Replicate** findings across model families including Llama, Qwen, and Gemma

This fork preserves that pipeline and adds experiments for capability refusal.

## Key Research Question

> When an instruction-tuned LLM says "I can't help with that" for safety reasons versus "I don't know" for epistemic reasons, are these behaviors mediated by the same geometric direction in activation space?

The capability-refusal extension adds a related question:

> When a model says it cannot answer because it lacks access, perception, tools, or another required capability, does that behavior use the same internal mechanism as safety refusal or epistemic abstention?

## Current Findings

The checked-in upstream replication results support a separation story rather than a shared dominant safety/epistemic direction:

- In Llama-3.1-8B-Instruct, the mean safety-vs-epistemic cosine similarity across layers is `0.049`, with a maximum of `0.183` at `model.layers.20`.
- Strong safety-direction ablation produces a large reduction in safety refusal while leaving epistemic abstention comparatively stable.
- A held-out linear probe on epistemic activations reaches `1.00` test accuracy at the selected layer `model.layers.9`.
- Cross-family replication remains qualitatively consistent across Llama, Qwen, and Gemma.

The capability-refusal extension adds evidence that capability refusal is **distinct but partially aligned with safety refusal**, while remaining more clearly separated from epistemic abstention.

## Useful Visuals and Outputs

### Original replication

- Direction geometry figure: [artifacts/figures/fig1_direction_comparison.pdf](artifacts/figures/fig1_direction_comparison.pdf)
- Cross-ablation figure: [artifacts/figures/fig2_cross_ablation.pdf](artifacts/figures/fig2_cross_ablation.pdf)
- Quantization drift figure: [artifacts/figures/fig3_quantization_drift.pdf](artifacts/figures/fig3_quantization_drift.pdf)
- Cross-model figure: [artifacts/figures/fig4_cross_model.pdf](artifacts/figures/fig4_cross_model.pdf)
- Llama direction comparison artifact: [artifacts/directions/meta_llama_Llama_3.1_8B_Instruct_comparison.json](artifacts/directions/meta_llama_Llama_3.1_8B_Instruct_comparison.json)
- Cross-ablation sweep artifact: [artifacts/cross_ablation/meta_llama_Llama_3.1_8B_Instruct_sweep.json](artifacts/cross_ablation/meta_llama_Llama_3.1_8B_Instruct_sweep.json)
- Cross-model combined artifact: [artifacts/cross_model/combined_results.json](artifacts/cross_model/combined_results.json)
- Epistemic probe artifact: [artifacts/probes/meta_llama_Llama_3.1_8B_Instruct_epistemic_probe.json](artifacts/probes/meta_llama_Llama_3.1_8B_Instruct_epistemic_probe.json)

### Capability-refusal extension

- Final summary table: [artifacts/extension/capability_extension_final_table.md](artifacts/extension/capability_extension_final_table.md)
- Final figure (PNG): [artifacts/extension/capability_extension_final_figure.png](artifacts/extension/capability_extension_final_figure.png)
- Final figure (PDF): [artifacts/extension/capability_extension_final_figure.pdf](artifacts/extension/capability_extension_final_figure.pdf)
- Consolidated results: [artifacts/extension/capability_final_results.json](artifacts/extension/capability_final_results.json)
- Human-scored blind annotation: [artifacts/extension/capability_full_blind_annotation.json](artifacts/extension/capability_full_blind_annotation.json)
- Paired statistical results: [artifacts/extension/capability_paired_stats.json](artifacts/extension/capability_paired_stats.json)
- Cross-model capability results: [artifacts/cross_model_capability/combined_results.json](artifacts/cross_model_capability/combined_results.json)

## Setup

```bash
# Clone this fork
git clone https://github.com/fractak/dual-direction-mech-interp-extension.git
cd dual-direction-mech-interp-extension

# Create / activate your environment as desired, then install core dependencies
pip install -e .

# Install with quantization backends
pip install -e ".[quant]"

# Install with plotting support
pip install -e ".[plot]"
```

For Hugging Face model loading in the extension experiments:

```bash
export HF_HUB_DISABLE_XET=1
```

## Original Pipeline

### Experiment 1: Direction Extraction & Geometry

```bash
# Extract safety-refusal direction
python scripts/extract_directions.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --direction-type safety \
    --output artifacts/directions/llama31_8b_safety.json \
    --load-in-4bit \
    --prompt-limit 200

# Extract epistemic-abstention direction
python scripts/extract_directions.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --direction-type epistemic \
    --output artifacts/directions/llama31_8b_epistemic.json \
    --load-in-4bit \
    --prompt-limit 200

# Compare geometry across layers
python scripts/compare_directions.py \
    --safety-artifact artifacts/directions/llama31_8b_safety.json \
    --epistemic-artifact artifacts/directions/llama31_8b_epistemic.json \
    --output artifacts/directions/llama31_8b_comparison.json
```

### Experiment 2: Cross-Ablation

```bash
python scripts/cross_ablation.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --safety-direction artifacts/directions/llama31_8b_safety.json \
    --epistemic-direction artifacts/directions/llama31_8b_epistemic.json \
    --output artifacts/cross_ablation/llama31_8b_results.json \
    --bootstrap-samples 2000 \
    --load-in-4bit
```

### Experiment 2b: Linear Probe Check

```bash
python scripts/linear_probe.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --direction-type epistemic \
    --direction-artifact artifacts/directions/llama31_8b_epistemic.json \
    --output artifacts/probes/llama31_8b_epistemic_probe.json \
    --load-in-4bit
```

This trains a logistic regression probe on held-out activations to test whether the extracted feature is linearly separable even when behavioral ablation effects are weak.

### Experiment 3: Quantization Perturbation

```bash
python scripts/quantization_sweep.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --output artifacts/quantization/llama31_8b_sweep.json
```

### Experiment 4: Cross-Model Replication

```bash
python scripts/cross_model_replication.py \
    --output artifacts/cross_model/ \
    --load-in-4bit
```

The current replication set covers Llama-3.1-8B-Instruct, Qwen3-8B, and the public 4-bit Gemma-2-9B-Instruct mirror `unsloth/gemma-2-9b-it-bnb-4bit`.

### Generate Original Figures

```bash
python scripts/generate_figures.py \
    --artifacts-dir artifacts/ \
    --output-dir artifacts/figures/
```

## Project Structure

```text
src/ddmi/
    editing/       Direction extraction, projection, ablation
    models/        Model loading, hooks, generation
    data/          Dataset loaders, prompt schemas
    evaluation/    Refusal detection, abstention detection, metrics
    utils/         IO, logging, seeds

scripts/           Runnable experiment and reproduction scripts
configs/           YAML experiment configs
data/              Capability-refusal datasets added by this fork
artifacts/         Directions, results, annotations, and figures
paper/             Upstream LaTeX source
tests/             Unit tests
```

## Hardware Requirements

The upstream experiments were designed for **2× NVIDIA T4 (16 GB each)** in Kaggle-style GPU environments.

Example upstream requirements:

- Llama-3.1-8B-Instruct in NF4: approximately 5 GB VRAM
- Direction extraction with 200 prompts: approximately 1 hour per model
- Cross-ablation evaluation: approximately 2 hours per model
- Full original pipeline: approximately 24 GPU-hours

The capability-refusal extension was run separately on an NVIDIA A30; see the extension environment section below.

## Datasets

### Original datasets

| Dataset | Purpose | Source |
|---|---|---|
| `ivnle/advbench_harmful_behaviors` | Harmful prompts, safety-refusal positive class | Hugging Face Hub |
| `tatsu-lab/alpaca` | Benign instructions, safety-refusal negative class | Hugging Face Hub |
| `rajpurkar/squad_v2` | Unanswerable questions, epistemic-abstention positive class | Hugging Face Hub |
| `mandarjoshi/trivia_qa` (`rc.nocontext`) | Answerable questions, epistemic-abstention negative class | Hugging Face Hub |

### Capability-refusal datasets

The extension adds matched capability-refusal datasets under `data/`:

```text
data/capability_pairs.json
data/capability_train_pairs.json
data/capability_eval_pairs.json
data/capability_eval_expanded.json
```

The train/evaluation split is family-disjoint so that capability probe performance is evaluated on unseen task families rather than paraphrases of training tasks.

## Citation

If you use the original Dual-Direction Mechanistic Interpretability work, please cite the upstream work:

```bibtex
@misc{aaliyan2026dualdirection,
    title={Two Refusals or One? Disentangling Safety and Epistemic Abstention Directions in Language Model Activations},
    author={Muhammad Aaliyan},
    year={2026},
    howpublished={ICML 2026 Mechanistic Interpretability Workshop Poster, \url{https://openreview.net/forum?id=dQJuu7jlZJ}},
    doi={10.5281/zenodo.20009857}
}
```

The capability-refusal work in this fork is an experimental extension and is not currently associated with a separate paper or preprint.

## License

MIT

---

# Capability-refusal extension

This project extends the original safety-refusal vs. epistemic-abstention analysis with a third direction: **capability refusal** — responses where the model declines because it lacks access, perception, tools, data, or another required capability.

The central question is whether capability refusal is internally represented by the same mechanism as safety refusal or epistemic abstention, or whether it forms a distinct activation-space direction.

## Experimental Design

Capability directions are extracted from matched incapable/capable prompt pairs using the same difference-in-means approach as the original experiments.

The final primary setup uses:

- Primary model: `meta-llama/Llama-3.1-8B-Instruct`
- Capability training set: 160 prompts from 16 task families
- Capability expanded evaluation set: 60 unseen task families
- Safety direction layer: `model.layers.22`
- Epistemic direction layer: `model.layers.9`
- Capability direction layer: `model.layers.24`
- Ablation: top-1 layer
- Ablation strength: `1.5`
- Seed: `42`

Capability behavior was evaluated using blinded manual annotation of all 180 baseline/intervention responses with a fixed semantic rubric.

A response was counted as a capability refusal when it explicitly communicated that the model could not directly provide the requested information because it lacked the required access, perception, tools, data, or capability.

## Environment

The capability-refusal extension experiments were run on Ubuntu 22.04 using:

- GPU: NVIDIA A30 (24 GB VRAM)
- PyTorch: `2.14.1+cu130`
- CUDA: `13.0`
- Python virtual environment: `.venv`
- Hugging Face model loading with `HF_HUB_DISABLE_XET=1`
- Compute environment: Fabric-managed GPU node

## Main Results

### Capability direction

On Llama-3.1-8B-Instruct:

- Peak capability separability occurs at layer 24: **0.672**
- Family-disjoint probe accuracy at the preselected capability layer 24: **0.925**

The family-disjoint result is reported at the preselected capability layer rather than selecting the best-performing layer after evaluation.

### Causal ablation

For the 60-task expanded capability evaluation set:

**Capability-direction ablation**

- Baseline capability-refusal rate: **0.950**
- Post-ablation capability-refusal rate: **0.817**
- Net reduction: **13.3 percentage points**
- Exact paired p-value: **0.0078**
- Bootstrap 95% CI: **[0.050, 0.217]**

**Safety-direction ablation**

- Baseline capability-refusal rate: **0.950**
- Post-ablation capability-refusal rate: **0.867**
- Net reduction: **8.3 percentage points**
- Exact paired p-value: **0.125**
- Bootstrap 95% CI: **[0.000, 0.167]**

This pattern suggests that capability refusal is causally sensitive to its own extracted direction while showing weaker transfer from safety-direction ablation.

### Cross-model geometry

The capability direction was also extracted in three model families:

| Model | Peak capability layer | Capability separability | Mean safety-capability cosine | Mean epistemic-capability cosine |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 24 | 0.672 | 0.206 | -0.090 |
| Qwen3-8B | 20 | 0.447 | 0.199 | -0.075 |
| Gemma-2-9B | 24 | 0.898 | 0.210 | -0.135 |

Across all three model families, mean safety-capability cosine similarity is approximately **0.20**, while epistemic-capability similarity is near zero or negative.

This supports a **distinct but partially safety-aligned** interpretation of capability refusal.

### Quantization robustness

At the selected capability layer:

| Precision | Cosine vs. FP16 | Separability |
|---|---:|---:|
| FP16 | 1.000 | 0.672 |
| NF4 | 1.000 | 0.672 |
| INT8 | 0.950 | 0.677 |

The capability direction remains geometrically stable under the tested NF4 and INT8 quantization settings.

## Capability Extension Reproduction

The main extension reproduction entry point is:

```bash
export HF_HUB_DISABLE_XET=1
./scripts/reproduce_capability_extension.sh
```

The capability direction can also be extracted directly with:

```bash
python scripts/extract_directions.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --direction-type capability \
    --capability-split train \
    --output artifacts/directions/capability_train_160.json \
    --load-in-4bit
```

### Blind annotation

The randomized blind review set can be regenerated with:

```bash
python scripts/build_full_blind_annotation.py
```

This rebuilds the randomized review material; it does **not** recreate the human judgments.

The completed human-scored annotation artifact used for the reported statistics is stored at:

```text
artifacts/extension/capability_full_blind_annotation.json
```

Paired statistics can be regenerated from the completed annotations with:

```bash
python scripts/capability_paired_stats.py
```

### Final outputs

Consolidated results:

```text
artifacts/extension/capability_final_results.json
```

Final summary table:

```text
artifacts/extension/capability_extension_final_table.md
```

Final figures:

```text
artifacts/extension/capability_extension_final_figure.png
artifacts/extension/capability_extension_final_figure.pdf
```

Cross-model capability results:

```text
artifacts/cross_model_capability/combined_results.json
```

Capability probe:

```text
artifacts/probes/capability_probe.json
```

Quantization results:

```text
artifacts/quantization/capability_sweep.json
```

## Interpretation

Taken together, the current results are most consistent with capability refusal being a **distinct but partially overlapping refusal-related mechanism**.

Capability refusal is:

- strongly linearly separable in activation space,
- causally affected by ablating its extracted direction,
- consistently more geometrically aligned with safety refusal than with epistemic abstention across three model families,
- and robust to the tested quantization settings.

The results do **not** establish that safety refusal and capability refusal are the same mechanism. Instead, they suggest partial alignment alongside meaningful causal and representational separation.
