# Safety Fine-Tuning Extension Design

## Research question

Does safety fine-tuning create a new refusal representation, or does it transform or recruit a pre-existing epistemic-abstention representation?

## Experiment matrix

### Checkpoints

Checkpoint A: base model

Checkpoint B: instruction-tuned model

Checkpoint C: safety/alignment-tuned model

### Within each checkpoint

For each checkpoint:

- Extract safety-refusal direction
- Extract epistemic-abstention direction
- Compute layerwise separability
- Compute within-checkpoint safety–epistemic cosine similarity
- Evaluate safety refusal behavior
- Evaluate epistemic abstention behavior

### Across checkpoints

Compare:

- cosine(S_A, S_B)
- cosine(S_B, S_C)
- cosine(S_A, S_C)

- cosine(E_A, E_B)
- cosine(E_B, E_C)
- cosine(E_A, E_C)

Also test:

- Cross-checkpoint safety-direction ablation transfer
- Cross-checkpoint epistemic-direction ablation transfer
- Cross-checkpoint probe transfer

## Main hypotheses

1. Safety fine-tuning may rotate or strengthen the safety-refusal direction.
2. The epistemic-abstention direction may remain comparatively stable.
3. If safety training recruits an existing abstention mechanism, safety and epistemic directions should become more aligned after tuning.
4. If safety training creates a distinct mechanism, the safety direction should diverge from the epistemic direction while becoming more behaviorally causal for refusal.
5. Cross-checkpoint transfer can distinguish representation reuse from checkpoint-specific mechanisms.

## Controls

Keep constant across checkpoints:

- Prompt datasets
- Prompt ordering
- Prompt limits
- Random seeds
- Direction extraction method
- Layer indexing
- Evaluation metrics
- Generation settings

Where architectures differ, compare normalized layer depth in addition to absolute layer index.

## Selected checkpoint design

Primary model family: Llama 3.1 8B.

- Checkpoint A — Base:
  `meta-llama/Llama-3.1-8B`

- Checkpoint B — Instruct:
  `meta-llama/Llama-3.1-8B-Instruct`

- Checkpoint C — Controlled safety fine-tune:
  a LoRA/QLoRA adapter trained by this project on top of
  `meta-llama/Llama-3.1-8B-Instruct`.

The main causal comparison is B -> C because the added training intervention
is known and controlled.

A -> B is treated as a broader post-training comparison because the Instruct
checkpoint includes instruction tuning and safety-related alignment rather
than an isolated safety-only intervention.

## Controlled safety fine-tuning data design

Checkpoint C must be trained on safety examples that are disjoint from the
held-out safety evaluation prompts.

### Training data

Target scale:

- 1,000–3,000 harmful instruction examples
- refusal-style assistant targets
- matched benign instruction examples to reduce generalized over-refusal

The safety fine-tuning dataset must not contain the held-out safety evaluation
prompts used for checkpoint comparison.

### Evaluation data

Use the same evaluation prompts for checkpoints A, B, and C:

- 200 held-out harmful prompts
- 200 benign control prompts
- the same epistemic-abstention evaluation set used in the original replication

Keep constant across checkpoints:

- prompt order
- random seed
- generation settings
- refusal / abstention detectors
- direction extraction procedure

### Primary causal comparison

The strongest causal comparison is B -> C.

If the safety direction changes after C while the epistemic direction remains
stable, that supports a safety-specific representational change.

If the safety direction becomes more aligned with the pre-existing epistemic
direction, that supports recruitment of an existing abstention-related
representation.

## Selected safety fine-tuning dataset

Checkpoint C will use:

`PKU-Alignment/PKU-SafeRLHF`

Training construction:

- Start from `meta-llama/Llama-3.1-8B-Instruct`.
- Select safety preference examples with a clearly safer response.
- Use the safer response as the supervised fine-tuning target.
- Sample approximately 2,000 safety examples for the initial controlled run.
- Add approximately 2,000 benign instruction/response examples to reduce
  generalized over-refusal.
- Do not use AdvBench examples during fine-tuning.
- Keep the original held-out safety and epistemic evaluation prompts unchanged
  across checkpoints A, B, and C.

Training method:

- QLoRA / parameter-efficient fine-tuning
- fixed dataset-construction seed: 67
- fixed training seed: 42
- save the adapter separately from the base model
- record all hyperparameters and dataset indices used to construct checkpoint C

The first experiment should favor a small, controlled intervention over maximizing
safety performance. The goal is to measure representational change caused by
additional safety training.
