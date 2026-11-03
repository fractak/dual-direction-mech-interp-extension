# Experiment protocol

The central finding is: equal access-limit acknowledgment on 60 archived contexts coexists with different sensitivity to one specified activation operator. The evidence does not establish three mechanisms, a general safety-training effect, or greater sensitivity at equal activation displacement.

## What is ready

- The 15 frozen input checksums verify. The main count table and paired bootstrap intervals rebuild on CPU.
- All four figures regenerate from frozen evidence.
- Independent reannotation templates are in `artifacts/review/archive/`: 546 unique prompt-response pairs represent 1,020 archived rows. No new human labels have been supplied.
- Measured prospective runs, same-layer receiver/vector comparisons, relative-displacement controls, review merging, and new-run analysis are implemented and tested on synthetic CPU models. Full Llama inference and training are not validated in this environment.

## Experiment sequence

### 1. Recover or replace the adapter

First try to recover the exact historical adapter and its training record. If unavailable, train new checkpoints and call them replications of the experimental design, not regeneration of historical outputs. Preserve every adapter, tokenizer, training state, dataset hash, model revision, package version and effective optimizer-step log.

Use at least three independent training seeds for a stronger paper. Compare the mixed 999 safety + 999 Alpaca recipe with an Alpaca-only control. Equal optimizer steps are useful but do not equalize tokens; report processed tokens and training loss. Treat the adapted model as the `safety_lora` receiver slot for compatibility, while clearly naming each adapter's actual recipe and seed in the paper and run directory.

The trainer now supports `--kind all|benign|safety` and `--revision`. Example (the revision below must be a real immutable model commit):

```bash
python scripts/train_safety_lora.py --kind all --seed 42 \
  --revision MODEL_COMMIT_SHA --max-steps 125 \
  --output-dir artifacts/adapters/mixed_seed42
python scripts/train_safety_lora.py --kind benign --seed 42 \
  --revision MODEL_COMMIT_SHA --max-steps 125 \
  --output-dir artifacts/adapters/alpaca_seed42
```

These are proposed new runs. The 125-step budget is a design choice approximating one pass over 1,998 examples at effective batch 16, not a recovered historical setting. Verify the installed training stack and do a short smoke run before committing a training budget.

### 2. Measure the displacement explanation on a pilot

Use the archived prompts only for diagnosis and calibration. Extract new capability directions from the original training pairs for each newly trained checkpoint. Store files as `artifacts/new_directions/original_instruct/capability.json` and `artifacts/new_directions/safety_lora/capability.json`. The new extraction script accepts `--revision`, records package versions and adapter file hashes, and refuses to overwrite frozen evidence.

```bash
python scripts/extract_directions.py --checkpoint original_instruct \
  --direction-type capability --capability-split train --prompt-limit 160 \
  --revision MODEL_COMMIT_SHA --load-in-4bit \
  --output artifacts/new_directions/original_instruct/capability.json
python scripts/extract_directions.py --checkpoint safety_lora \
  --adapter-path artifacts/adapters/mixed_seed42 \
  --direction-type capability --capability-split train --prompt-limit 160 \
  --revision MODEL_COMMIT_SHA --load-in-4bit \
  --output artifacts/new_directions/safety_lora/capability.json
```

Copy `configs/publication_pilot.json` and replace its revision placeholder with the exact model commit. The full grid has 124 conditions and 7,440 responses on 60 prompts, per adapter. The CPU plan step prints the count before inference. For a smaller pilot, use layer `[20]`, projection strengths `[1.0, 1.5]`, relative budgets `[0.03]`, and the six random seeds: 26 conditions and 1,560 responses. Relative budgets are provisional design values; calibrate them on pilot data before freezing a confirmatory target.

```bash
python scripts/run_publication_study.py plan \
  --config configs/my_pilot.json \
  --direction-root artifacts/new_directions \
  --output artifacts/plans/pilot.json
python scripts/run_publication_study.py run \
  --plan artifacts/plans/pilot.json \
  --adapter-path artifacts/adapters/mixed_seed42 \
  --output-dir artifacts/runs/pilot_seed42
```

Each layer includes unedited observation of both vectors, both vector sources crossed with both receivers at that same layer, projection strengths including exactly 1, and random controls at the same relative budget. Baseline generations repeat across observation layers; these repetitions are not independent samples.

### 3. Freeze a separate evaluation before inspecting its outputs

Author a new evaluation set independently of this pilot. Suggested minimum design: 120 inaccessible-information prompts across at least 12 domains, with several wording styles and an explicit domain identifier. This is a practical starting scope, not a power calculation. Use the pilot's variance and domain dependence to set the final sample size. Pair each with an evidence-supplied version for a separately reported utility check; the new runner currently evaluates the inaccessible prompts only.

Prompt-file schema:

```json
[
  {"id": "unique_context_id", "domain": "domain_name", "prompt": "A new inaccessible-information request"}
]
```

Do not use that example as an evaluation item. Human-review the set for answerability, paraphrase leakage, accidental clues, domain balance and semantic overlap with training and extraction inputs. Exact-string checks alone are insufficient. Do not call archived prompts held out again.

Set `purpose` to `confirmatory`, supply the new prompt file and new direction root, and freeze the plan before generation. The planner rejects exact overlap with frozen capability/training prompts and recorded extraction hashes, and checks direction checkpoint/model revisions. It cannot verify whether a human already inspected the prompts or enforce external preregistration. Archive the plan hash with a dated protocol before outcomes are seen.

**Proposed primary contrast:** difference between receiver changes in unsupported-assertion rate, using the adapted capability direction at layer 20 and one relative budget chosen on the pilot. Use one budget in the confirmatory config; retain other sources and randoms as predeclared secondary controls. Specify this contrast, direction, layer, budget, rubric and exclusion rules in a signed-off protocol before collection. The general analyzer reports every contrast as exploratory; it does not promote results to confirmatory status automatically.

**Secondary outcomes:** acknowledgment, instructions, other, last-prefill projection and displacement; truncation and output-limit rates; same-layer source transfer; random controls. Do not choose the best layer or budget on the evaluation set.

### 4. Independent review

Use `docs/ANNOTATION_RUBRIC.md`. Give each reviewer only their CSV and the rubric. Keep `PRIVATE_review_map.json` with the experiment owner; it identifies source conditions. Reviewers must work independently. The software cannot establish human independence from file names.

```bash
python scripts/review_responses.py export \
  --source artifacts/runs/pilot_seed42/run.json \
  --output-dir artifacts/review/pilot_seed42
python scripts/review_responses.py merge \
  --map artifacts/review/pilot_seed42/PRIVATE_review_map.json \
  --reviewer-a artifacts/review/pilot_seed42/reviewer_A.csv \
  --reviewer-b artifacts/review/pilot_seed42/reviewer_B.csv \
  --output-dir artifacts/review/pilot_merged
```

Merge reports raw agreement, a confusion matrix and Cohen's kappa on unique pairs before adjudication. A kappa of `null` means expected agreement is one, so kappa is undefined. Disagreements retain no final label. Complete `pilot_merged/adjudications.csv`, then repeat merge with `--adjudications` and a new output directory. Old archived outcomes remain intact under their original fields; independent judgments are stored separately.

```bash
python scripts/analyze_reviewed_run.py \
  --plan artifacts/runs/pilot_seed42/plan.json \
  --reviewed-run artifacts/review/pilot_resolved/reviewed_source.json \
  --output artifacts/analysis/pilot_seed42.json
```

The analyzer rejects incomplete runs, unresolved labels, changed condition settings, missing telemetry and broken prompt pairing. It provides paired prompt intervals and domain-cluster intervals when domains are supplied. Report the sampling assumptions. Domains and prompts do not substitute for training seeds; do not pool all runs as independent response rows. Report per-seed effects and a separately designed across-seed analysis.

### 5. Interpret the possible outcomes

| Result | Interpretation supported |
|---|---|
| Larger projection-edit effect disappears under relative matching | Different realized displacement is a plausible contributor; do not claim greater downstream sensitivity. |
| Gap persists under relative matching and exceeds random controls | Stronger evidence for direction-specific differential sensitivity at comparable relative movement; still no complete circuit identification. |
| Similar gap follows Alpaca-only adaptation | The effect is not specific to safety examples under this training design. |
| Gap depends strongly on one seed, wording family or rubric | Report heterogeneity; narrow or revise the claim. |
| Baseline behavior differs on new prompts | Equal baseline rates were specific to the archived sample; report the difference directly. |
