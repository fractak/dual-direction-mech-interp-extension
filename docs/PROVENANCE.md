# Evidence provenance

## Scope

This repository supports the access-limit acknowledgment study: the fixed activation operator, checkpoint-specific controls and doses, direction geometry, and the secondary refusal/reason overlap study.

Names are descriptive throughout active code, filenames, commands, figures, and recomputed output. Source filenames and blind IDs in provenance retain historical spelling so they can be matched to the original snapshot. In that snapshot, `B`/`B_instruct` meant `original_instruct`; `C`/`C_safety_lora` meant `safety_lora`. A single `C` also sometimes meant the capability contrast, which is why broad text replacement would have been unsafe.

## Preserved evidence

- 1,020 capability response records in 17 groups of 60. Prompt strings, response strings, and final annotation decisions are unchanged.
- 1,040 overlap response records in 52 groups of 20. All four original Boolean annotation fields are retained.
- Six complete 32-layer direction collections. Original Instruct safety and epistemic geometry uses the model-named collections, alongside `capability_train_160.json`; Safety LoRA uses its three archived collections. The nearby `*_200.json` variants were not silently averaged into these vectors.
- The 80-pair capability training set, 60-context expanded evaluation set, 140-prompt overlap matrix, and 1,998-example mixed training dataset.
- Historical behavior and resampling summaries as explicitly marked summary-only context. Their absent raw outputs or second-seed vectors cannot be reconstructed.

`evidence/manifest.json` records SHA-256 hashes of all frozen normalized inputs, every source file used in migration, and the complete original snapshot. Row-level source references connect normalized records to the original response and annotation files. The migration translates `explicit_capability` to `acknowledgment`, `instructions_only` to `instructions`, and `fabricated_answer` to `unsupported`; all-zero labels map to `other`. It does not change judgments.

The original training manifest's exclusion flag is retained byte-for-byte. See the training section of [EXPERIMENTS.md](EXPERIMENTS.md) for its known schema issue.

## Removed from the active tree

| Historical material | Disposition |
|---|---|
| Five root-level control scripts and source-editing random-seed shell loop | Replaced by parameterized `scripts/run_interventions.py` |
| Overlapping statistical builders and legacy detector-based summaries | Replaced by `scripts/analyze_paper.py` and shared paired statistics |
| Three old figure generators and hard-coded paper plotting counts | Replaced by one evidence-driven four-figure generator |
| Old cross-family replication, quantization, probes, triple-direction pipeline, notebook, and experiment configs | Preserved in the historical backup; outside this study's supported workflow |
| Unused combined/evaluation pair variants and overlap prompt overrides | Preserved in the historical backup; frozen training/evaluation inputs remain |
| Duplicate blind review templates, superseded results, and generated figures | Preserved in the historical backup; normalized final evidence remains |
| Machine-specific export manifest and old design document | Preserved in the historical backup; replaced by current protocol/provenance documentation |
| Detector/probe/legacy metrics modules and their tests | Preserved in the historical backup; no longer imported by the active workflow |

## Limitations retained

Adapter weights, effective training logs, exact historical model revisions, and extraction split/prompt hashes are absent. Re-running a preserved recipe does not recreate the archived checkpoint exactly. A single training recipe with both safety and Alpaca examples does not identify the cause of an observed change.

The main comparison matches the activation operator, not its realized displacement. Orthogonality does not imply independent mechanisms. Unsupported assertions are not verified false private-state reports. The overlap rubric measures expressed reasons, which may accompany non-refusal. Counts and intervals are conditional on one represented checkpoint pair and the evaluated prompt contexts.
