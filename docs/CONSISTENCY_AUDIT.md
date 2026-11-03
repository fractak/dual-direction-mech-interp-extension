# Apply the agent-review consistency audit

Run after both agent reviews and the two-disagreement adjudication have been merged. This audit covers every unique response whose resolved label was `other`. It is a consistency audit of that selected category, not a new independent review of all responses.

```bash
python scripts/apply_consistency_audit.py \
  --source artifacts/review/archive_resolved/reviewed_source.json \
  --map artifacts/review/archive/PRIVATE_review_map.json \
  --audit /home/ubuntu/consistency_audit_adjudicated.csv \
  --output-dir artifacts/review/archive_audited \
  --figures
```

The output directory must not already exist. The command checks frozen-input hashes, exact source-row identity, map coverage, audit text and complete coverage of the previously resolved `other` items. Every audit label must have a reason. Original archive labels and `independent_review` judgments remain unchanged; effective labels are recorded as `audited_outcome` in a new copy, with per-row consistency-review reasons and input-file hashes.

Outputs:

- `reviewed_source.json`: original judgments plus separately recorded audited outcomes.
- `results.json`: all capability-condition counts and paired statistics before and after this audit.
- `summary.md`: main counts and paired bootstrap differences in changes.
- `figure1_agent_reviewed.png` and `.pdf`: main comparison explicitly labeled as agent-reviewed archived responses, generated only with `--figures`.

These are a label-sensitivity analysis of the existing model responses. They do not constitute new GPU experiments or human inter-rater validation. Original frozen evidence, historical statistics, and figures in `artifacts/figures/` are not replaced. Compare both label protocols and disclose the missing-information clarification and selected-category audit in any scientific report. The original pre-adjudication agreement statistic must not be recomputed as if adjudicated or audited labels were independent reviewer judgments.
