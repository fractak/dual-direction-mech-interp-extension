# Fresh-family replication and position control

The reviewed exports preserve the prompts, responses, telemetry, both reviewer
labels, adjudications, and plan hashes. The reference JSONs include source
SHA-256 hashes, condition counts, paired family-bootstrap results, and the
reviewer-only and uncapped sensitivities. The position reference also records
the token-mask audit and a separate sensitivity using earlier labels for exact
duplicate prompt-response pairs.

Run from the repository root with NumPy installed and Git LFS inputs hydrated.

```bash
python scripts/reproduction/analyze_replication_positions.py family_replication \
  artifacts/review/family_replication_resolved/reviewed_source.json \
  --output /tmp/family_replication_results.json

python scripts/reproduction/analyze_replication_positions.py position_control \
  artifacts/review/position_control_resolved/reviewed_source.json \
  --output /tmp/position_control_results.json
```

To recompute the exact-pair sensitivity and token-mask audit from the saved
exports and earlier primary study:

```bash
python scripts/reproduction/audit_replication_positions.py --results-dir /tmp
```

To verify the merges, supply `--review-dir` with the completed reviewer CSVs
and `--adjudications-dir` with both adjudication CSVs.

Compare the generated analyses, condition counts, and source/plan hashes with
the matching files in `reproduction/reference/`. The additional audit fields in
those references describe checks against the blinded CSVs, position telemetry,
and earlier primary-study labels. The frozen analyses use 20,000 paired
family-bootstrap draws and NumPy seed 42. The two primary position comparisons
use 97.5% marginal intervals for nominal simultaneous 95% coverage.

The experiment bundle contains the protocols, prompt construction inputs,
position patch, and runner. For generation, retain the completed plans from
the three seed runs and the position-control full run. Those plans use the
existing model and adapter inputs. The separately distributed manuscript
contains the LaTeX sources and compiled PDF.
