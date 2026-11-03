# Validation record

Validation date: November 2, 2026 (America/Chicago).

- Verified all 15 frozen input hashes.
- Recomputed the archived main table and paired intervals: acknowledgment 58 -> 49 original versus 58 -> 6 adapted; unsupported 0 -> 1 versus 0 -> 40; acknowledgment difference in changes -71.7 percentage points, interval [-83.3, -60.0].
- Regenerated four PNG and four PDF figures from archived inputs.
- Full tests: **69 passed, 1 skipped**. See `artifacts/validation/tests.txt`.
- The skipped float16 cast-back measurement test requires CPU information unavailable in this sandbox's PyTorch runtime. The float32 version passes. This is not a successful FP16/NF4 model validation.
- New modules and prospective/review entry points pass the configured Ruff checks. Existing unrelated code was not reformatted.
- Synthetic tests cover equal relative movement despite different projections, projection sign reversal, tuple preservation, cache-phase assumptions, zero-projection policy, hook cleanup on failure, successful instrumented generation, plan checksums, receiver/source planning, label deduplication, reviewer disagreement/adjudication, source-mutation detection, prompt pairing and domain-cluster intervals.

Limits: no GPU, full Llama model, model authentication or historical adapter was available. There is no new adapter training, full-model extraction, measured Llama response generation, independent human labeling, confirmatory dataset, or cross-seed replication in this release. Dependency versions used for CPU validation are in `artifacts/validation/environment.json`; they are not a certified GPU environment or a full dependency lock.
