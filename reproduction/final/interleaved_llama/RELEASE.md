# Reproduce interleaved Llama Figure 1, Table 1, and intervals

With NumPy and Matplotlib installed, run from this directory:

```bash
python build_release.py build family_counts.csv reproduced
```

The family table has 720 cells, one per cohort, family, receiver, and
condition, with four prompt phrasings per cell. It retains all four label
counts for adjudication and both reviewer streams, plus cap and input
truncation counts. The script reconstructs the matched family contrasts,
uses independent between-set resampling with 20,000 paired family draws
and seed 42, and verifies estimates and cap counts against reference.json.
Each estimate array is [point, lower 95%, upper 95%] in percentage points.
The evidence and format receiver contrasts each subtract the original
receiver's baseline-to-edit change from the mean of the adapter changes.

reviewer_provenance.md records the available exact evaluator versions,
reviewer prompts, and settings. An unavailable field requires a
contemporaneous record to fill. agreement.json identifies the completed
reviewer CSVs and adjudications by their SHA-256 hashes. manifest.json
pins the reviewed source, frozen plan identifiers, and packaged files.
The aggregate package recomputes the paper's statistics from saved labels.
Regeneration requires the original frozen plans, model, adapters, and runner.

Commit the verified release directory in the experiment repository and
link its immutable commit URL in the manuscript. If making a release tag,
point it to that same commit.
