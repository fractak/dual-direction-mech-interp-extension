# Reproduce the final paper

The manuscript source is distributed separately. This repository contains the
experiment implementation and the saved inputs, results, and figures described
below. "Reproduce" here means rerunning the numerical analysis from saved data.
GPU regeneration is a separate workflow.

## CPU verification

From the checkout root, with Python 3.10 or newer:

    python3 -m venv .venv-figures
    source .venv-figures/bin/activate
    python -m pip install -r requirements-paper-figures.txt
    python scripts/reproduce_final.py

This verifies the interleaved Llama and Qwen estimates, the paired seed-42 gap,
and the held-out dose sweep and matched-training estimates, including the
reviewer-specific sensitivities bundled in the releases. It regenerates the
interleaved Llama family ECDF. Outputs go to
artifacts/analysis/reproduced_final/. Use --output with a fresh path for
subsequent runs.

For all seven figures and the supplementary saved-input analyses, hydrate LFS,
then run:

    git lfs pull
    python scripts/reproduce_release.py --check-inputs cpu
    python scripts/reproduce_final.py --legacy \
      --reviewed-archive reproduction/final/late-reviewed-v1.zip \
      --output artifacts/analysis/reproduced_final_full

The --legacy option also writes artifacts/reproduced/. Move any previous output
with that name aside before starting. The six legacy figures are copied into
the same output directory as the newly generated interleaved figure. Figure
PDFs used by the manuscript are under reproduction/final/figures/. Plot
regeneration can vary at the PDF-byte level with the rendering stack.

The late-reviewed archive is constructed on the original experiment server.
Its verification covers the fresh-family and boundary-position estimates,
exact-pair label sensitivity, attenuation, joint relabeling, both human audits,
the Qwen uncapped sensitivity, and the reported token localization estimates.
The eight source files with retained reference hashes are verified by SHA-256.
The localization export is checked against the condition design, published
point estimates, and reported marginal intervals.

## Prepare the server-side inputs

On the original worktree with the experiment results, first run:

    python reproduction/final/server_inputs.py

The checker locates the required reviewed exports by SHA-256. If it reports a
missing file, add --search /path/to/that/export.json. Provide
--localization-reviewed PATH if its reviewed export is stored in a different
directory. Rerun the successful command with --write to create
reproduction/final/late-reviewed-v1.zip. The archive includes experimental
prompt and response text and review labels. Keep the private blind maps outside
the release. Use the full CPU verification above before staging the archive.

## GPU regeneration

The historical inventory and existing commands are in REPRODUCE_RELEASE.md.
To archive the earlier and later frozen GPU plans, adapter weights, training
inputs, and current runner source from the original server, run:

    python reproduction/final/pack_frozen_gpu.py /home/ubuntu/ddmi_final_gpu.tar.gz \
      --runner /home/ubuntu/receiver_robustness \
      --runner /home/ubuntu/interleaved_replication \
      --runner /home/ubuntu/replication_positions \
      --runner /home/ubuntu/token_localization

Supply each experiment runner directory still available, including the Qwen
runner directory if it is separate. The archive sidecar pins the actual
source snapshot by file hash and indexes the later plans. The archive does
not bundle upstream base-model weights. Obtain the pinned revisions and
compatible GPU environment separately. Unpack into a separate worktree or
directory. Use the matching adapter and frozen plan with the recorded
experiment runner. See the individual runner protocols for extraction and
training. The saved labels allow verification without re-running judges.
Historical provenance gaps and missing judge prompts and versions still limit
exact recreation of some generations and annotations.

## Coverage and release boundaries

- The two versioned releases under reproduction/final/releases/ match the
  manuscript attachments. The human-audit JSONs give the saved audit summaries.
- The historical archived detector-rate and second-seed geometry summaries are
  preserved as references. Their raw inputs were not retained.
- The CPU runner checks core numeric outputs and figure construction. It does
  not regenerate the manuscript's layout or claim pixel-identical PDFs.
- Keep model output text, prompt text, and original labels from the late
  archive with the release if publishing it. Private review maps, credentials,
  live journals, smoke runs, and manuscript files are outside the staging list.
