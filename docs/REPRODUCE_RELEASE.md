# Reproduce the reported results

The manuscript is distributed separately. This repository release includes
the experiment code, numerical references, and figure generators.

## CPU rebuild

From the repository root, in the CPU environment:

    python -m pip install -r requirements-paper-figures.txt
    python scripts/reproduce_release.py --check-inputs cpu
    python scripts/reproduce_release.py

Outputs go to artifacts/reproduced/. The archived analysis verifies its
evidence manifest and rebuilds four figures. The measured analyses use
resolved labels and telemetry, then rebuild the measured dose and
learned-direction figures. Swap intervals use per-prompt continuation
scores. A preexisting output directory must be moved aside before a rerun.

reproduction/inputs.json lists exact relative paths and sizes from the GPU
checkout audit. A clean checkout needs the saved input files at those paths.
If publishing the code, distribute those files as a companion evidence
archive or through a suitable large-file store, with a SHA-256 manifest.
The numerical references alone cannot reproduce the paired intervals.
The expected resolved run hashes and frozen plan identifiers are recorded
in the numerical references where available.

The resolved review JSONs reproduce the analysis conditional on the
saved labels. To independently audit labeling, retain the original reviewer
files and adjudication records. Keep private maps out of a blinded handoff.

## GPU reruns

Use the full plans listed in reproduction/inputs.json. The plans embed the
prompts, direction vectors, condition grids, model revisions, and seeds.
The adapter files are required separately. For each plan, specify the
matching receiving adapter and a new run directory:

    python scripts/run_publication_study.py run --plan PLAN --adapter-path ADAPTER --output-dir NEW_RUN

The Llama seed-42, seed-43, and seed-44 plans use their corresponding
mixed-data adapters. Alpaca plans use the Alpaca-only seed-42 adapter.
Qwen plans use the Qwen3-8B mixed-data seed-42 adapter. The saved
training metadata and dataset preserve the training recipes. Original
seed-42 Llama training metadata lacks the historical base revision.

For activation swaps, use scripts/reproduction/run_swap.py with the
frozen activation-swap plan and a new output directory. It saves prompt
checkpoints and reproduces target-token scores without free generation.

Historical archived runs remain a CPU reanalysis path. Some original
generation environment details and extraction inputs were not recorded.

## Cleanup

The inventory-derived cleanup list contains smoke-run directories and
an intermediate recovery backup. The archive command previews paths by
default and moves them outside the checkout with --apply. Full runs,
frozen plans, adapter artifacts, review records, and tracked historical
evidence are retained.

## Package the saved inputs

On the GPU checkout, generate a CPU companion archive with:

    python scripts/pack_reproduction_inputs.py /home/ubuntu/ddmi_cpu_inputs.tar.gz

Add --include-gpu to include the frozen GPU plans, direction files, and
adapter artifacts. Verify the archive using its .sha256 file, then extract
it into a clean checkout before running the CPU rebuild. The repository
contains the archived evidence it already tracks. The companion archive
contains the additional measured-study inputs.
