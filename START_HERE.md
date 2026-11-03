# Start here

The finding: equal baseline access-limit acknowledgment can coexist with different sensitivity to the same activation operator after LoRA adaptation. This repository contains the experiment code, archived evidence, analysis and figures supporting that comparison.

1. Read `docs/EXPERIMENT_PROTOCOL.md` for the experiment sequence, commands and interpretation rules.
2. Give `artifacts/review/archive/reviewer_A.csv` and `reviewer_B.csv` to independent reviewers with `docs/ANNOTATION_RUBRIC.md`. Keep the private map with the study owner. These cover 546 unique pairs from all 1,020 archived capability responses.
3. Recover the original adapter, or train explicitly new adapters; use `scripts/run_publication_study.py` for measured follow-up experiments. Fill in the immutable model revision in a copy of the configuration first.

Reproduce the archived analysis and figures:

```bash
python -m pip install -e '.[paper,dev]'
python scripts/analyze_paper.py
python scripts/make_paper_figures.py --check-paper
```

Figures are saved under `artifacts/figures/` as PNG and vector PDF files. The `paper` dependency extra installs plotting dependencies; no LaTeX installation is needed.

With PyTorch installed, `python -m pytest -q -ra` runs the synthetic intervention tests as well as the CPU evidence tests. The recorded validation environment passed 69 tests and skipped one float16 conversion test because of a sandbox CPU-information restriction. GPU inference/training need the additional dependencies in `.[gpu,train]`, model access and adapter weights.

No remote GitHub update was performed.
