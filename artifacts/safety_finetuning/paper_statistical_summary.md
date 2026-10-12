# Paper Statistical Summary

## Primary confirmatory comparisons

| ID | Comparison | n | Before | After | Effect | 95% CI | Raw p | Holm p | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| P1 | B→C harmful-request refusal | 200 | 0.950 | 0.995 | +0.045 | [+0.020, +0.075] | 0.0039 | 0.0039 | detector |
| P2 | B→C unanswerable abstention | 200 | 0.780 | 0.560 | -0.220 | [-0.280, -0.160] | 8.36e-12 | 2.51e-11 | detector |
| P3 | C safety refusal under C-safety-direction ablation | 200 | 0.995 | 0.270 | -0.725 | [-0.785, -0.665] | 4.48e-44 | 1.79e-43 | detector |
| P4 | C capability refusal under C-capability-direction ablation | 60 | 0.967 | 0.317 | -0.650 | [-0.767, -0.517] | 3.82e-11 | 7.64e-11 | independent blind annotation |

Holm correction is applied only to the four pre-designated primary confirmatory comparisons.

## Secondary behavioral comparisons

| Comparison | n | Before | After | Effect | 95% CI | Raw p | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| B→C benign refusal | 200 | 0.000 | 0.045 | +0.045 | [+0.020, +0.075] | 0.0039 | detector |
| B→C answerable-question abstention | 200 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 1.0000 | detector |
| B→C capability refusal | 60 | 0.950 | 0.967 | +0.017 | [+0.000, +0.050] | 1.0000 | independent blind annotation |

## Statistical interpretation

- Paired binary behavioral changes use exact McNemar/binomial tests over discordant pairs.
- Effect sizes are absolute changes in behavioral rate.
- 95% confidence intervals use paired bootstrap resampling.
- Geometry analyses are descriptive effect-size analyses with prompt-resampling stability controls.
- Capability labels are described as independent blind annotations.
