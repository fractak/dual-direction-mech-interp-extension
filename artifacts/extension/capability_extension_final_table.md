# Capability Refusal Extension — Final Results

## Primary Llama results

| Measure | Result |
|---|---:|
| Peak capability separability (model.layers.24) | 0.672 |
| Held-out probe accuracy (model.layers.24) | 0.925 |
| Capability ablation: net capability-refusal drop | 0.133 |
| Capability ablation: exact paired p-value | 0.0078 |
| Capability ablation: bootstrap 95% CI | [0.050, 0.217] |
| Safety ablation: net capability-refusal drop | 0.083 |
| Safety ablation: exact paired p-value | 0.1250 |
| Safety ablation: bootstrap 95% CI | [0.000, 0.167] |

## Cross-model geometry

| Model | Capability peak | Cap. sep. | Mean S-E | Mean S-C | Mean E-C |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | L24 | 0.672 | 0.049 | 0.206 | -0.090 |
| Qwen3-8B | L20 | 0.447 | -0.001 | 0.199 | -0.075 |
| Gemma-2-9B | L24 | 0.898 | 0.112 | 0.210 | -0.135 |

## Quantization robustness

| Precision | Direction cosine vs FP16 | Capability separability |
|---|---:|---:|
| fp16 | 1.000 | 0.672 |
| bnb_nf4 | 1.000 | 0.672 |
| bnb_int8 | 0.950 | 0.677 |
