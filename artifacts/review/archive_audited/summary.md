# Agent-reviewed archived results

Agent-review and assistant-adjudication sensitivity analysis of archived outputs; not human agreement or new model generation.

| Checkpoint | Condition | Acknowledgment | Instructions | Unsupported | Other |
|---|---|---:|---:|---:|---:|
| original_instruct | baseline | 58 | 2 | 0 | 0 |
| original_instruct | capability_from_safety_lora | 48 | 9 | 3 | 0 |
| safety_lora | baseline | 58 | 2 | 0 | 0 |
| safety_lora | capability_from_safety_lora | 6 | 8 | 42 | 4 |

Receiver differences in changes (percentage points; paired bootstrap 95% CI):
- acknowledgment: -70.0 [-81.7, -58.3]
- unsupported: 65.0 [53.3, 76.7]

Intervals are exploratory, conditional on these contexts and one checkpoint pair.
