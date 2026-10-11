# Safety Fine-Tuning Extension — Final Results

## Behavioral change: B → C

| Behavior | B | C | Change | Paired result |
|---|---:|---:|---:|---|
| Harmful-request refusal | 0.950 | 0.995 | +0.045 | p=0.003906 |
| Benign refusal | 0.000 | 0.045 | +0.045 | p=0.003906 |
| Unanswerable abstention | 0.780 | 0.560 | -0.220 | p=8.36e-12 |
| Answerable abstention | 0.000 | 0.000 | 0.000 | p=1.0 |
| Capability refusal, blind human | 0.950 | 0.967 | +0.017 | p=1.0 |

## Representational geometry

| Quantity | Result |
|---|---:|
| Mean safety B→C cosine | 0.788 |
| Mean epistemic B→C cosine | 0.819 |
| Mean capability B→C cosine | 0.695 |
| Safety peak separability | 0.741 → 1.009 |
| Epistemic peak separability | 1.041 → 0.908 |
| Capability peak separability | 0.672 → 0.665 |
| Mean safety–capability cosine | 0.206 → 0.285 |
| Mean safety–epistemic cosine | 0.049 → 0.043 |

## Causal transfer

| Target model | Ablated direction | Main behavioral effect |
|---|---|---:|
| B | B safety | safety refusal -0.735 |
| B | C safety | safety refusal -0.835 |
| C | B safety | safety refusal +0.005 |
| C | C safety | safety refusal -0.725 |
| B | B capability | capability refusal -0.100 |
| B | C capability | capability refusal -0.517 |
| C | B capability | capability refusal -0.017 |
| C | C capability | capability refusal -0.967 |

## Main interpretation

Safety fine-tuning does not merely amplify a fixed refusal direction. It reorganizes a broader refusal-related representational system.

Safety refusal becomes stronger and its causally effective direction is reorganized. Epistemic abstention falls substantially and its representation rotates, while direct epistemic-direction ablation remains weak; notably, safety-direction ablation restores much of the lost epistemic abstention in the tuned checkpoint. Capability refusal behavior remains nearly unchanged on 60 unseen task families even though its representation rotates most strongly and its causal control shifts dramatically toward the post-tuning direction.

Together, the results support a broader reorganization of refusal-related representations rather than simple amplification of a single fixed abstention/refusal axis.
