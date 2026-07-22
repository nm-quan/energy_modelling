# D1 — constraint-method ablation (training = T4)

| method | guarantee | MAE 3h | MAE 6h | MAE 12h | MAE blackout | bal>1MW steps | ramp cells | neg cells | SOC days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 none (= T4) | no | 97.5 | 96.5 | 94.4 | 95.5 | 110100 | 495 | 218572 | 0 |
| C1 soft λ=0.1 | no | 81.5 | 82.4 | 85.2 | 93.4 | 110091 | 340 | 163832 | 0 |
| C2 posthoc Π(C0) | YES | 86.7 | 89.6 | 92.8 | 96.4 | 0 | 0 | 0 | 0 |
| C3 RAYEN fixed-D | YES | 50.0 | 73.7 | 110.7 | 146.6 | 0 | 0 | 0 | 92 |
| ref: causal (left ctx only) (≡ C0 for blackout: no flanks to mask) | no | 91.6 | 94.6 | 96.2 | 95.5 | 110063 | 620 | 196905 | 0 |
| ref: pro-rata share×nd | no | 191.2 | 190.1 | 193.8 | 186.6 | 0 | 2145 | 0 | 0 |

_Feasibility floor (blackout, map applied to the TRUTH): F_Π = 0.01 MW; F_rayen = 145.58 MW. Π costs a perfect prediction ~nothing; a large F_rayen explains a large C3 MAE as the mechanism's conservatism, not the network._

**Best guaranteed-feasible: C2 posthoc Π(C0)** (blackout MAE 96.4 MW).
