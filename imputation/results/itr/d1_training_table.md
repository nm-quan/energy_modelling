# D1 — training-method ablation (T1–T4, raw fills, MAE in MW)

| arm | MAE 3h | MAE 6h | MAE 12h | MAE blackout | val blackout-MAE (T* metric) |
| --- | --- | --- | --- | --- | --- |
| T1 | 93.3 | 91.7 | 89.0 | 93.8 | 97.5 |
| T2 | 83.3 | 84.7 | 85.3 | 91.3 | 95.5 |
| T3 | 194.7 | 189.6 | 186.3 | 95.8 | 94.8 |
| T4 | 97.5 | 96.5 | 94.4 | 95.5 | 92.7 |

**T\* = T4** (val blackout-MAE 92.7 MW). ⚠ TIE within 5%: ['T2', 'T3'] — treat as equivalent; rerun with seeds only if the choice matters.
