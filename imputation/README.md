# Gap-imputation track — the counterfactual reframe

The simulation ("respond to +g% midday demand") recast as **constrained subspace
imputation** (supervisor's reframe; lit_review.md themes 8–9). Not step-ahead
forecasting: we KNOW the data on both sides of the 11:00–14:00 window, and inside
it we know the totals (net_demand, demand, price, calendar) at every step — only
the 6-source *breakdown* is hidden.

## The subspace (why "impute", not "forecast")

Per 5-min row, feature columns split:

| | columns | role |
| --- | --- | --- |
| **known** everywhere | net_demand[6], demand[7], price[8], calendar[9–16] | given inside the gap too |
| **unknown** in gap | the 6 sources [0–5] | the imputation target |

`net_demand = SIGN·sources` **exactly** (verified: max residual 0.000 MW), so inside
the gap the model is handed the exact *sum* of the six hidden numbers and must
recover the *split* — 5 free degrees of freedom, pinned further by both boundaries,
ramps, box, SOC.

## Pipeline (per constraints/lit_review.md theme 9)

1. **gap_data.py** — build windows `[ctxL | gap | ctxR]` from the 4.8-yr hist flats
   (`prepared.npz`, no parquets). Test = real 11–14 windows (186 days); train =
   random-position masks over 394k rows (calendar carries time-of-day).
2. **constraints.py** — the two-sided **ramp tube**: forward cone from p_L +
   backward cone from p_R, box, then balance, enforced by forward/backward POCS.
   Guarantees every consecutive pair incl. **both seams** (the 2:05 fix) is
   ramp-feasible. Feasible iff `|p_R − p_L| ≤ (N+1)·r` per channel.
3. **baseline.py** — zero-learning bar: per-source linear interpolation between
   boundaries (± constraint projection). The imputation-track "persistence".
4. **model.py / train.py** — bi-LSTM over the window + mask channel, ramp-tube
   head, trained on masked + perturbed windows. *(next)*

## First result — baseline reconstruction WAPE (186 real test days)

Mask the real 11–14 window, fill, score vs measured truth (answer key exists).

| method | macro | coal | hydro | ocgt | gas_steam | batt_chg | **batt_dis** | ramp viol | n_neg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear interp | **0.345** | 0.023 | 0.273 | 0.108 | 0.215 | 0.317 | **1.136** | 0 | 0 |
| interp + constraint proj | 0.406 | 0.021 | 0.388 | 0.357 | 0.215 | 0.317 | 1.136 | 0 | 0 |

Reading:
- **Coal is nearly free to impute (0.023)** — smooth and slow, the boundaries
  almost determine it. Interpolation basically solves coal.
- **Batteries are the wall (batt_dis 1.14)** — they swing fast mid-gap, so
  boundary interpolation is hopeless. Same channel that dominates the forecasting
  WAPE. This is where a learned model must earn its keep.
- **Naive constraint projection makes reconstruction WORSE** (0.345 → 0.406):
  forcing `SIGN·P = net_demand` shoves the residual onto hydro/ocgt (0.27→0.39,
  0.11→0.36). Interpolation ignores net_demand, so its sum is ~1000 MW off; the
  projection fixes the sum but guesses the wrong channels. **The bi-LSTM's job is
  exactly this:** read net_demand and route the balance to the right channels —
  beat 0.345 macro *while* staying constraint-clean.
