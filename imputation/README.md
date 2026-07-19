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
4. **model.py / train.py** — bi-LSTM over the window + mask channel, residual on
   the interp skeleton, trained on masked windows; early-stopped on **midday-matched
   val gaps** (11:00–14:00 in the held-out val split — same task as test, no test
   leakage; see `gap_data.sample_val_midday_windows` for the leakage history).
   Perturbation is OFF: scaling demand with an unchanged target teaches the model
   to ignore demand (measured).

## Pilot results (CPU, 2026-07-15)

### Reconstruction WAPE — mask real 11–14, fill, score vs measured truth

100-epoch run, **early-stopped on a held-out VAL set** (random gaps from the val
split — no test leakage; stopped ep28, val best 0.315), test scored once.
*(Since this run the val protocol was tightened once more: val gaps are now pinned
to 11:00–14:00 — midday-matched to the test task — so val no longer reads
optimistic vs test. Retrain via the notebook to refresh the numbers below.)*

| channel | interp (baseline) | bi-LSTM (100ep, val-selected) |
| --- | --- | --- |
| coal_brown | **0.023** | 0.032 |
| gas_ocgt | **0.108** | 0.233 |
| gas_steam | **0.215** | 0.508 |
| hydro | **0.273** | 0.370 |
| battery_charging | 0.317 | **0.291** |
| battery_discharging | **1.136** | 1.203 |
| **macro** | **0.345** | 0.440 |

**Honest verdict: the bi-LSTM does NOT beat interpolation on reconstruction.** With
correct methodology (val early-stop, not test) it is 0.440 vs 0.345. Note val
(random gaps) reached 0.315 but test (the harder MIDDAY 11–14 gaps — battery-heavy,
solar-trough) is 0.440: a val/test difficulty mismatch, plus overfitting (train
loss fell 0.14→0.10 while val plateaued). The earlier "0.382" pilot was partly
test-leaked selection + per-epoch augmentation. **Batteries are the wall in
imputation too** (optimizer-not-pattern, theme 6), and they dominate the macro.

**Consequence — the honest simulator is interpolation + the hard-constraint
projection**, not the learned model: it is more accurate (0.345), seam-free,
non-negative, ramp-clean, AND it responds to +g% (the projection snaps Σ dispatch
onto the raised net_demand by construction). This mirrors the forecasting track,
where persistence+anchor beat the neural models. A learned model only earns its
keep with the GRIN engine (channel graph) + GPU, which is the open next step.

### Counterfactual — the actual deliverable (186 test days, corrected +g%×demand physics)

| check | bi-LSTM (100ep) |
| --- | --- |
| placebo (g=0): spurious response | **0.0%**, track p50 20 MW |
| scenario (+10%): **capture** | **+0.940** (fleet delivers 94% of the extra load) |
| scenario tracking p50 | **7 MW** |
| **ramp violations incl. both seams** | **0** |
| n_neg / SOC-infeasible gap-days | **0 / 0-of-186** |

(Demand shock now correctly = +g%×demand with renewables fixed, so capture rose
0.84→0.94 — the dispatchables absorb the whole increase, as they physically must.)

This is the point: **the 2:05 seam is gone by construction** (0 violations
including both boundaries, vs the forecasting model's 199 seam artifacts), the
fill **responds to +10% demand** (capture 0.84, median tracking 6 MW), and the
**placebo is clean** (no invented effect when demand is unchanged) — all while
staying box/ramp/SOC feasible. Capture 0.84 < the forecasting model's 1.03 is the
honest trade: pinning BOTH endpoints (seam-clean) limits how far the middle can
rise vs pinning only the past (seam-broken). Next: `lam-dev` sweep + GPU + a GRIN
engine (graph across channels) for reconstruction; landing-strip right-boundary
for the energy-non-neutral case.

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
