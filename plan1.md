# Plan 1 — BiLSTM+RAYEN imputers retrained with renewables features, 3 benchmark arms, counterfactual

## 1. Data (local — preprocessing stays on this machine)

New feature set = current 17 + 4 renewables columns → **21 features**:
`wind`, `solar_utility` (NOT solar_rooftop), `wind_curtailment`, `solar_curtailment`.

- wind / solar_utility: already in `data/preprocessed/hist/5min/net_dispatch_totdem/table.parquet` (full history).
- curtailment: `data/vic_curtailment_hist.parquet` (pulled 2026-07-22 via `script/pull_curtailment_hist.py`, /market endpoint, pro key). 500,832 rows, 2021-10-01→2026-07-05, aligns 1:1 with the table. **Clip negative values to 0** (API artifact: wind min −294 MW).
- `net_demand` **feature** is redefined demand-side (user decision): `nd = demand_mw − wind − solar_utility − wind_curtailment − solar_curtailment` — actual renewables and curtailment, in training data and counterfactual alike (the model is fed this nd, and curt=0 in the free window).
- **Balance target of the constraint map** stays the supply-side sum (Σ SIGN·dispatch; counterfactual = supply base + the §6 deltas). Measured reason (`plan1_data.py`): Σ dispatch − nd_demand-side averages ±500 MW (max 4.9 GW, interconnector); balancing the 6 fuels to it gives a 97 MW-MAE feasibility floor for a *perfect* model and unreachable gaps (resid up to 2.3 GW). Feature = user's nd; map target = feasible sum.
- Rebuild flats (`prepared.npz`) with the 21 features; splits unchanged (test = 2026-01-01→). Commit the npz so Colab can clone-and-train.
- Constraints: canon full-history `CAPS`/`RAMPS` from `ml/check_caps.py` (CONSTRAINT.md).

## 2. Models (4)

| name | what changes vs baseline |
|---|---|
| **baseline** | BiLSTM + RAYEN (`rayen_traj`), retrained on the 21-feature data, 3h gaps |
| **cost-loss** | baseline + λ_cost · dispatch-cost term (§4 prices; battery charging enters with a **negative sign** — signed-volume convention of the price table — so cycling isn't punished) |
| **size-aware layer** | baseline with the balance correction made size-aware: replace the equal `/6` split (in `_balance_project` + the RAYEN tangent removal) with a level-proportional split, w_i ∝ P_i |
| **optimization** (trained through) | QP projection replaces RAYEN: min ‖P − fill‖² s.t. balance ∩ ramp ∩ box over the 36-step gap, differentiable (cvxpylayers/OptNet). **Runs LAST — own section after everything incl. the counterfactual — in case it is slow.** |

## 3. Training

- Colab, CUDA. New notebook: `colab/imputation/plan1_training.ipynb` — training only; data prep stays local. Ends with zip-and-download of weights/history so everything is **saved back to the local machine** (`imputation/results/plan1/`).
- Epochs **30**, early stoppage patience **10**. Batch 128, AdamW lr 1e-3 wd 1e-5, seed 0, n_train 40k 3h-gap windows (existing `train.py` recipe otherwise).
- **Train loss and validation loss are both MSE** (early stop on val MSE).
- Persist per-epoch train/val history to JSON (not just notebook prints) and **show the loss curves after training** — one figure, all arms.

## 4. Benchmark

Metrics, one harness, same test gap set for all 4 models:

- **MAE** (MW) — per channel + aggregate
- **MRE (%)** — per channel + aggregate (Σ|err|/Σ|truth|; per-channel gas_steam will be volatile, aggregate is the headline)
- **Dispatch cost ($)** — Σ_i c_i·E_i with battery charging signed negative; this is the axis cost-loss can win on (it will lose MAE/MRE by construction)
- Feasibility counters (balance/ramp/neg/SOC) — must be 0 for every mapped output

Prices ($/MWh, capture-price table; renewables listed for reference — they are inputs, not dispatched):

| channel | $/MWh | | reference | $/MWh |
|---|---|---|---|---|
| battery_charging | 17.65 (signed −) | | solar_rooftop | 11.76 |
| coal_brown | 62.39 | | solar_utility | 22.75 |
| battery_discharging | 107.16 | | wind | 35.16 |
| hydro | 121.77 | | | |
| gas_ocgt | 151.85 | | | |
| gas_steam | 211.18 | | | |

Implied merit order for cost-loss: coal → battery → hydro → OCGT → gas_steam.

## 5. Deliverable A — actual test set (per model, 4 models)

Stacked **actual vs predicted** graph: 4 days, one random 3h gap per day, the predicted (gap) segment highlighted. Stack includes wind/solar/curtailment layers (repo `stack_plots` convention).

## 6. Counterfactual — rebound 2%, reduce 2%

Shift: FixedPercentageShift(reb=2, red=2), free window 11:00–14:00, price→0 in window.

Net-demand construction ("curtailment is free" recipe, applied as deltas on the supply-side base so the base stays exactly feasible):

- **Reduced (off-window) intervals:** Δnd = Δtotal-demand (renewables + curtailments unchanged, curtailment features stay actual).
- **Free intervals:** nd is computed with the actual curtailment credited as free — Δnd = Δtotal-demand − curtailment_actual — and the model is then fed **curtailment features = 0** in the window (wind/solar unchanged). I.e. the worked example: nd = demand − wind − solar − curt = 2500 is what the model sees, alongside curt = 0.

Deliverable B (per model): stacked graphs —

1. **actual** stacked graph
2. **scaled before and after** — every dispatch source outside the free interval scaled per-step by nd_after/nd_before (demand ↓3% ⇒ nd ↓ by Δdemand ⇒ each source × nd_after/nd_before); only the free window is masked and model-filled (3h gap fill)
3. **actual + masked window** — off-window stays the actual dispatch; only the 3h free window is model predictions

Known limit (measured): on high-curtailment days the credit makes the free window *provably* infeasible at the seam — e.g. 2026-01-17, curt@11:00 = 2.3 GW ⇒ certificate residual 193 MW on 1 cell (the fleet cannot ramp down that fast from the pinned 10:55 boundary). The map leaves that one cell short and the figure scripts report it; all other cells are exact.

## 7. Order of execution

1. Local: merge curtailment → rebuild 21-feature flats → commit npz
2. Colab: train baseline, cost-loss, size-aware (30 ep / pat 10) → download weights + history
3. Local: benchmark table (MAE, MRE, $, feasibility) + loss curves + Deliverable A
4. Local: counterfactual runs + Deliverable B
5. **Separate final section:** the optimization (QP trained-through) arm — train, add its row to the benchmark, its Deliverable A/B figures

## 8. Fixed knobs

- **λ_cost = 0.1** (user-set), cost term normalized by 1e5 ($100/MWh × 1000 MW reference) so it is O(1) next to the MSE.
- Full price table (all sources incl. renewables) saved to `data/fuel_costs.json`; cost-loss uses the 6 dispatch entries with battery_charging signed negative.
- QP arm: SOC left to the posthoc map (3h gaps rarely bind SOC); balance/ramp/box inside the QP.
