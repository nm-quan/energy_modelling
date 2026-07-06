# Findings — accuracy (WAPE) vs demand response

Scenario: `FixedPercentageShift(rebound=20%, reduction=10%)`

## One-step teacher-forced (clean, no feedback loop)

| model | R² | WAPE | net_demand | coal | hydro | gas_ocgt | batt_dis |
| ----- | -- | ---- | ---------- | ---- | ----- | -------- | -------- |
| LSTM no-RevIN         | 0.9727 | 0.2024 | +62.1% | +39.8% | +441.1% | −12.6% | +505.3% |
| LSTM +RevIN           | 0.9747 | 0.1252 |  +3.9% |  +1.7% |  +47.2% |  −3.3% |   +8.3% |
| iTransformer +RevIN   | 0.9765 | 0.1132 |  +0.1% |  +0.1% |   +0.9% |  −0.6% |   +3.0% |
| LSTM selective-RevIN  | 0.9583 | 0.1807 |  +5.2% |  +2.4% |  +35.4% |  −0.2% |  +92.1% |



wape of model = wape of coal + wape + hydro + .. /6 





## Soft Constraints ((y-yo) + λ·(Σ SIGN·pred − net_demand)^2)

| λ | R² | WAPE | net_demand | coal | batt_dis |
| - | -- | ---- | ---------- | ---- | -------- |
| 0   | 0.9395 | 0.2503 |  +4.1% | +2.4% |  +24.3% |
| 1   | 0.9373 | 0.2541 |  +5.7% | +2.5% |  +49.9% |
| 3   | 0.9334 | 0.2630 |  +6.1% | +3.1% |  +41.5% |
| 10  | 0.9190 | 0.2964 |  +7.9% | +3.4% | +109.5% |
| 30  | 0.8903 | 0.3544 | +10.8% | +3.5% | +193.1% |
| 100 | 0.8241 | 0.4655 | +11.7% | +3.8% | +226.2% |

## Dish-TS 


| model | R² | WAPE | net_demand | batt_dis |
| ----- | -- | ---- | ---------- | -------- |

| Dish-TS own   | 0.9516 | 0.2880 | +17.7% | +119.3% |
| Dish-TS cross | 0.9418 | 0.2916 |  +3.5% |  +14.5% |
| Dish-TS exo   | 0.5633 | 2.0827 | +34.5% | +456.0% |

## RevIN input su

| perturbation on the lookback window | RevIN retains |
| ----------------------------------- | ------------- |
| localized free-window bump (this shift) | ~75–110% (survives) |
| uniform whole-windowd +85% lift          | ~0% (cancelled) |

## Demand-anchored head (structural: sum(SIGN·pred) == net_demand exactly)

Trained THROUGH the head (full convergence): response +179.2% but WAPE collapsed
to 0.3375 — batt_chg is the only free variable in `need = nd + chg`, so MSE
distorts it into a slack absorber and every channel inherits the error.

INFERENCE-ONLY anchoring (head bolted onto the converged blind checkpoints, no
training — the head is parameter-free): keeps the backbone's accuracy AND the
full structural response. Fixes: rescale set excludes gas_steam (on only 1.7%
of test steps; passthrough, subtracted from `need` — identity still exact) and
ReLU instead of softplus (softplus's +0.7 MW floor on near-zero preds wrecked
gas_steam's tiny WAPE denominator). See sweep_eqnd/infer_anchor*_compare.md.

| model | R² | WAPE | net_demand | coal | hydro | gas_ocgt | batt_dis |
| ----- | -- | ---- | ---------- | ---- | ----- | -------- | -------- |
| iTransformer +clamp (blind baseline) | 0.9767 | 0.1068 | +0.1% | +0.1% | +0.9% | −0.6% | +3.0% |
| **iTransformer +anchor(nosteam,relu)** | 0.9763 | **0.1075** | **+179.2%** | +147.6% | +107.0% | +97.8% | +157.7% |
| LSTM+RevIN +clamp | 0.9748 | 0.1185 | +3.9% | +1.7% | +47.2% | −3.3% | +8.3% |
| LSTM+RevIN +anchor(nosteam,relu) | 0.9746 | 0.1188 | +179.2% | +144.8% | +205.6% | +91.1% | +156.7% |

Responsiveness is free: +0.0007 WAPE vs the clamped blind baseline, and better
than the old unclamped project best (0.1132). Closed loop (ar_free_rollout
nd_mode=scenario, the sweep path): anchored iTransformer no longer collapses —
tracking |Σpred − nd| p50 70 MW vs 341 MW bare / 143 MW prod LSTM, response
+179.2% exact (infer_anchor_cl_compare.md). Caveats: aggregate response is
exact by construction, but the per-channel split is the blind model's mix
rescaled ~proportionally (batt_chg passthrough ⇒ 0% response); the anchor uses
nd(t−1), so vs actual nd(t) the residual is the 5-min nd ramp (p50 53 MW,
p95 198 MW) — anchoring to nd(t) would need a net_demand_next input feature.
