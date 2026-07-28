# MAE / MRE per output channel — 400 random test 3h gaps (seed 123)

Both models trained **1 epoch**. Channels 1–6 are dispatch, 7–8 are the new curtailment outputs.

`recursive` is the as-trained map, whose balance target `nd_eff = nd − (curt_actual − curt_pred)` uses the TRUE in-gap curtailment. That is fine for a counterfactual but is **leakage** when scoring imputation, since curtailment is masked out of the inputs. `recursive*` is the same weights with the credit removed. The gap between those two rows is the leak.

## MAE (MW)

| model | hydro | coal | gas_steam | gas_ocgt | batt_chg | batt_dis | wind_curt | solar_curt | disp agg | curt agg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp | 63.4 | 85.8 | 2.0 | 15.0 | 64.7 | 72.5 | 61.2 | 24.7 | **50.6** | **43.0** |
| plain | 102.9 | 180.8 | 20.9 | 51.4 | 92.2 | 80.6 | 53.0 | 35.8 | **88.2** | **44.4** |
| recursive | 60.6 | 109.8 | 15.8 | 37.3 | 54.7 | 56.2 | 57.2 | 30.0 | **55.7** | **43.6** |
| recursive* | 57.6 | 97.2 | 14.9 | 35.5 | 50.6 | 55.8 | 57.2 | 30.0 | **51.9** | **43.6** |

## MRE (%)  =  100 · Σ|err| / Σ|truth|

| model | hydro | coal | gas_steam | gas_ocgt | batt_chg | batt_dis | wind_curt | solar_curt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp | 20.7 | 2.4 | 21.6 | 15.0 | 47.7 | 62.8 | 25.6 | 31.0 |
| plain | 33.6 | 5.0 | 226.2 | 51.3 | 68.0 | 69.8 | 22.1 | 44.8 |
| recursive | 19.8 | 3.0 | 170.9 | 37.2 | 40.3 | 48.7 | 23.9 | 37.7 |
| recursive* | 18.8 | 2.7 | 161.8 | 35.4 | 37.3 | 48.3 | 23.9 | 37.7 |

_MRE is unstable where the truth is mostly zero — the denominator collapses. Share of gap cells at or below 1 MW: hydro 33%, coal 0%, gas_steam 96%, gas_ocgt 75%, batt_chg 7%, batt_dis 30%, wind_curt 56%, solar_curt 66%._

## Feasibility of the filled window

| model | worst balance resid (MW) | ramp overshoot (MW) | most-negative (MW) |
| --- | ---: | ---: | ---: |
| interp | 1,502.19 | 0.00 | -0.00 |
| plain | 1,552.05 | 579.80 | -133.51 |
| recursive | 1,005.33 | 0.00 | 0.00 |
| recursive* | 0.00 | 0.00 | 0.00 |

_`recursive` is audited against nd; its residual is the curtailment credit, not a solver error. `recursive*` has no credit, so its residual is the head's true balance error._

