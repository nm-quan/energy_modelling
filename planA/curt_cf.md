# Counterfactual over 10 days — planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00–14:00, rebound 2.4% / reduce 2.4%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. Both models trained **1 epoch** and both predict wind and solar curtailment.

## Feasibility

Raw = the filled day straight out of the model. Projected = after planA's `project_full_day`. Balance is audited against the target each arm used.

| mode | arm | raw bal>1MW | raw ramp | raw neg | projected bal/ramp/neg/SOC |
| --- | --- | ---: | ---: | ---: | ---: |
| scaled | plain | 360 | 7 | 365 | 0 / 0 / 0 / 0 |
| scaled | recursive | 0 | 0 | 0 | 0 / 0 / 0 / 0 |
| masked | plain | 360 | 7 | 357 | 0 / 0 / 0 / 0 |
| masked | recursive | 0 | 0 | 0 | 0 / 0 / 0 / 0 |

_Window cells: 360 steps x 6 channels; full day 2880 steps._

## Counterfactual response, scaled (MWh in the window)

| channel | plain | recursive | actual-grid evening turn-up |
| --- | ---: | ---: | ---: |
| hydro | +3,349 (13%) | +5,956 (16%) | 30.2% |
| coal | +21,355 (81%) | +18,583 (50%) | 40.4% |
| gas steam | +3 (0%) | +549 (1%) | 4.9% |
| gas OCGT | +551 (2%) | +3,052 (8%) | 11.1% |
| battery in (load) | -798 (3%) | -6,008 (16%) | 7.4% |
| battery out | +351 (1%) | +3,355 (9%) | 6.0% |
| **Σ signed** | +26,407 | +37,503 | — |
| _demand rise to be met_ | _+37,558_ | _+37,558_ | — |

## Counterfactual response, masked (MWh in the window)

| channel | plain | recursive | actual-grid evening turn-up |
| --- | ---: | ---: | ---: |
| hydro | +3,211 (12%) | +5,102 (14%) | 30.2% |
| coal | +21,083 (81%) | +19,682 (52%) | 40.4% |
| gas steam | +6 (0%) | +458 (1%) | 4.9% |
| gas OCGT | +567 (2%) | +2,677 (7%) | 11.1% |
| battery in (load) | -819 (3%) | -6,923 (18%) | 7.4% |
| battery out | +341 (1%) | +2,661 (7%) | 6.0% |
| **Σ signed** | +26,026 | +37,503 | — |
| _demand rise to be met_ | _+37,558_ | _+37,558_ | — |

## Curtailment (MWh in the window, masked mode)

| channel | ACTUAL | plain base → cf | Δ | recursive base → cf | Δ |
| --- | ---: | --- | ---: | --- | ---: |
| wind curtailment | **9,088** | 1,843 → 1,221 | -622 | 119 → 94 | -25 |
| solar curtailment | **1,605** | 1,799 → 1,191 | -607 | 186 → 155 | -31 |

_Worst curtailment credit applied to the recursive balance target: 1,732 MW._

**The coupling suppressed the curtailment prediction.** The plain arm predicts 3,642 MWh
of curtailment against an actual 10,693; the recursive arm predicts only 305. In the
recursive arm curtailment feeds the balance target, so a large curtailment prediction
immediately moves the dispatch it is scored on — after one epoch the easiest way to
avoid that penalty is to keep the prediction near zero. Making curtailment *mean*
something also made it *harder to learn*, which is a real cost of the coupling and not
something more training is guaranteed to fix.

## Peak-fill extra — same protocol, window on the evening peak

`dispatch_study/` shows the evening turn-up is 1,997 MW against 314 MW in the morning, and midday has no battery-discharge signal (corr +0.035 vs +0.287). Same planA masked protocol, window moved to 17:00–20:00.

| channel | plain | recursive | actual-grid evening turn-up |
| --- | ---: | ---: | ---: |
| hydro | +8,560 (44%) | +13,087 (35%) | 30.2% |
| coal | +2,981 (15%) | +5,088 (14%) | 40.4% |
| gas steam | +1,481 (8%) | +1,168 (3%) | 4.9% |
| gas OCGT | +4,348 (22%) | +5,903 (16%) | 11.1% |
| battery in (load) | -177 (1%) | -399 (1%) | 7.4% |
| battery out | +1,818 (9%) | +11,899 (32%) | 6.0% |
| **Σ signed** | +19,366 | +37,544 | — |
| _demand rise to be met_ | _+37,558_ | _+37,558_ | — |

