# Full test-set counterfactual — 4 days, 2026-01-01 .. 2026-01-04

Free window 11:00-14:00, rebound 2.4% / reduce 2.4%. Outside the window every fuel is scaled uniformly by `nd_cf/nd_base`; inside it the model fills. `project_full_day` (cyclic_project, 400 iters, SOC on) is applied over all 288 steps of every day.

Demand moved in the window by **+63,562 MWh** net over the test set (rebound and reduction do not cancel exactly because the window's demand level differs from the rest of the day).

## Total change per fuel, whole test set (MWh, vs the real dispatch)

| fuel | UNIFORM scaling | hardnet_2025_alloc_sup | model − uniform |
| --- | ---: | ---: | ---: |
| hydro | -174 | -10 | +164 |
| coal | +6,008 | -6,121 | -12,129 |
| gas steam | +0 | +0 | +0 |
| gas OCGT | +0 | +240 | +240 |
| battery in | +4,816 | -6,888 | -11,704 |
| battery out | -324 | -303 | +21 |
| **Σ signed (= net demand served)** | +693 | +693 | |

## The same, restricted to the free window

| fuel | UNIFORM scaling | hardnet_2025_alloc_sup | model − uniform | real-grid share |
| --- | ---: | ---: | ---: | ---: |
| hydro | +1 | +165 | +164 | 29.3% |
| coal | +15,144 | +3,015 | -12,129 | 43.4% |
| gas steam | +0 | +0 | +0 | 3.7% |
| gas OCGT | +0 | +240 | +240 | 9.7% |
| battery in | +5,113 | -6,590 | -11,704 | 6.6% |
| battery out | +78 | +99 | +21 | 7.4% |

_Shares of the signed total, for comparison with dispatch_study Study A:_

| allocation | hydro | coal | gas steam | gas OCGT | battery in | battery out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UNIFORM | 0.0% | 149.8% | 0.0% | 0.0% | 50.6% | 0.8% |
| hardnet_2025_alloc_sup | 1.6% | 29.8% | 0.0% | 2.4% | 65.2% | 1.0% |
| REAL GRID (Study A) | 29.3% | 43.4% | 3.7% | 9.7% | 6.6% | 7.4% |

## Curtailment (MWh over the test set)

Curtailment is a parallel prediction: planB removed the curtailment credit, so it never enters the balance target. Uniform scaling has no opinion on it -- it leaves curtailment at the actual value by construction.

| arm | actual | base-case predicted | counterfactual | Δ (cf − base) | Δ vs actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| hardnet_2025_alloc_sup | 16,659 | 14,786 | 11,824 | -2,962 | -4,835 |

## Feasibility of the RAW filled window, before project_full_day

| arm | SOC in head | balance max | below 0 | above cap | ramp | day SOC swing max | over budget |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hardnet_2025_alloc_sup | no | 2.21e-04 | 0.00e+00 | 0.00e+00 | 3.13e+01 | 3,024 MWh | 0.0 MWh |

_Day SOC swing is over all 288 steps with the model's window spliced in; the budget is 4,536 MWh. For reference the ACTUAL dispatch swings up to 3,762 MWh and the uniform reference up to 5,402 MWh on the same days._

