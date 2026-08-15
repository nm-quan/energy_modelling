# Full test-set counterfactual — 186 days, 2026-01-01 .. 2026-07-05

Free window 11:00-14:00, rebound 2.4% / reduce 2.4%. Outside the window every fuel is scaled uniformly by `nd_cf/nd_base`; inside it the model fills. `project_full_day` (cyclic_project, 400 iters, SOC on) is applied over all 288 steps of every day.

Demand rises by **+617,114 MWh** inside the free window and falls outside it, netting **+63,562 MWh** over the whole day. The window figure is what the fleet must actually re-dispatch.

## Energy totals — real vs counterfactual (whole day, all 288 steps)

| fuel | real (MWh) | uniform (MWh) | % | hardnet_2025_alloc_sup (MWh) | % |
| --- | ---: | ---: | ---: | ---: | ---: |
| hydro | 1,317,140 | 1,290,472 | -2.02% | 1,511,544 | +14.76% |
| coal | 16,312,465 | 16,511,239 | +1.22% | 16,082,476 | -1.41% |
| gas steam | 34,975 | 34,715 | -0.74% | 44,629 | +27.60% |
| gas OCGT | 370,477 | 368,146 | -0.63% | 417,662 | +12.74% |
| battery in | 569,128 | 664,080 | +16.68% | 545,108 | -4.22% |
| battery out | 474,699 | 463,700 | -2.32% | 492,988 | +3.85% |
| **net demand served** | 17,940,629 | 18,004,191 | +0.35% | 18,004,191 | +0.35% |
| _gross generation_ | 18,509,757 | 18,668,271 | +0.86% | 18,549,299 | +0.21% |

## Energy totals — real vs counterfactual (free window 11:00-14:00 only)

| fuel | real (MWh) | uniform (MWh) | % | hardnet_2025_alloc_sup (MWh) | % |
| --- | ---: | ---: | ---: | ---: | ---: |
| hydro | 40,595 | 51,159 | +26.02% | 272,232 | +570.60% |
| coal | 1,755,129 | 2,459,001 | +40.10% | 2,030,238 | +15.67% |
| gas steam | 2,558 | 3,223 | +25.99% | 13,137 | +413.51% |
| gas OCGT | 29,088 | 36,188 | +24.41% | 85,704 | +194.63% |
| battery in | 238,088 | 346,597 | +45.58% | 227,626 | -4.39% |
| battery out | 9,679 | 13,103 | +35.37% | 42,391 | +337.98% |
| **net demand served** | 1,598,962 | 2,216,077 | +38.59% | 2,216,077 | +38.59% |
| _gross generation_ | 1,837,050 | 2,562,674 | +39.50% | 2,443,702 | +33.02% |

## Total change per fuel, whole test set (MWh, vs the real dispatch)

| fuel | UNIFORM scaling | hardnet_2025_alloc_sup | hardnet_2025_alloc_sup − uniform |
| --- | ---: | ---: | ---: |
| hydro | -26,669 | +194,404 | +221,073 |
| coal | +198,773 | -229,989 | -428,763 |
| gas steam | -260 | +9,654 | +9,914 |
| gas OCGT | -2,332 | +47,184 | +49,516 |
| battery in | +94,952 | -24,019 | -118,971 |
| battery out | -11,000 | +18,289 | +29,289 |
| **Σ signed (= net demand served)** | +63,562 | +63,562 | |

## The same, restricted to the free window

| fuel | UNIFORM scaling | hardnet_2025_alloc_sup | hardnet_2025_alloc_sup − uniform | real-grid share |
| --- | ---: | ---: | ---: | ---: |
| hydro | +10,564 | +231,637 | +221,073 | 29.3% |
| coal | +703,871 | +275,109 | -428,763 | 43.4% |
| gas steam | +665 | +10,579 | +9,914 | 3.7% |
| gas OCGT | +7,100 | +56,616 | +49,516 | 9.7% |
| battery in | +108,509 | -10,462 | -118,971 | 6.6% |
| battery out | +3,424 | +32,712 | +29,289 | 7.4% |
| **Σ signed** | +617,114 | +617,114 | | needs +617,114 |

_Shares of the signed total, for comparison with dispatch_study Study A:_

| allocation | hydro | coal | gas steam | gas OCGT | battery in | battery out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UNIFORM | 1.7% | 114.1% | 0.1% | 1.2% | 17.6% | 0.6% |
| hardnet_2025_alloc_sup | 37.5% | 44.6% | 1.7% | 9.2% | 1.7% | 5.3% |
| REAL GRID (Study A) | 29.3% | 43.4% | 3.7% | 9.7% | 6.6% | 7.4% |

_Shares can exceed 100%. Uniform scaling multiplies the battery CHARGING channel by the same factor as everything else, so when demand rises it charges MORE -- adding load that the generators must then also cover. That is a real defect of the uniform reference, not a reporting artifact: it is what 'scale every fuel to match the new net demand' does to a channel whose sign is negative._
## Battery energy conservation

A battery is a container, not a source: over the test set the stored energy `eta*charge - discharge/eta` must come back to where it started. The real dispatch does exactly that. A SWING bound (max E - min E) cannot see a slow drift, which is why this table, not the feasibility table, is the one that catches an impossible fill.

| case | charge (MWh) | discharge (MWh) | stored ΔE | per day | % of reservoir/day |
| --- | ---: | ---: | ---: | ---: | ---: |
| REAL dispatch | 569,128 | 474,699 | -51 | -0 | -0.0% |
| UNIFORM scaling | 664,080 | 463,700 | +98,707 | +531 | +11.2% |
| hardnet_2025_alloc_sup  RAW head output | 545,108 | 492,988 | -42,013 | -226 | -4.8% |
| hardnet_2025_alloc_sup  after project_full_day | 545,108 | 492,988 | -42,013 | -226 | -4.8% |

_Reservoir 4,736 MWh. The real fleet's discharge/charge ratio is 0.8341 against eta^2 = 0.8340, i.e. it cycles exactly._

## Curtailment (MWh over the test set)

Curtailment is a parallel prediction: planB removed the curtailment credit, so it never enters the balance target. Uniform scaling has no opinion on it -- it leaves curtailment at the actual value by construction.

| arm | actual | base-case predicted | counterfactual | Δ (cf − base) | Δ vs actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| hardnet_2025_alloc_sup | 348,058 | 363,041 | 255,606 | -107,435 | -92,452 |

## Feasibility of the RAW filled window, before project_full_day

| arm | SOC in head | balance max | below 0 | above cap | ramp IN window | ramp outside | swing to gap end | over budget | swing whole day |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardnet_2025_alloc_sup | no | 9.02e-04 | 0.00e+00 | 2.44e-05 | 8.03e-04 | 2.45e+02 | 3,350 MWh | 0.0 MWh | 3,563 MWh |

_Day SOC swing is over all 288 steps with the model's window spliced in; budget is 4,536 MWh (reservoir 4,736 less two 100 MWh margins). Uniform-scaling reference for the same days: _
_`ramp outside` is not the model's: outside the free window the day IS the uniform reference, and scaling coal by nd_cf/nd_base scales its ramps too, which pushes them past the envelope. Inside the window, where the head acts, the ramp residual is float32 noise._
_`swing to gap end` is midnight to the close of the free window -- the only stretch the head can bound, since it is seeded with the day's history and controls nothing after 14:00. `swing whole day` includes the evening peak, where the battery does most of its work and the head has no say. Budget 4,536 MWh. For reference the ACTUAL dispatch swings up to 3,762 MWh over the day and the uniform reference up to 5,402 MWh._

