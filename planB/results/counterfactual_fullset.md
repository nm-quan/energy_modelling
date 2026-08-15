# Full test-set counterfactual — 186 days, 2026-01-01 .. 2026-07-05

Free window 11:00-14:00, rebound 2.4% / reduce 2.4%. Outside the window every fuel is scaled uniformly by `nd_cf/nd_base`; inside it the model fills. `project_full_day` (cyclic_project, 400 iters, SOC on) is applied over all 288 steps of every day.

Demand rises by **+617,114 MWh** inside the free window and falls outside it, netting **+63,562 MWh** over the whole day. The window figure is what the fleet must actually re-dispatch.

## Energy totals — real vs counterfactual (whole day, all 288 steps)

| fuel | real (MWh) | uniform (MWh) | % | hardnet_2025_alloc_sup (MWh) | % |
| --- | ---: | ---: | ---: | ---: | ---: |
| hydro | 1,317,140 | 1,290,472 | -2.02% | 1,371,608 | +4.14% |
| coal | 16,312,465 | 16,511,239 | +1.22% | 16,051,536 | -1.60% |
| gas steam | 34,975 | 34,715 | -0.74% | 40,988 | +17.19% |
| gas OCGT | 370,477 | 368,146 | -0.63% | 386,933 | +4.44% |
| battery in | 569,128 | 664,080 | +16.68% | 349,455 | -38.60% |
| battery out | 474,699 | 463,700 | -2.32% | 502,581 | +5.87% |
| **net demand served** | 17,940,629 | 18,004,191 | +0.35% | 18,004,191 | +0.35% |
| _gross generation_ | 18,509,757 | 18,668,271 | +0.86% | 18,353,646 | -0.84% |

## Energy totals — real vs counterfactual (free window 11:00-14:00 only)

| fuel | real (MWh) | uniform (MWh) | % | hardnet_2025_alloc_sup (MWh) | % |
| --- | ---: | ---: | ---: | ---: | ---: |
| hydro | 40,595 | 51,159 | +26.02% | 132,276 | +225.84% |
| coal | 1,755,129 | 2,459,001 | +40.10% | 1,999,278 | +13.91% |
| gas steam | 2,558 | 3,223 | +25.99% | 9,496 | +271.17% |
| gas OCGT | 29,088 | 36,188 | +24.41% | 54,971 | +88.98% |
| battery in | 238,088 | 346,597 | +45.58% | 32,039 | -86.54% |
| battery out | 9,679 | 13,103 | +35.37% | 52,095 | +438.23% |
| **net demand served** | 1,598,962 | 2,216,077 | +38.59% | 2,216,077 | +38.59% |
| _gross generation_ | 1,837,050 | 2,562,674 | +39.50% | 2,248,116 | +22.38% |

## Total change per fuel, whole test set (MWh, vs the real dispatch)

| fuel | UNIFORM scaling | hardnet_2025_alloc_sup | hardnet_2025_alloc_sup − uniform |
| --- | ---: | ---: | ---: |
| hydro | -26,669 | +54,468 | +81,137 |
| coal | +198,773 | -260,929 | -459,702 |
| gas steam | -260 | +6,013 | +6,273 |
| gas OCGT | -2,332 | +16,455 | +18,787 |
| battery in | +94,952 | -219,673 | -314,625 |
| battery out | -11,000 | +27,882 | +38,881 |
| **Σ signed (= net demand served)** | +63,562 | +63,562 | |

## The same, restricted to the free window

| fuel | UNIFORM scaling | hardnet_2025_alloc_sup | hardnet_2025_alloc_sup − uniform | real-grid share |
| --- | ---: | ---: | ---: | ---: |
| hydro | +10,564 | +91,681 | +81,117 | 29.3% |
| coal | +703,871 | +244,149 | -459,722 | 43.4% |
| gas steam | +665 | +6,937 | +6,273 | 3.7% |
| gas OCGT | +7,100 | +25,882 | +18,782 | 9.7% |
| battery in | +108,509 | -206,049 | -314,558 | 6.6% |
| battery out | +3,424 | +42,416 | +38,992 | 7.4% |
| **Σ signed** | +617,114 | +617,114 | | needs +617,114 |

_Shares of the signed total, for comparison with dispatch_study Study A:_

| allocation | hydro | coal | gas steam | gas OCGT | battery in | battery out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UNIFORM | 1.7% | 114.1% | 0.1% | 1.2% | 17.6% | 0.6% |
| hardnet_2025_alloc_sup | 14.9% | 39.6% | 1.1% | 4.2% | 33.4% | 6.9% |
| REAL GRID (Study A) | 29.3% | 43.4% | 3.7% | 9.7% | 6.6% | 7.4% |

_Shares can exceed 100%. Uniform scaling multiplies the battery CHARGING channel by the same factor as everything else, so when demand rises it charges MORE -- adding load that the generators must then also cover. That is a real defect of the uniform reference, not a reporting artifact: it is what 'scale every fuel to match the new net demand' does to a channel whose sign is negative._
## Curtailment (MWh over the test set)

Curtailment is a parallel prediction: planB removed the curtailment credit, so it never enters the balance target. Uniform scaling has no opinion on it -- it leaves curtailment at the actual value by construction.

| arm | actual | base-case predicted | counterfactual | Δ (cf − base) | Δ vs actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| hardnet_2025_alloc_sup | 348,058 | 363,041 | 255,606 | -107,435 | -92,452 |

## Feasibility of the RAW filled window, before project_full_day

| arm | SOC in head | balance max | below 0 | above cap | ramp IN window | ramp outside | swing to gap end | over budget | swing whole day |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardnet_2025_alloc_sup | no | 1.00e-03 | 0.00e+00 | 2.44e-05 | 7.58e-04 | 2.45e+02 | 3,639 MWh | 0.0 MWh | 4,642 MWh |

_Day SOC swing is over all 288 steps with the model's window spliced in; budget is 4,536 MWh (reservoir 4,736 less two 100 MWh margins). Uniform-scaling reference for the same days: _
_`ramp outside` is not the model's: outside the free window the day IS the uniform reference, and scaling coal by nd_cf/nd_base scales its ramps too, which pushes them past the envelope. Inside the window, where the head acts, the ramp residual is float32 noise._
_`swing to gap end` is midnight to the close of the free window -- the only stretch the head can bound, since it is seeded with the day's history and controls nothing after 14:00. `swing whole day` includes the evening peak, where the battery does most of its work and the head has no say. Budget 4,536 MWh. For reference the ACTUAL dispatch swings up to 3,762 MWh over the day and the uniform reference up to 5,402 MWh._

