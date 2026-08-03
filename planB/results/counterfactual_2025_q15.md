# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 1.5% / reduce 1.5%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| hardnet_2025 | bilstm | hardnet | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| hardnet_2025 | 0 | 24 | 0 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | hardnet_2025 | REAL GRID |
| --- | ---: | ---: |
| hydro | +5,548 (23%) | 29.3% |
| coal | +7,931 (32%) | 43.4% |
| gas steam | +2,644 (11%) | 3.7% |
| gas OCGT | +2,190 (9%) | 9.7% |
| battery in (load) | -3,639 (15%) | 6.6% |
| battery out | +2,566 (10%) | 7.4% |
| **Σ signed** | +24,519 | needs +24,519 |

## Response, masked (MWh in the window, share of the total)

| channel | hardnet_2025 | REAL GRID |
| --- | ---: | ---: |
| hydro | +5,386 (22%) | 29.3% |
| coal | +8,619 (35%) | 43.4% |
| gas steam | +2,208 (9%) | 3.7% |
| gas OCGT | +1,646 (7%) | 9.7% |
| battery in (load) | -4,522 (18%) | 6.6% |
| battery out | +2,138 (9%) | 7.4% |
| **Σ signed** | +24,519 | needs +24,519 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| hardnet_2025 | 10,693 | 12,223 | 7,892 | -4,332 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

