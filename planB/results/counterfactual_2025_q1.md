# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 1% / reduce 1%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| hardnet_2025 | bilstm | hardnet | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| hardnet_2025 | 0 | 50 | 0 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | hardnet_2025 | REAL GRID |
| --- | ---: | ---: |
| hydro | +3,730 (23%) | 29.3% |
| coal | +5,952 (36%) | 43.4% |
| gas steam | +1,136 (7%) | 3.7% |
| gas OCGT | +810 (5%) | 9.7% |
| battery in (load) | -3,484 (21%) | 6.6% |
| battery out | +1,234 (8%) | 7.4% |
| **Σ signed** | +16,346 | needs +16,346 |

## Response, masked (MWh in the window, share of the total)

| channel | hardnet_2025 | REAL GRID |
| --- | ---: | ---: |
| hydro | +3,557 (22%) | 29.3% |
| coal | +6,276 (38%) | 43.4% |
| gas steam | +879 (5%) | 3.7% |
| gas OCGT | +596 (4%) | 9.7% |
| battery in (load) | -4,002 (24%) | 6.6% |
| battery out | +1,036 (6%) | 7.4% |
| **Σ signed** | +16,346 | needs +16,346 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| hardnet_2025 | 10,693 | 13,020 | 9,468 | -3,552 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

