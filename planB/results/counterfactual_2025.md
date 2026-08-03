# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 2.4% / reduce 2.4%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| hardnet_2025 | bilstm | hardnet | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| hardnet_2025 | 14 | 50 | 0 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | hardnet_2025 | REAL GRID |
| --- | ---: | ---: |
| hydro | +8,174 (21%) | 29.3% |
| coal | +10,056 (26%) | 43.4% |
| gas steam | +5,519 (14%) | 3.7% |
| gas OCGT | +5,482 (14%) | 9.7% |
| battery in (load) | -4,051 (10%) | 6.6% |
| battery out | +5,482 (14%) | 7.4% |
| **Σ signed** | +38,763 | needs +39,230 |

## Response, masked (MWh in the window, share of the total)

| channel | hardnet_2025 | REAL GRID |
| --- | ---: | ---: |
| hydro | +8,071 (21%) | 29.3% |
| coal | +11,124 (29%) | 43.4% |
| gas steam | +4,890 (13%) | 3.7% |
| gas OCGT | +4,670 (12%) | 9.7% |
| battery in (load) | -5,483 (14%) | 6.6% |
| battery out | +4,736 (12%) | 7.4% |
| **Σ signed** | +38,974 | needs +39,230 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| hardnet_2025 | 10,693 | 13,020 | 7,230 | -5,790 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

