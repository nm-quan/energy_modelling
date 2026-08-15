# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 2.4% / reduce 2.4%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| hardnet_2025_base | bilstm | hardnet | p99.9 |
| hardnet_2025_pk30 | bilstm | hardnet | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| hardnet_2025_base | 0 | 24 | 0 |
| hardnet_2025_pk30 | 0 | 24 | 0 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | hardnet_2025_base | hardnet_2025_pk30 | REAL GRID |
| --- | ---: | ---: | ---: |
| hydro | +8,457 (22%) | +8,020 (20%) | 29.3% |
| coal | +10,321 (26%) | +9,803 (25%) | 43.4% |
| gas steam | +5,138 (13%) | +5,344 (14%) | 3.7% |
| gas OCGT | +5,648 (14%) | +6,516 (17%) | 9.7% |
| battery in (load) | -3,963 (10%) | -3,665 (9%) | 6.6% |
| battery out | +5,703 (15%) | +5,882 (15%) | 7.4% |
| **Σ signed** | +39,230 | +39,230 | needs +39,230 |

## Response, masked (MWh in the window, share of the total)

| channel | hardnet_2025_base | hardnet_2025_pk30 | REAL GRID |
| --- | ---: | ---: | ---: |
| hydro | +8,253 (21%) | +7,980 (20%) | 29.3% |
| coal | +11,506 (29%) | +10,711 (27%) | 43.4% |
| gas steam | +4,585 (12%) | +4,877 (12%) | 3.7% |
| gas OCGT | +4,683 (12%) | +5,618 (14%) | 9.7% |
| battery in (load) | -5,348 (14%) | -4,951 (13%) | 6.6% |
| battery out | +4,855 (12%) | +5,092 (13%) | 7.4% |
| **Σ signed** | +39,230 | +39,230 | needs +39,230 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| hardnet_2025_base | 10,693 | 12,223 | 6,582 | -5,641 |
| hardnet_2025_pk30 | 10,693 | 10,085 | 6,133 | -3,952 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

