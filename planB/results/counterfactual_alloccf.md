# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 2.4% / reduce 2.4%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| hardnet_2025_base | bilstm | hardnet | p99.9 |
| hardnet_2025_alloc_unsup | bilstm | hardnet_alloc | p99.9 |
| hardnet_2025_alloc_sup | bilstm | hardnet_alloc | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| hardnet_2025_base | 0 | 24 | 0 |
| hardnet_2025_alloc_unsup | 0 | 24 | 0 |
| hardnet_2025_alloc_sup | 0 | 24 | 0 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | hardnet_2025_base | hardnet_2025_alloc_unsup | hardnet_2025_alloc_sup | REAL GRID |
| --- | ---: | ---: | ---: | ---: |
| hydro | +8,457 (22%) | +8,794 (22%) | +7,289 (19%) | 29.3% |
| coal | +10,321 (26%) | +17,350 (44%) | +20,085 (51%) | 43.4% |
| gas steam | +5,138 (13%) | +1,577 (4%) | +348 (1%) | 3.7% |
| gas OCGT | +5,648 (14%) | +1,975 (5%) | +1,257 (3%) | 9.7% |
| battery in (load) | -3,963 (10%) | -4,336 (11%) | -5,662 (14%) | 6.6% |
| battery out | +5,703 (15%) | +5,198 (13%) | +4,588 (12%) | 7.4% |
| **Σ signed** | +39,230 | +39,230 | +39,230 | needs +39,230 |

## Response, masked (MWh in the window, share of the total)

| channel | hardnet_2025_base | hardnet_2025_alloc_unsup | hardnet_2025_alloc_sup | REAL GRID |
| --- | ---: | ---: | ---: | ---: |
| hydro | +8,253 (21%) | +8,227 (21%) | +6,612 (17%) | 29.3% |
| coal | +11,506 (29%) | +18,265 (47%) | +20,568 (52%) | 43.4% |
| gas steam | +4,585 (12%) | +1,333 (3%) | +311 (1%) | 3.7% |
| gas OCGT | +4,683 (12%) | +1,475 (4%) | +1,036 (3%) | 9.7% |
| battery in (load) | -5,348 (14%) | -5,508 (14%) | -6,781 (17%) | 6.6% |
| battery out | +4,855 (12%) | +4,422 (11%) | +3,922 (10%) | 7.4% |
| **Σ signed** | +39,230 | +39,230 | +39,230 | needs +39,230 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| hardnet_2025_base | 10,693 | 12,223 | 6,582 | -5,641 |
| hardnet_2025_alloc_unsup | 10,693 | 10,719 | 5,244 | -5,476 |
| hardnet_2025_alloc_sup | 10,693 | 9,976 | 4,928 | -5,048 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

