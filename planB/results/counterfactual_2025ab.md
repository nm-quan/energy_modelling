# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 2.4% / reduce 2.4%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| hardnet_2025 | bilstm | hardnet | p99.9 |
| unconstrained_2025 | bilstm | none | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| hardnet_2025 | 0 | 24 | 0 |
| unconstrained_2025 | 360 | 26 | 565 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | hardnet_2025 | unconstrained_2025 | REAL GRID |
| --- | ---: | ---: | ---: |
| hydro | +8,457 (22%) | +344 (7%) | 29.3% |
| coal | +10,321 (26%) | +3,553 (72%) | 43.4% |
| gas steam | +5,138 (13%) | +148 (3%) | 3.7% |
| gas OCGT | +5,648 (14%) | +199 (4%) | 9.7% |
| battery in (load) | -3,963 (10%) | -454 (9%) | 6.6% |
| battery out | +5,703 (15%) | +240 (5%) | 7.4% |
| **Σ signed** | +39,230 | +4,939 | needs +39,230 |

## Response, masked (MWh in the window, share of the total)

| channel | hardnet_2025 | unconstrained_2025 | REAL GRID |
| --- | ---: | ---: | ---: |
| hydro | +8,253 (21%) | +337 (7%) | 29.3% |
| coal | +11,506 (29%) | +3,534 (72%) | 43.4% |
| gas steam | +4,585 (12%) | +150 (3%) | 3.7% |
| gas OCGT | +4,683 (12%) | +198 (4%) | 9.7% |
| battery in (load) | -5,348 (14%) | -452 (9%) | 6.6% |
| battery out | +4,855 (12%) | +234 (5%) | 7.4% |
| **Σ signed** | +39,230 | +4,905 | needs +39,230 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| hardnet_2025 | 10,693 | 12,223 | 6,582 | -5,641 |
| unconstrained_2025 | 10,693 | 10,126 | 5,748 | -4,377 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

