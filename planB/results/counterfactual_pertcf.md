# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 2.4% / reduce 2.4%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| hardnet_2025_base | bilstm | hardnet | p99.9 |
| hardnet_2025_pert1 | bilstm | hardnet | p99.9 |
| hardnet_2025_pert10 | bilstm | hardnet | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| hardnet_2025_base | 0 | 24 | 0 |
| hardnet_2025_pert1 | 0 | 24 | 0 |
| hardnet_2025_pert10 | 0 | 24 | 0 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | hardnet_2025_base | hardnet_2025_pert1 | hardnet_2025_pert10 | REAL GRID |
| --- | ---: | ---: | ---: | ---: |
| hydro | +8,457 (22%) | +8,454 (22%) | +7,584 (19%) | 29.3% |
| coal | +10,321 (26%) | +10,781 (27%) | +15,576 (40%) | 43.4% |
| gas steam | +5,138 (13%) | +4,628 (12%) | +2,728 (7%) | 3.7% |
| gas OCGT | +5,648 (14%) | +5,504 (14%) | +3,620 (9%) | 9.7% |
| battery in (load) | -3,963 (10%) | -3,987 (10%) | -4,948 (13%) | 6.6% |
| battery out | +5,703 (15%) | +5,876 (15%) | +4,774 (12%) | 7.4% |
| **Σ signed** | +39,230 | +39,230 | +39,230 | needs +39,230 |

## Response, masked (MWh in the window, share of the total)

| channel | hardnet_2025_base | hardnet_2025_pert1 | hardnet_2025_pert10 | REAL GRID |
| --- | ---: | ---: | ---: | ---: |
| hydro | +8,253 (21%) | +8,197 (21%) | +7,237 (18%) | 29.3% |
| coal | +11,506 (29%) | +12,029 (31%) | +16,447 (42%) | 43.4% |
| gas steam | +4,585 (12%) | +4,030 (10%) | +2,215 (6%) | 3.7% |
| gas OCGT | +4,683 (12%) | +4,545 (12%) | +2,894 (7%) | 9.7% |
| battery in (load) | -5,348 (14%) | -5,402 (14%) | -6,312 (16%) | 6.6% |
| battery out | +4,855 (12%) | +5,027 (13%) | +4,125 (11%) | 7.4% |
| **Σ signed** | +39,230 | +39,230 | +39,230 | needs +39,230 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| hardnet_2025_base | 10,693 | 12,223 | 6,582 | -5,641 |
| hardnet_2025_pert1 | 10,693 | 12,178 | 6,638 | -5,540 |
| hardnet_2025_pert10 | 10,693 | 12,020 | 8,269 | -3,751 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

