# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 2.4% / reduce 2.4%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| saits | saits | hardnet | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| saits | 0 | 24 | 0 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | saits | REAL GRID |
| --- | ---: | ---: |
| hydro | +11,834 (30%) | 29.3% |
| coal | +11,061 (28%) | 43.4% |
| gas steam | +2,743 (7%) | 3.7% |
| gas OCGT | +5,062 (13%) | 9.7% |
| battery in (load) | -5,583 (14%) | 6.6% |
| battery out | +2,947 (8%) | 7.4% |
| **Σ signed** | +39,230 | needs +39,230 |

## Response, masked (MWh in the window, share of the total)

| channel | saits | REAL GRID |
| --- | ---: | ---: |
| hydro | +11,190 (29%) | 29.3% |
| coal | +12,256 (31%) | 43.4% |
| gas steam | +2,238 (6%) | 3.7% |
| gas OCGT | +4,309 (11%) | 9.7% |
| battery in (load) | -6,883 (18%) | 6.6% |
| battery out | +2,355 (6%) | 7.4% |
| **Σ signed** | +39,230 | needs +39,230 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| saits | 10,693 | 12,133 | 10,236 | -1,897 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

