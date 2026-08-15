# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 2.4% / reduce 2.4%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| hardnet | bilstm | hardnet | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| hardnet | 0 | 24 | 0 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | hardnet | REAL GRID |
| --- | ---: | ---: |
| hydro | +9,963 (25%) | 29.3% |
| coal | +11,553 (29%) | 43.4% |
| gas steam | +3,945 (10%) | 3.7% |
| gas OCGT | +4,224 (11%) | 9.7% |
| battery in (load) | -5,185 (13%) | 6.6% |
| battery out | +4,360 (11%) | 7.4% |
| **Σ signed** | +39,230 | needs +39,230 |

## Response, masked (MWh in the window, share of the total)

| channel | hardnet | REAL GRID |
| --- | ---: | ---: |
| hydro | +9,822 (25%) | 29.3% |
| coal | +12,742 (32%) | 43.4% |
| gas steam | +3,276 (8%) | 3.7% |
| gas OCGT | +3,258 (8%) | 9.7% |
| battery in (load) | -6,628 (17%) | 6.6% |
| battery out | +3,504 (9%) | 7.4% |
| **Σ signed** | +39,230 | needs +39,230 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| hardnet | 10,693 | 12,771 | 8,714 | -4,057 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

