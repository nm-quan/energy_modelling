# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 2.4% / reduce 2.4%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| unconstrained | bilstm | none | p99.9 |
| rayen | bilstm | rayen | p99.9 |
| hardnet | bilstm | hardnet | p99.9 |
| brits | brits | hardnet | p99.9 |
| saits | saits | hardnet | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| unconstrained | 358 | 45 | 56 |
| rayen | 0 | 24 | 0 |
| hardnet | 0 | 24 | 0 |
| brits | 0 | 24 | 0 |
| saits | 0 | 24 | 0 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | unconstrained | rayen | hardnet | brits | saits | REAL GRID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hydro | +6,389 (17%) | +11,280 (29%) | +9,963 (25%) | +11,522 (29%) | +11,834 (30%) | 29.3% |
| coal | +26,455 (71%) | +10,014 (26%) | +11,553 (29%) | +12,818 (33%) | +11,061 (28%) | 43.4% |
| gas steam | +355 (1%) | +660 (2%) | +3,945 (10%) | +2,090 (5%) | +2,743 (7%) | 3.7% |
| gas OCGT | +1,320 (4%) | +4,718 (12%) | +4,224 (11%) | +4,973 (13%) | +5,062 (13%) | 9.7% |
| battery in (load) | -1,695 (5%) | -7,164 (18%) | -5,185 (13%) | -4,251 (11%) | -5,583 (14%) | 6.6% |
| battery out | +1,200 (3%) | +5,393 (14%) | +4,360 (11%) | +3,576 (9%) | +2,947 (8%) | 7.4% |
| **Σ signed** | +37,413 | +39,230 | +39,230 | +39,230 | +39,230 | needs +39,230 |

## Response, masked (MWh in the window, share of the total)

| channel | unconstrained | rayen | hardnet | brits | saits | REAL GRID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hydro | +6,127 (16%) | +10,364 (26%) | +9,822 (25%) | +11,891 (30%) | +11,190 (29%) | 29.3% |
| coal | +26,596 (71%) | +10,128 (26%) | +12,742 (32%) | +13,435 (34%) | +12,256 (31%) | 43.4% |
| gas steam | +341 (1%) | +618 (2%) | +3,276 (8%) | +1,688 (4%) | +2,238 (6%) | 3.7% |
| gas OCGT | +1,276 (3%) | +4,352 (11%) | +3,258 (8%) | +4,505 (11%) | +4,309 (11%) | 9.7% |
| battery in (load) | -1,741 (5%) | -9,356 (24%) | -6,628 (17%) | -4,695 (12%) | -6,883 (18%) | 6.6% |
| battery out | +1,130 (3%) | +4,413 (11%) | +3,504 (9%) | +3,017 (8%) | +2,355 (6%) | 7.4% |
| **Σ signed** | +37,210 | +39,230 | +39,230 | +39,230 | +39,230 | needs +39,230 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| unconstrained | 10,693 | 9,184 | 7,520 | -1,664 |
| rayen | 10,693 | 12,286 | 7,463 | -4,823 |
| hardnet | 10,693 | 12,771 | 8,714 | -4,057 |
| brits | 10,693 | 6,099 | 5,808 | -292 |
| saits | 10,693 | 12,133 | 10,236 | -1,897 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

