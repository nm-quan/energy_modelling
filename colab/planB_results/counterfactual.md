# planB counterfactual - 10 days, planA protocol

Days 2026-06-01 .. 2026-06-10. Free window 11:00-14:00, rebound 2.4% / reduce 2.4%, both modes, context = the mode's own off-window dispatch, `project_full_day` over all 288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.

| arm | backbone | head | envelope |
| --- | --- | --- | --- |
| unconstrained | BiLSTM | none | p99.9 |
| rayen | BiLSTM | rayen | p99.9 |
| hardnet | BiLSTM | hardnet | p99.9 |
| brits | BRITS | hardnet | p99.9 |

## Feasibility of the raw filled window (before `project_full_day`)

| arm | balance > 1 MW | ramp | negative |
| --- | ---: | ---: | ---: |
| unconstrained | 360 | 69 | 117 |
| rayen | 15 | 50 | 0 |
| hardnet | 14 | 50 | 0 |
| brits | 14 | 50 | 0 |

_360 window steps x 6 channels = 2160 cells; ramp counted over the full day._

## Response, scaled (MWh in the window, share of the total)

| channel | unconstrained | rayen | hardnet | brits | REAL GRID |
| --- | ---: | ---: | ---: | ---: | ---: |
| hydro | +5,007 (16%) | +11,215 (30%) | +7,247 (19%) | +11,606 (31%) | 29.3% |
| coal | +22,798 (73%) | +12,473 (34%) | +22,161 (60%) | +11,817 (32%) | 43.4% |
| gas steam | +446 (1%) | +1,273 (3%) | +716 (2%) | +1,443 (4%) | 3.7% |
| gas OCGT | +969 (3%) | +3,419 (9%) | +3,173 (9%) | +4,093 (11%) | 9.7% |
| battery in (load) | -1,264 (4%) | -2,664 (7%) | -1,890 (5%) | -4,541 (12%) | 6.6% |
| battery out | +680 (2%) | +6,123 (16%) | +1,985 (5%) | +3,677 (10%) | 7.4% |
| **Σ signed** | +31,164 | +37,166 | +37,172 | +37,178 | needs +37,558 |

## Response, masked (MWh in the window, share of the total)

| channel | unconstrained | rayen | hardnet | brits | REAL GRID |
| --- | ---: | ---: | ---: | ---: | ---: |
| hydro | +4,791 (15%) | +10,620 (28%) | +6,988 (19%) | +11,948 (32%) | 29.3% |
| coal | +22,805 (74%) | +12,895 (35%) | +22,744 (61%) | +12,450 (33%) | 43.4% |
| gas steam | +439 (1%) | +1,234 (3%) | +613 (2%) | +1,105 (3%) | 3.7% |
| gas OCGT | +975 (3%) | +3,234 (9%) | +3,141 (8%) | +3,768 (10%) | 9.7% |
| battery in (load) | -1,293 (4%) | -3,809 (10%) | -2,122 (6%) | -4,831 (13%) | 6.6% |
| battery out | +638 (2%) | +5,560 (15%) | +1,751 (5%) | +3,257 (9%) | 7.4% |
| **Σ signed** | +30,941 | +37,352 | +37,358 | +37,360 | needs +37,558 |

## Curtailment in the window (MWh) - a PARALLEL prediction

planB removed planA's curtailment credit, so curtailment does not enter the balance target and cannot move energy into or out of the dispatch. These are the models' predictions of what was available and spilled.

| arm | actual | base-case predicted | counterfactual | delta |
| --- | ---: | ---: | ---: | ---: |
| unconstrained | 10,693 | 9,032 | 7,290 | -1,742 |
| rayen | 10,693 | 11,206 | 7,416 | -3,789 |
| hardnet | 10,693 | 6,785 | 5,642 | -1,143 |
| brits | 10,693 | 4,684 | 5,091 | +407 |

_A negative delta means the model spills less under the higher-demand counterfactual, which is the physically expected direction even though nothing forces it here._

