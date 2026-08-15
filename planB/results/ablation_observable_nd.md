# What the constraint head costs when net demand is not the labels

400 random test 3h gaps, seed 123, checkpoints from `colab/planB_results`.

`SIGN·dispatch` minus the observable `demand − wind − solar − curtailment`, inside the gap: mean +505 MW, mean absolute 506 MW, max 4,303 MW.

## MAE (MW)

| configuration | hydro | coal | steam | ocgt | bat_chg | bat_dis | **aggregate** | balance vs true nd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardnet — as published (nd = SIGN·truth, in features AND head) | 52.2 | 72.3 | 5.9 | 22.6 | 42.7 | 51.4 | **41.2** | 0.0 |
| hardnet — observable nd in the head only | 61.6 | 238.0 | 4.9 | 23.1 | 203.6 | 58.6 | **98.3** | 3,271.6 |
| hardnet — observable nd in features AND head | 63.3 | 246.3 | 4.9 | 23.2 | 196.6 | 57.7 | **98.7** | 3,271.6 |
| interp + head, nd = SIGN·truth | 53.3 | 80.9 | 19.9 | 27.7 | 55.2 | 58.0 | **49.2** | 0.0 |
| interp + head, observable nd | 58.6 | 228.6 | 11.1 | 21.0 | 239.6 | 55.1 | **102.3** | 3,271.6 |

