# Ablation — is it the network, or the head plus a leaked net demand?

400 random test 3h gaps, seed 123. Dispatch channels only (curtailment is untouched by the head and is not comparable here).

`interp + head`, `persistence + head` and `zeros + head` contain **zero learned parameters**. They differ from the trained arms only in what direction is handed to the projection.

## MAE (MW)

| arm | hydro | coal | steam | ocgt | bat_chg | bat_dis | **aggregate** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp (no nd) | 63.4 | 85.8 | 2.0 | 15.0 | 64.7 | 72.5 | **50.6** |
| interp + head | 53.3 | 80.9 | 19.9 | 27.7 | 55.2 | 58.0 | **49.2** |
| persistence + head | 93.2 | 132.0 | 41.6 | 54.3 | 92.3 | 81.2 | **82.4** |
| zeros + head | 268.0 | 1,352.2 | 285.4 | 392.8 | 128.6 | 380.4 | **467.9** |
| hardnet (trained, 1,222,664 params) | 105.6 | 168.1 | 14.4 | 53.6 | 86.3 | 78.3 | **84.4** |
| rayen (trained, 1,223,049 params) | 74.4 | 193.6 | 52.1 | 60.4 | 62.7 | 61.5 | **84.1** |
| unconstrained (trained, 1,222,664 params) | 95.6 | 161.4 | 17.7 | 48.9 | 87.3 | 73.6 | **80.7** |

## Balance residual (max |SIGN·P − nd|, MW)

| arm | max balance error |
| --- | ---: |
| interp (no nd) | 1,502.19 |
| interp + head | 0.00 |
| persistence + head | 0.00 |
| zeros + head | 0.00 |
| hardnet (trained, 1,222,664 params) | 0.00 |
| rayen (trained, 1,223,049 params) | 0.00 |
| unconstrained (trained, 1,222,664 params) | 1,370.24 |

## What this says

- `persistence + head`: 82.4 MW, +67.7% vs `interp + head` — **worse than a parameter-free projection**
- `zeros + head`: 467.9 MW, +851.6% vs `interp + head` — **worse than a parameter-free projection**
- `hardnet (trained, 1,222,664 params)`: 84.4 MW, +71.6% vs `interp + head` — **worse than a parameter-free projection**
- `rayen (trained, 1,223,049 params)`: 84.1 MW, +71.0% vs `interp + head` — **worse than a parameter-free projection**
- `unconstrained (trained, 1,222,664 params)`: 80.7 MW, +64.2% vs `interp + head` — **worse than a parameter-free projection**

