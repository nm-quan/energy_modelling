# planB — per-channel MAE / MRE, 400 random test 3h gaps (seed 123)

All arms: 1 epoch, hidden 192, one MSE over 8 z-scored channels, relu in z-space for curtailment, split output heads, no curtailment credit, **p99 ramp envelope**.

| arm | backbone | head | params |
| --- | --- | --- | ---: |
| interp | — | — | 0 |
| unconstrained | bilstm | none | 1,222,664 |
| rayen | bilstm | rayen | 1,223,049 |
| hardnet | bilstm | hardnet | 1,222,664 |
| brits | brits | hardnet | 382,704 |
| saits | saits | hardnet | 1,213,197 |

## MAE (MW)

| arm | hydro | coal | steam | ocgt | bat_chg | bat_dis | wind_cu | sol_cu | disp agg | curt agg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp | 63.4 | 85.8 | 2.0 | 15.0 | 64.7 | 72.5 | 61.2 | 24.7 | **50.6** | **43.0** |
| unconstrained | 95.6 | 161.4 | 17.7 | 48.9 | 87.3 | 73.6 | 108.8 | 45.0 | **80.7** | **76.9** |
| rayen | 75.1 | 116.0 | 15.5 | 35.5 | 45.8 | 57.4 | 89.7 | 40.9 | **57.5** | **65.3** |
| hardnet | 52.2 | 72.4 | 6.0 | 22.5 | 42.8 | 51.4 | 80.0 | 39.6 | **41.2** | **59.8** |
| brits | 156.7 | 234.0 | 46.3 | 87.4 | 84.4 | 69.7 | 167.9 | 72.5 | **113.1** | **120.2** |
| saits | 57.3 | 80.1 | 3.1 | 20.2 | 51.0 | 52.6 | 55.3 | 25.1 | **44.1** | **40.2** |

## MRE (%)

| arm | hydro | coal | steam | ocgt | bat_chg | bat_dis | wind_cu | sol_cu |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp | 20.7 | 2.4 | 21.6 | 15.0 | 47.7 | 62.8 | 25.6 | 31.0 |
| unconstrained | 31.2 | 4.4 | 191.6 | 48.8 | 64.3 | 63.7 | 45.5 | 56.4 |
| rayen | 24.5 | 3.2 | 167.7 | 35.4 | 33.8 | 49.7 | 37.5 | 51.2 |
| hardnet | 17.0 | 2.0 | 65.2 | 22.5 | 31.6 | 44.5 | 33.4 | 49.6 |
| brits | 51.1 | 6.4 | 501.6 | 87.2 | 62.2 | 60.3 | 70.1 | 90.9 |
| saits | 18.7 | 2.2 | 34.1 | 20.1 | 37.6 | 45.5 | 23.1 | 31.5 |

_MRE is unreliable where the truth is mostly zero. Share of gap cells at or below 1 MW: hydro 33%, coal 0%, steam 96%, ocgt 75%, bat_chg 7%, bat_dis 30%, wind_cu 56%, sol_cu 66%._

## Feasibility against the p99.9 envelope

| arm | balance (MW) | ramp overshoot (MW) | below zero (MW) | above cap (MW) |
| --- | ---: | ---: | ---: | ---: |
| interp | 1,502.19 | 0.00 | 0.00 | 0.00 |
| unconstrained | 1,370.24 | 877.53 | 114.92 | 6.46 |
| rayen | 0.00 | 0.00 | 0.00 | 0.00 |
| hardnet | 62.24 | 0.00 | 0.00 | 0.00 |
| brits | 63.13 | 0.00 | 0.00 | 0.00 |
| saits | 62.48 | 0.00 | 0.00 | 0.00 |

_`interp` is a straight line between the pinned boundaries, so its per-step change is (pR-pL)/37 and it cannot break a ramp limit unless bridge feasibility fails -- which it does not on any of these windows. The constrained arms' balance residual is the shortfall where nd is not attainable inside the tightened box, not a solver error; the feasibility floor at p99.9 is 0.21 MW MAE and 12.8 MW worst balance._

