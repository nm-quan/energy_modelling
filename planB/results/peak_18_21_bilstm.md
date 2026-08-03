# Peak filling 18:00-21:00 — 185 test days

Gap = 36 steps opening at 18:00, right edge pinned at 21:00. Every test day in 2026-01-01..2026-07-04. Statistics are over all 185 x 36 = 6,660 cells per channel; the figure is the day-average of the same arrays.

## MAE (MW)

| arm | hydro | coal | steam | ocgt | bat_chg | bat_dis | wind_cu | sol_cu | disp agg | curt agg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp | 95.3 | 74.8 | 3.9 | 17.7 | 8.8 | 129.8 | 45.1 | 9.8 | **55.1** | **27.5** |
| unconstrained | 147.4 | 154.2 | 32.0 | 95.5 | 8.3 | 201.4 | 50.7 | 4.0 | **106.5** | **27.4** |
| rayen | 84.7 | 93.4 | 38.4 | 62.0 | 7.8 | 97.5 | 60.2 | 5.8 | **63.9** | **33.0** |
| hardnet | 168.2 | 175.9 | 28.3 | 116.3 | 7.0 | 226.7 | 52.1 | 5.7 | **120.4** | **28.9** |

## MRE (%)

| arm | hydro | coal | steam | ocgt | bat_chg | bat_dis | wind_cu | sol_cu |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp | 14.5 | 1.9 | 18.5 | 10.9 | 123.1 | 36.6 | 54.6 | 218.6 |
| unconstrained | 22.5 | 3.8 | 151.6 | 58.7 | 115.6 | 56.7 | 61.3 | 89.4 |
| rayen | 12.9 | 2.3 | 181.5 | 38.1 | 108.2 | 27.5 | 72.9 | 129.2 |
| hardnet | 25.6 | 4.4 | 134.2 | 71.4 | 97.3 | 63.9 | 63.1 | 127.1 |

_Share of window cells at or below 1 MW: hydro 5%, coal 0%, steam 94%, ocgt 59%, bat_chg 17%, bat_dis 3%, wind_cu 65%, sol_cu 91%._

## Window energy mix (% of gross generation, day-averaged)

| arm | coal | hydro | gas OCGT | gas steam | battery out |
| --- | ---: | ---: | ---: | ---: | ---: |
| actual | 77.1 | 12.6 | 3.1 | 0.4 | 6.8 |
| interp | 79.0 | 11.8 | 3.0 | 0.4 | 5.9 |
| unconstrained | 78.2 | 13.1 | 4.6 | 0.9 | 3.2 |
| rayen | 76.0 | 12.8 | 4.1 | 1.1 | 6.0 |
| hardnet | 77.2 | 14.5 | 4.7 | 0.9 | 2.7 |

_Gross generation excludes battery charging, which is a load._
