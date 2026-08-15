# Peak filling 18:00-21:00 — 185 test days

Gap = 36 steps opening at 18:00, right edge pinned at 21:00. Every test day in 2026-01-01..2026-07-04. Statistics are over all 185 x 36 = 6,660 cells per channel; the figure is the day-average of the same arrays.

## MAE (MW)

| arm | hydro | coal | steam | ocgt | bat_chg | bat_dis | wind_cu | sol_cu | disp agg | curt agg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp | 95.3 | 74.8 | 3.9 | 17.7 | 8.8 | 129.8 | 45.1 | 9.8 | **55.1** | **27.5** |
| hardnet_2025_alloc_sup | 83.8 | 91.8 | 5.3 | 22.2 | 6.6 | 90.4 | 52.3 | 4.4 | **50.0** | **28.3** |

## MRE (%)

| arm | hydro | coal | steam | ocgt | bat_chg | bat_dis | wind_cu | sol_cu |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp | 14.5 | 1.9 | 18.5 | 10.9 | 123.1 | 36.6 | 54.6 | 218.6 |
| hardnet_2025_alloc_sup | 12.8 | 2.3 | 25.0 | 13.6 | 92.2 | 25.5 | 63.3 | 97.8 |

_Share of window cells at or below 1 MW: hydro 5%, coal 0%, steam 94%, ocgt 59%, bat_chg 17%, bat_dis 3%, wind_cu 65%, sol_cu 91%._

## Window energy mix (% of gross generation, day-averaged)

| arm | coal | hydro | gas OCGT | gas steam | battery out |
| --- | ---: | ---: | ---: | ---: | ---: |
| actual | 77.1 | 12.6 | 3.1 | 0.4 | 6.8 |
| interp | 79.0 | 11.8 | 3.0 | 0.4 | 5.9 |
| hardnet_2025_alloc_sup | 78.5 | 11.9 | 3.0 | 0.4 | 6.1 |

_Gross generation excludes battery charging, which is a load._
