# Evening peak 18:00-21:00 — per-day accuracy, no averaging

185 test days, 2026-01-01 .. 2026-07-04. MAE is over the 36 window steps x 6 dispatch channels of that day alone.

## The 6 days drawn

| day | interp MAE | hardnet_2025_alloc_sup MAE | interp WAPE | hardnet_2025_alloc_sup WAPE | interp MRE | hardnet_2025_alloc_sup MRE | winner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-01-01 | 83.2 | 70.4 | 14.2% | 12.1% | 26.8% * | 31.9% * | hardnet_2025_alloc_sup |
| 2026-01-02 | 87.3 | 23.7 | 11.7% | 3.2% | 32.6% * | 21.2% * | hardnet_2025_alloc_sup |
| 2026-01-03 | 47.1 | 23.0 | 6.7% | 3.2% | 26.9% * | 23.3% * | hardnet_2025_alloc_sup |
| 2026-01-04 | 52.3 | 33.7 | 8.8% | 5.7% | 62.7% * | 41.7% * | hardnet_2025_alloc_sup |
| 2026-01-05 | 61.5 | 34.5 | 8.7% | 4.9% | 40.2% * | 29.7% * | hardnet_2025_alloc_sup |
| 2026-01-06 | 100.9 | 81.3 | 9.8% | 7.9% | 32.4% * | 967.0% * | hardnet_2025_alloc_sup |

_MAE and WAPE are the numbers to read. WAPE pools numerator and denominator over all six channels; MRE averages per-channel ratios and is marked `*` on days where at least one channel was off for the whole window, which makes its denominator ~0 -- on 2026-01-06 gas steam ran 0.0 MW and alone scored 5,633%._

## Distribution over all 185 days (MAE, MW)

| arm | min | p10 | median | mean | p90 | max | WAPE (all days) | days it beats interp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp | 13.5 | 29.2 | 49.6 | 55.1 | 95.0 | 165.8 | 6.3% |  |
| hardnet_2025_alloc_sup | 10.8 | 25.3 | 48.7 | 50.0 | 77.1 | 122.2 | 5.8% | 96 / 185  (52%) |

## Hardest and easiest days for `hardnet_2025_alloc_sup`

| | day | interp MAE | hardnet_2025_alloc_sup MAE |
| --- | --- | ---: | ---: |
| best | 2026-01-16 | 23.8 | 10.8 |
| best | 2026-05-22 | 22.2 | 15.6 |
| best | 2026-03-01 | 16.1 | 16.1 |
| worst | 2026-01-25 | 112.2 | 122.2 |
| worst | 2026-01-18 | 116.7 | 113.6 |
| worst | 2026-07-03 | 71.3 | 100.4 |

## Per-channel MAE on the drawn days (MW)

| day | arm | hydro | coal | steam | ocgt | bat_chg | bat_dis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-01-01 | interp | 65.4 | 169.8 | 0.0 | 0.0 | 0.9 | 263.1 |
| 2026-01-01 | hardnet_2025_alloc_sup | 84.4 | 207.3 | 0.0 | 0.0 | 1.7 | 129.1 |
| 2026-01-02 | interp | 55.9 | 284.4 | 0.0 | 0.0 | 2.4 | 181.4 |
| 2026-01-02 | hardnet_2025_alloc_sup | 31.9 | 63.3 | 0.0 | 0.0 | 1.9 | 45.3 |
| 2026-01-03 | interp | 68.5 | 122.9 | 0.0 | 0.0 | 2.7 | 88.4 |
| 2026-01-03 | hardnet_2025_alloc_sup | 35.5 | 54.9 | 0.0 | 0.0 | 3.3 | 44.2 |
| 2026-01-04 | interp | 40.1 | 217.6 | 0.0 | 0.0 | 14.6 | 41.5 |
| 2026-01-04 | hardnet_2025_alloc_sup | 86.8 | 42.4 | 0.0 | 0.0 | 5.0 | 67.9 |
| 2026-01-05 | interp | 54.4 | 245.2 | 0.0 | 0.0 | 5.7 | 63.9 |
| 2026-01-05 | hardnet_2025_alloc_sup | 54.2 | 49.4 | 0.0 | 0.0 | 3.3 | 100.3 |
| 2026-01-06 | interp | 95.6 | 25.0 | 0.0 | 186.1 | 3.5 | 295.1 |
| 2026-01-06 | hardnet_2025_alloc_sup | 210.8 | 29.2 | 1.6 | 133.4 | 3.7 | 109.1 |

_MRE is unreliable where the truth is mostly zero. Share of window cells at or below 1 MW: hydro 5%, coal 0%, steam 94%, ocgt 59%, bat_chg 17%, bat_dis 3%._
