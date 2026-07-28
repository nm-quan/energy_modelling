# Who serves the Victorian evening peak

Source: `data/preprocessed/hist/5min/net_dispatch_ren/table.parquet`, 474,336 five-minute intervals, 2022-01-01 to 2026-07-05. All local (AEST) time.

**Evening peak** = 17:00–21:00, chosen from the data (mean demand exceeds 5,900 MW across exactly these hours and tops out 18:00–19:00). **Midday trough** = 11:00–14:00, the daily minimum. **Active** = output above 10 MW.

### Running level — who is ON during the peak

| fuel | mean all-day (MW) | mean trough 11-14 (MW) | mean peak 17-21 (MW) | % of peak energy |
| --- | ---: | ---: | ---: | ---: |
| coal | 3,568.9 | 3,046.4 | 3,853.4 | 76.8% |
| hydro | 301.2 | 62.2 | 665.4 | 13.3% |
| gas OCGT | 96.4 | 32.0 | 253.3 | 5.0% |
| gas steam | 42.8 | 12.2 | 111.0 | 2.2% |
| battery out | 40.6 | 7.1 | 126.0 | 2.5% |
| battery in (load) | 49.6 | 155.4 | 7.2 | 0.1% |

### Appearance — how often each fuel shows up in the peak at all

| fuel | % days active in peak | % peak steps active |
| --- | ---: | ---: |
| coal | 100.0% | 100.0% |
| hydro | 99.5% | 93.9% |
| gas OCGT | 61.9% | 56.1% |
| gas steam | 25.9% | 25.7% |
| battery out | 100.0% | 74.5% |
| battery in (load) | 86.8% | 12.7% |

### Peak ramp — who is turned UP to serve the evening rise

| fuel | peak ramp (MW) | % of total turn-up |
| --- | ---: | ---: |
| coal | +807.0 | 40.4% |
| hydro | +603.1 | 30.2% |
| gas OCGT | +221.2 | 11.1% |
| gas steam | +98.8 | 4.9% |
| battery out | +118.9 | 6.0% |
| battery in (load) | -148.2 | 7.4% |

_Battery charging is a load: it contributes to the turn-up by *reducing*, so its ramp is negative and its turn-up share is the magnitude of that reduction._

## Yearly evening peak

| year | days | mean peak demand (MW) | mean peak net demand (MW) | peak hour | trough→peak net-demand rise (MW) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 365 | 6,277 | 4,970 | 18:00 | 1,934 |
| 2023 | 365 | 6,121 | 4,598 | 19:00 | 2,208 |
| 2024 | 366 | 6,249 | 4,748 | 18:00 | 2,446 |
| 2025 | 365 | 6,419 | 4,518 | 19:00 | 2,851 |
| 2026 | 186 | 6,713 | 4,799 | 18:00 | 2,714 |

_2021 excluded (Oct–Dec only); 2026 runs to 5 July._

## Peak turn-up composition by year (% of the evening turn-up)

| year | coal | hydro | gas OCGT | gas steam | battery out | battery in (load) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 28.9% | 38.2% | 17.1% | 11.0% | 2.2% | 2.6% |
| 2023 | 46.5% | 34.3% | 9.7% | 4.4% | 2.4% | 2.7% |
| 2024 | 46.0% | 30.3% | 12.4% | 3.9% | 3.4% | 4.1% |
| 2025 | 41.5% | 23.5% | 9.5% | 3.7% | 9.6% | 12.2% |
| 2026 | 36.7% | 24.6% | 4.8% | 0.7% | 14.6% | 18.5% |

