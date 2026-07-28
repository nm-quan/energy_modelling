# How often is each fuel absent?

Source: 474,336 five-minute intervals, 2022-01-01 to 2026-07-05. **Absent** = at or below 10 MW.

## Longest continuous absence

| fuel | longest gap | from | to |
| --- | ---: | --- | --- |
| coal | **0.0 days** | 2022-01-01 | 2026-07-05 |
| hydro | **1.7 days** | 2026-01-11 | 2026-01-12 |
| gas OCGT | **16.8 days** | 2025-12-20 | 2026-01-06 |
| gas steam | **101.7 days** | 2026-01-30 | 2026-05-12 |
| battery out | **1.0 days** | 2024-09-24 | 2024-09-25 |
| battery in (load) | **0.7 days** | 2022-01-03 | 2022-01-04 |
| **no gas at all** | **16.8 days** | 2025-12-20 | 2026-01-06 |

## Share of days on which the fuel never runs

| fuel | days never active | share |
| --- | ---: | ---: |
| coal | 0 / 1,647 | 0.0% |
| hydro | 2 / 1,647 | 0.1% |
| gas OCGT | 494 / 1,647 | 30.0% |
| gas steam | 1,179 / 1,647 | 71.6% |
| battery out | 0 / 1,647 | 0.0% |
| battery in (load) | 0 / 1,647 | 0.0% |
| **no gas at all** | 479 / 1,647 | **29.1%** |

## Full calendar months in which gas_steam never once ran

- `2023-10`, `2023-11`, `2024-01`, `2024-10`, `2025-11`, `2026-02`, `2026-03`, `2026-04`

That is **8 of 54 full months**.

## What separates a gas day from a no-gas day

| | mean daily peak net demand |
| --- | ---: |
| days with no gas | 4,631 MW |
| days with gas | 5,752 MW |

Gas is a **threshold** phenomenon, not a daily one. It is the fuel of high-net-demand days, and on an ordinary day it simply does not exist.

## Why this matters for the imputation model

`gas_steam` is at or below 10 MW on 72% of days and has gone 102 consecutive days without running. The correct prediction for that channel is *exactly zero*, most of the time.

An MSE loss on z-scored targets cannot express that: it will settle on a small positive number every step, which is wrong on the ~72% of days the channel is off and barely penalised on the rest. This is the direct cause of the 248% gas_steam MRE in the interpolation benchmark — the denominator is near zero because the truth is near zero, while the prediction is not.

By contrast hydro and battery discharge have never been absent for more than about two days. They are genuinely always-on channels and an always-on estimator suits them.

Practical consequence: `gas_steam` (and to a lesser degree `gas_ocgt`) wants zero-inflated treatment — predict P(on) and the level separately — or at minimum a metric that does not reward a small constant.

