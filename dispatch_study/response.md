# How the Victorian grid actually responds — two studies

Source: 474,336 five-minute intervals, 2022-01-01 to 2026-07-05, local (AEST) time. Descriptive only, no model.

## Study A — marginal response: who supplies the next MW?

OLS slope of Δfuel on Δnet-demand over each horizon: **MW of that fuel per MW of net-demand change**. This is the merit order for *changes*, which is the thing a demand-shift study needs — as opposed to the merit order for *levels*, which just says coal is big.

| horizon | coal | hydro | gas OCGT | gas steam | battery out | battery in (load) | Σ (signed) | n pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 min | +0.150 | +0.154 | +0.019 | +0.005 | +0.089 | -0.109 | 0.527 | 474,335 |
| 30 min | +0.252 | +0.214 | +0.057 | +0.018 | +0.068 | -0.058 | 0.666 | 474,330 |
| 1 hour | +0.291 | +0.224 | +0.067 | +0.023 | +0.066 | -0.049 | 0.721 | 474,324 |
| 3 hours | +0.336 | +0.227 | +0.075 | +0.029 | +0.057 | -0.051 | 0.775 | 474,300 |

_Battery charging carries a negative sign in the balance, so a negative slope there means charging FALLS as net demand rises — which supplies power. Σ (signed) below 1 is the interconnector: the rest of the change is imported or exported rather than met by the local fleet._

## Study B — activation: what turns each fuel on?

Active = output above 10 MW. Switch-on level = the net demand at which the fuel is active more than half the time.

| fuel | switch-on net demand (MW) | median net demand when active (MW) | median price when active ($/MWh) | overall active |
| --- | ---: | ---: | ---: | ---: |
| coal | always | 3,830 | 65 | 100.0% |
| hydro | 2,739 | 4,231 | 91 | 70.3% |
| gas OCGT | 5,058 | 4,914 | 148 | 26.9% |
| gas steam | 6,708 | 5,103 | 171 | 12.8% |
| battery out | 5,058 | 4,326 | 97 | 37.6% |
| battery in (load) | always | 3,093 | 11 | 41.7% |

### P(active) by net-demand decile

| decile (net demand MW) | coal | hydro | gas OCGT | gas steam | battery out | battery in (load) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 (-3,559–1,534) | 100% | 21% | 1% | 0% | 19% | 80% |
| D2 (1,534–2,427) | 100% | 33% | 3% | 0% | 20% | 68% |
| D3 (2,427–3,051) | 100% | 52% | 5% | 2% | 24% | 56% |
| D4 (3,051–3,499) | 100% | 62% | 10% | 4% | 28% | 46% |
| D5 (3,499–3,830) | 100% | 72% | 13% | 6% | 32% | 40% |
| D6 (3,830–4,126) | 100% | 81% | 19% | 8% | 37% | 33% |
| D7 (4,126–4,432) | 100% | 88% | 27% | 11% | 42% | 30% |
| D8 (4,432–4,787) | 100% | 95% | 41% | 17% | 47% | 27% |
| D9 (4,787–5,329) | 100% | 99% | 60% | 27% | 55% | 22% |
| D10 (5,329–8,086) | 100% | 100% | 90% | 53% | 72% | 14% |

### Battery state by hour (% of steps)

| hour | charging | discharging |
| --- | ---: | ---: |
| 00:00 | 26% | 35% |
| 01:00 | 36% | 26% |
| 02:00 | 49% | 18% |
| 03:00 | 55% | 15% |
| 04:00 | 50% | 20% |
| 05:00 | 37% | 32% |
| 06:00 | 25% | 51% |
| 07:00 | 30% | 51% |
| 08:00 | 40% | 38% |
| 09:00 | 55% | 26% |
| 10:00 | 69% | 18% |
| 11:00 | 79% | 13% |
| 12:00 | 83% | 12% |
| 13:00 | 79% | 15% |
| 14:00 | 72% | 20% |
| 15:00 | 59% | 28% |
| 16:00 | 40% | 43% |
| 17:00 | 21% | 65% |
| 18:00 | 10% | 82% |
| 19:00 | 9% | 80% |
| 20:00 | 11% | 71% |
| 21:00 | 16% | 59% |
| 22:00 | 25% | 44% |
| 23:00 | 24% | 43% |

### Battery state by price quintile (% of steps)

| price quintile ($/MWh) | charging | discharging |
| --- | ---: | ---: |
| Q1 (-1,000 to -0) | 74% | 17% |
| Q2 (-0 to 36) | 53% | 26% |
| Q3 (36 to 85) | 32% | 40% |
| Q4 (85 to 140) | 27% | 49% |
| Q5 (140 to 17,500) | 23% | 56% |

