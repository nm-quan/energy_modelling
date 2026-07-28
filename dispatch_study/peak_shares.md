# Fuel proportions during the two daily peaks

Source: 474,336 five-minute intervals, 2022-01-01 to 2026-07-05, local AEST.

Victoria has **two** demand peaks. They are different events and are not served the same way, so each is measured against the trough it rises out of.

| window | hours | mean demand (MW) | mean net demand (MW) |
| --- | --- | ---: | ---: |
| overnight trough | 02:00–05:00 | 4,999 | 3,561 |
| MORNING peak | 06:00–09:00 | 5,484 | 3,809 |
| midday trough | 11:00–14:00 | 4,373 | 2,319 |
| EVENING peak | 17:00–21:00 | 6,317 | 4,719 |

## 1. Proportion of generation in each peak

Share of the five dispatchable generators' energy in the window. Battery charging is a load and sits outside the denominator (its mean MW is listed for reference).

| fuel | MORNING peak 06–09 | EVENING peak 17–21 | both peaks pooled |
| --- | ---: | ---: | ---: |
| coal | 87.8% | 76.9% | 81.1% |
| hydro | 7.9% | 13.3% | 11.2% |
| gas OCGT | 2.2% | 5.1% | 4.0% |
| gas steam | 0.7% | 2.2% | 1.7% |
| battery out | 1.4% | 2.5% | 2.1% |
| _battery in (load) (load, excluded)_ | _22 MW_ | _7 MW_ | _—_ |

## 2. Mean output by window (MW)

| fuel | overnight trough | MORNING peak | midday trough | EVENING peak |
| --- | ---: | ---: | ---: | ---: |
| coal | 3,684.8 | 3,633.5 | 3,046.4 | 3,853.4 |
| hydro | 166.5 | 325.8 | 62.2 | 665.4 |
| gas OCGT | 33.4 | 90.1 | 32.0 | 253.3 |
| gas steam | 12.8 | 30.6 | 12.2 | 111.0 |
| battery out | 8.8 | 56.3 | 7.1 | 126.0 |
| battery in (load) | 54.4 | 21.9 | 155.4 | 7.2 |

## 3. Turn-up — who is switched ON for each peak

Peak mean minus the mean of the trough that peak rises out of. Battery charging contributes by *reducing*, so its turn-up is the size of that reduction.

| fuel | morning turn-up (MW) | share | evening turn-up (MW) | share |
| --- | ---: | ---: | ---: | ---: |
| coal | -51.4 | -16.4% | +807.0 | 40.4% |
| hydro | +159.2 | 50.8% | +603.1 | 30.2% |
| gas OCGT | +56.6 | 18.1% | +221.2 | 11.1% |
| gas steam | +17.7 | 5.7% | +98.8 | 4.9% |
| battery out | +47.6 | 15.2% | +118.9 | 6.0% |
| battery in (load) | +32.5 | 10.3% | +148.2 | 7.4% |

_Total turn-up: morning 314 MW, evening 1,997 MW._

## 4. Appearance — % of steps active (>10 MW) in each peak

| fuel | morning | evening |
| --- | ---: | ---: |
| coal | 100.0% | 100.0% |
| hydro | 75.7% | 93.9% |
| gas OCGT | 27.0% | 56.1% |
| gas steam | 7.5% | 25.7% |
| battery out | 46.6% | 74.5% |
| battery in (load) | 31.5% | 12.7% |

## 5a. MORNING peak — generation share by year

| year | coal | hydro | gas OCGT | gas steam | battery out |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 84.8% | 10.2% | 3.2% | 1.2% | 0.7% |
| 2023 | 90.4% | 7.6% | 1.1% | 0.2% | 0.7% |
| 2024 | 88.1% | 7.4% | 2.6% | 0.9% | 1.0% |
| 2025 | 88.0% | 6.8% | 2.2% | 0.9% | 2.1% |
| 2026 | 88.3% | 6.7% | 1.4% | 0.1% | 3.4% |

## 5b. EVENING peak — generation share by year

| year | coal | hydro | gas OCGT | gas steam | battery out |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 73.2% | 15.0% | 6.9% | 4.1% | 0.8% |
| 2023 | 80.0% | 13.6% | 3.7% | 1.7% | 1.0% |
| 2024 | 77.9% | 13.2% | 5.6% | 1.9% | 1.4% |
| 2025 | 76.5% | 11.8% | 5.0% | 2.0% | 4.7% |
| 2026 | 77.5% | 12.2% | 3.1% | 0.4% | 6.8% |

