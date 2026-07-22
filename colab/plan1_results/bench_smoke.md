# plan1 benchmark — 60 test 3h gaps (seed 123), map balance = supply-side nd

| model | MAE agg (MW) | MRE agg (%) | cost pred ($) | Δcost vs truth ($) | bal>1MW | ramp | neg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interp+map (ref) | 47.1 | 6.30 | 57,016,720 | +909,239 | 0 | 0 | 0 |
| baseline | 47.1 | 6.30 | 57,016,720 | +909,239 | 0 | 0 | 0 |
| cost | 47.1 | 6.30 | 57,016,720 | +909,239 | 0 | 0 | 0 |
| size_aware | 50.9 | 6.81 | 56,221,460 | +113,979 | 0 | 0 | 0 |

## per-channel MRE (%)

| model | hydro | coal_brown | gas_steam | gas_ocgt | battery_charging | battery_discharging |
| --- | --- | --- | --- | --- | --- | --- |
| interp+map (ref) | 14.4 | 1.9 | 248.3 | 22.5 | 65.0 | 49.2 |
| baseline | 14.4 | 1.9 | 248.3 | 22.5 | 65.0 | 49.2 |
| cost | 14.4 | 1.9 | 248.3 | 22.5 | 65.0 | 49.2 |
| size_aware | 14.6 | 3.1 | 18.9 | 12.5 | 59.1 | 58.7 |

## per-channel MAE (MW)

| model | hydro | coal_brown | gas_steam | gas_ocgt | battery_charging | battery_discharging |
| --- | --- | --- | --- | --- | --- | --- |
| interp+map (ref) | 52.8 | 70.2 | 20.7 | 27.2 | 55.7 | 55.7 |
| baseline | 52.8 | 70.2 | 20.7 | 27.2 | 55.7 | 55.7 |
| cost | 52.8 | 70.2 | 20.7 | 27.2 | 55.7 | 55.7 |
| size_aware | 53.6 | 117.9 | 1.6 | 15.2 | 50.6 | 66.5 |
