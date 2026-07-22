# plan1 benchmark — 400 test 3h gaps (seed 123), map balance = supply-side nd

| model | MAE agg (MW) | MRE agg (%) | cost pred ($) | Δcost vs truth ($) | bal>1MW | ramp | neg | SOC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interp+map (ref) | 47.6 | 6.66 | 356,043,386 | +6,538,048 | 0 | 0 | 0 | 0 |
| baseline | 50.0 | 6.99 | 356,383,821 | +6,878,484 | 0 | 0 | 0 | 0 |
| cost | 47.9 | 6.70 | 355,993,780 | +6,488,443 | 0 | 0 | 0 | 0 |
| size_aware | 53.9 | 7.54 | 354,153,457 | +4,648,120 | 0 | 0 | 0 | 0 |

## per-channel MRE (%)

| model | hydro | coal_brown | gas_steam | gas_ocgt | battery_charging | battery_discharging |
| --- | --- | --- | --- | --- | --- | --- |
| interp+map (ref) | 16.9 | 2.1 | 279.1 | 29.3 | 51.7 | 52.0 |
| baseline | 16.8 | 2.3 | 335.3 | 29.5 | 52.0 | 52.9 |
| cost | 17.0 | 2.1 | 282.2 | 29.5 | 51.8 | 51.7 |
| size_aware | 19.7 | 3.1 | 127.5 | 32.1 | 51.0 | 52.7 |

## per-channel MAE (MW)

| model | hydro | coal_brown | gas_steam | gas_ocgt | battery_charging | battery_discharging |
| --- | --- | --- | --- | --- | --- | --- |
| interp+map (ref) | 52.9 | 75.7 | 18.4 | 26.7 | 57.5 | 54.4 |
| baseline | 52.7 | 85.1 | 22.2 | 26.8 | 57.9 | 55.3 |
| cost | 53.3 | 77.3 | 18.7 | 26.9 | 57.6 | 54.0 |
| size_aware | 62.0 | 112.1 | 8.4 | 29.2 | 56.8 | 55.1 |
