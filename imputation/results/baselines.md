# Gap-imputation baselines — reconstruction WAPE on real test 11:00-14:00 windows

186 test days, 36-step gap, 72-step context each side. Reconstruction WAPE = fill vs measured truth (answer key exists — we masked real data). This is the bar the bi-LSTM must beat.

| method | macro | hydro | coal_brown | gas_steam | gas_ocgt | battery_charging | battery_discharging | ramp | n_neg | bal_max_mw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interp | 0.3451 | 0.2729 | 0.0227 | 0.2149 | 0.1076 | 0.3167 | 1.1357 | 0 | 0 | 1003.2 |
| interp+proj | 0.4056 | 0.3881 | 0.0211 | 0.2149 | 0.3571 | 0.3167 | 1.1357 | 0 | 0 | 648.7 |
