# plan1 baselines — classical imputers, 400 test 3h gaps, rayen map

| method | MAE agg (MW) | MRE agg (%) | MRE hydro | MRE coal_brown | MRE gas_steam | MRE gas_ocgt | MRE battery_charging | MRE battery_discharging |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | 50.8 | 7.10 | 17.6 | 2.3 | 326.0 | 31.9 | 52.9 | 53.3 |
| knn | 51.8 | 7.24 | 17.8 | 2.4 | 332.0 | 34.2 | 53.6 | 53.0 |
| mice | 88.9 | 12.43 | 49.1 | 4.1 | 387.3 | 66.6 | 57.4 | 76.1 |
| mf | 52.8 | 7.39 | 17.8 | 2.5 | 363.5 | 36.6 | 51.3 | 52.5 |
| interp | 47.6 | 6.66 | 16.9 | 2.1 | 279.1 | 29.3 | 51.7 | 52.0 |
