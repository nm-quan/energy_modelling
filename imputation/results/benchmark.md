# Imputation benchmark (general eval, 800 windows, seed 123)

Same windows, same projection Π. macro=mean per-channel WAPE; micro=Σerr/Σtruth (stable); midday=the 11-14 deployment slice. Bar = interpolation + projection. `classical/*` (mean/knn/mice/mf) impute the 6 sources from the known drivers only (no boundary/temporal structure); `bilstm/*` are the learned modes.

| method | macro_WAPE | micro_WAPE | midday_micro_WAPE | ramp_overshoot_mw | balance_resid_max_mw | n_neg |
| --- | --- | --- | --- | --- | --- | --- |
| interp+projection | 0.5297 | 0.0662 | 0.0730 | 1.5e-06 | 7.3e-05 | 0 |
| classical/mean | 2.4059 | 0.2327 | 0.2364 | 1.5e-06 | 4.3e-01 | 0 |
| classical/knn | 1.2972 | 0.1599 | 0.1727 | 1.5e-06 | 5.6e-01 | 0 |
| classical/mice | 1.3576 | 0.2679 | 0.2451 | 1.2e-05 | 5.8e-01 | 0 |
| classical/mf | 1.3707 | 0.1928 | 0.2072 | 1.5e-06 | 5.2e-01 | 0 |
| bilstm/none | _not run yet_ | | | | | |
| bilstm/posthoc | _not run yet_ | | | | | |
| bilstm/unrolled | _not run yet_ | | | | | |
| bilstm/rayen_traj | _not run yet_ | | | | | |
