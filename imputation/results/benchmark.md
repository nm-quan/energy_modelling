# Constraint-mode benchmark (general eval, 800 windows, seed 123)

Same windows, same posthoc projection. macro=mean per-channel WAPE; micro=Σerr/Σtruth (stable); midday=the 11-14 deployment slice. Bar = interpolation + projection.

| method | macro_WAPE | micro_WAPE | midday_micro_WAPE | ramp_overshoot_mw | balance_resid_max_mw | n_neg |
| --- | --- | --- | --- | --- | --- | --- |
| interp+projection | 0.5297 | 0.0662 | 0.0750 | 1.5e-06 | 7.3e-05 | 0 |
| bilstm/posthoc | _not trained yet_ | | | | | |
| bilstm/unrolled | _not trained yet_ | | | | | |
| bilstm/rayen_traj | _not trained yet_ | | | | | |
