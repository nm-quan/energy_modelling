# Hist constrained demand-shift simulation

Scenario: FixedPercentageShift rebound=4.8556%, reduction=4.8556%, free window 11:00-14:00. The default value is the maximum equal rebound/reduction over val-tail+test under demand cap 10783.7 MW: q_max=4.8556%. Hist net_demand input is `demand_mw - wind - solar_utility` because hist demand is already VIC-island adjusted. Rows use closed-loop free-window rollout; base_WAPE/base_R2 score the no-shift rollout over the response region, while violation metrics are on the full scenario test rollout.

| model | base_WAPE | base_R2 | demand_in_pct | nd_resp_pct | coal_resp_pct | hydro_resp_pct | ocgt_resp_pct | batt_dis_resp_pct | track_p50_mw | track_p95_mw | bal_own_max_mw | n_demand_in | mismatch_in_pct | n_ramp | ramp_excess_pct | n_neg | soc_feasible | soc_swing_pct | alpha_active_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm_rayen | 0.5871 | 0.5103 | 45.6 | -0.0145 | -0.0079 | -0.7500 | 0.2367 | 2.8541 | 510.6 | 1942.6 | 0.0012 | 53519 | 17.0 | 41 | 0.0056 | 0 | no | 405.5 | — |
| itransformer_rayen | 0.5813 | 0.4532 | 45.6 | -0.0238 | 0.1285 | 1.7629 | 1.0107 | -18.4 | 512.3 | 1944.8 | 0.0011 | 53519 | 17.0 | 43 | 0.0062 | 0 | no | 1957.5 | — |
| lstm_task7+DR | 6.5795 | -38.7 | 45.6 | 51.1 | 28.0 | 142.7 | 107.2 | 100.7 | 375.9 | 1378.5 | 0.0015 | 53365 | 12.6 | 742 | 0.1430 | 0 | no | 4220.0 | 82.8 |
| itransformer_task7+DR | 143.7 | -10598.1 | 45.6 | -55.5 | -30.8 | -68.1 | -65.7 | -71.9 | 1152.1 | 1673.0 | 0.0072 | 53354 | 38.2 | 788 | 1.2559 | 0 | no | 1007.4 | 90.8 |
