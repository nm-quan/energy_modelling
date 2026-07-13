# Hist constrained demand-shift simulation

Scenario: FixedPercentageShift rebound=2.0000%, reduction=2.0000%, free window 11:00-14:00. The default value is the maximum equal rebound/reduction over val-tail+test under demand cap 10783.7 MW: q_max=4.8556%. Hist net_demand input is `demand_mw - wind - solar_utility` because hist demand is already VIC-island adjusted. Rows use closed-loop free-window rollout; base_WAPE/base_R2 score the no-shift rollout over the response region, while violation metrics are on the full scenario test rollout.

| model | base_WAPE | base_R2 | demand_in_pct | nd_resp_pct | coal_resp_pct | hydro_resp_pct | ocgt_resp_pct | batt_dis_resp_pct | track_p50_mw | track_p95_mw | bal_own_max_mw | n_demand_in | mismatch_in_pct | n_ramp | ramp_mean_mw | ramp_mean_mwh | ramp_max_mw | ramp_total_mwh | ramp_excess_pct | n_neg | soc_feasible | soc_swing_pct | alpha_active_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| itransformer_rayen | 0.5830 | 0.4529 | 18.8 | -0.0081 | 0.0537 | 0.7625 | -1.2746 | -4.3534 | 331.7 | 730.1 | 0.0011 | 53085 | 9.4076 | 43 | 332.0 | 27.7 | 1478.2 | 1189.5 | 0.0063 | 0 | no | 2078.8 | — |
