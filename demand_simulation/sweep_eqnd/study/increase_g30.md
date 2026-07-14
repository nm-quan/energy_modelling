# Study demand-scenario rollout — increase_g30

Scenario **increase**, g=30%, free window 11:00-14:00, demand cap 10783.7 MW, n=53568 steps. nd input = supply-delta (base nd + demand change; in-distribution). response_capture = (nd_scen - nd_base rollout mean delta) / (input nd mean delta) over the response region — 1.0 = fleet delivers the full simulated increase. track_free = |SIGN·pred − nd_scen_input| in the response region. Ramp counts split: free window / seam (first TF step after the window; a frozen model snaps here) / rest. SOC per calendar day + legacy whole-window swing (drift artifact, transparency only).

| model | base_WAPE | base_R2 | demand_in_pct | nd_resp_pct | response_capture | coal_resp_pct | hydro_resp_pct | ocgt_resp_pct | batt_dis_resp_pct | track_free_p50_mw | track_free_p95_mw | bal_own_max_mw | mismatch_in_pct | n_ramp | n_ramp_free | n_ramp_seam | n_ramp_tf | ramp_max_mw | n_neg | soc_day_feasible_pct | soc_worst_day_pct | soc_window_swing_pct | secs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_persistence | 0.5120 | 0.4823 | 29.2 | 48.1 | 1.0290 | 43.8 | 34.2 | 31.9 | 35.9 | 90.8 | 356.4 | 0.0014 | 2.0188 | 668 | 387 | 93 | 188 | 1919.6 | 0 | 99.5 | 100.8 | 1410.0 | 14.4 |
| itransformer_rayenfd+spt | 2.3222 | -1.8796 | 29.2 | 48.0 | 1.0289 | 7.6999 | 180.2 | 133.1 | 183.1 | 90.8 | 356.4 | 160.3 | 2.0186 | 259 | 0 | 169 | 90 | 1600.3 | 0 | 88.7 | 134.0 | 9522.3 | 143.9 |
| persistence | 0.4855 | 0.4770 | 29.2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1140.2 | 2347.8 | 0.0009 | 5.1985 | 41 | 0 | 41 | 0 | 1256.9 | 0 | 100.0 | 99.8 | 1268.8 | 5.1000 |
