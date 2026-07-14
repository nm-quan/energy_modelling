# Study demand-scenario rollout — increase_g5

Scenario **increase**, g=5%, free window 11:00-14:00, demand cap 10783.7 MW, n=53568 steps. nd input = supply-delta (base nd + demand change; in-distribution). response_capture = (nd_scen - nd_base rollout mean delta) / (input nd mean delta) over the response region — 1.0 = fleet delivers the full simulated increase. track_free = |SIGN·pred − nd_scen_input| in the response region. Ramp counts split: free window / seam (first TF step after the window; a frozen model snaps here) / rest. SOC per calendar day + legacy whole-window swing (drift artifact, transparency only).

| model | base_WAPE | base_R2 | demand_in_pct | nd_resp_pct | response_capture | coal_resp_pct | hydro_resp_pct | ocgt_resp_pct | batt_dis_resp_pct | track_free_p50_mw | track_free_p95_mw | bal_own_max_mw | mismatch_in_pct | n_ramp | n_ramp_free | n_ramp_seam | n_ramp_tf | ramp_max_mw | n_neg | soc_day_feasible_pct | soc_worst_day_pct | soc_window_swing_pct | secs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_persistence | 0.5120 | 0.4823 | 4.8591 | 8.0087 | 1.0290 | 7.3050 | 5.7054 | 5.3169 | 5.9753 | 74.3 | 257.1 | 0.0012 | 1.8334 | 220 | 135 | 85 | 0 | 839.3 | 0 | 100.0 | 99.6 | 1335.5 | 14.3 |
| itransformer_rayenfd+spt | 2.3222 | -1.8796 | 4.8591 | 8.0087 | 1.0290 | 2.6942 | 20.7 | 13.0 | 22.2 | 74.3 | 257.1 | 0.0009 | 1.8334 | 185 | 0 | 183 | 2 | 2054.6 | 0 | 100.0 | 92.4 | 3417.5 | 141.8 |
| persistence | 0.4855 | 0.4770 | 4.8591 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 246.2 | 907.9 | 0.0009 | 2.5404 | 41 | 0 | 41 | 0 | 1256.9 | 0 | 100.0 | 99.8 | 1268.8 | 4.8000 |
