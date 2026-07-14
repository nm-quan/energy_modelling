# Study demand-scenario rollout — increase_g20

Scenario **increase**, g=20%, free window 11:00-14:00, demand cap 10783.7 MW, n=53568 steps. nd input = supply-delta (base nd + demand change; in-distribution). response_capture = (nd_scen - nd_base rollout mean delta) / (input nd mean delta) over the response region — 1.0 = fleet delivers the full simulated increase. track_free = |SIGN·pred − nd_scen_input| in the response region. Ramp counts split: free window / seam (first TF step after the window; a frozen model snaps here) / rest. SOC per calendar day + legacy whole-window swing (drift artifact, transparency only).

| model | base_WAPE | base_R2 | demand_in_pct | nd_resp_pct | response_capture | coal_resp_pct | hydro_resp_pct | ocgt_resp_pct | batt_dis_resp_pct | track_free_p50_mw | track_free_p95_mw | bal_own_max_mw | mismatch_in_pct | n_ramp | n_ramp_free | n_ramp_seam | n_ramp_tf | ramp_max_mw | n_neg | soc_day_feasible_pct | soc_worst_day_pct | soc_window_swing_pct | secs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_persistence | 0.5120 | 0.4823 | 19.4 | 32.0 | 1.0290 | 29.2 | 22.8 | 21.3 | 23.9 | 84.7 | 330.0 | 0.0012 | 1.9448 | 601 | 341 | 92 | 168 | 1224.8 | 0 | 99.5 | 100.3 | 1380.2 | 14.6 |
| itransformer_rayenfd+spt | 2.3222 | -1.8796 | 19.4 | 32.0 | 1.0290 | 5.7195 | 114.1 | 78.0 | 117.0 | 84.7 | 330.0 | 0.0009 | 1.9448 | 170 | 0 | 169 | 1 | 1770.3 | 0 | 96.8 | 115.8 | 7192.2 | 152.4 |
| persistence | 0.4855 | 0.4770 | 19.4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 719.7 | 1721.9 | 0.0009 | 3.9284 | 41 | 0 | 41 | 0 | 1256.9 | 0 | 100.0 | 99.8 | 1268.8 | 5.0000 |
