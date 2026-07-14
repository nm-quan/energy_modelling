# Study demand-scenario rollout — increase_g10

Scenario **increase**, g=10%, free window 11:00-14:00, demand cap 10783.7 MW, n=53568 steps. nd input = supply-delta (base nd + demand change; in-distribution). response_capture = (nd_scen - nd_base rollout mean delta) / (input nd mean delta) over the response region — 1.0 = fleet delivers the full simulated increase. track_free = |SIGN·pred − nd_scen_input| in the response region. Ramp counts split: free window / seam (first TF step after the window; a frozen model snaps here) / rest. SOC per calendar day + legacy whole-window swing (drift artifact, transparency only).

| model | base_WAPE | base_R2 | demand_in_pct | nd_resp_pct | response_capture | coal_resp_pct | hydro_resp_pct | ocgt_resp_pct | batt_dis_resp_pct | track_free_p50_mw | track_free_p95_mw | bal_own_max_mw | mismatch_in_pct | n_ramp | n_ramp_free | n_ramp_seam | n_ramp_tf | ramp_max_mw | n_neg | soc_day_feasible_pct | soc_worst_day_pct | soc_window_swing_pct | secs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_persistence | 0.5120 | 0.4823 | 9.7182 | 16.0 | 1.0290 | 14.6 | 11.4 | 10.6 | 12.0 | 78.1 | 287.9 | 0.0013 | 1.8699 | 385 | 267 | 90 | 28 | 924.3 | 0 | 100.0 | 99.8 | 1350.4 | 13.8 |
| itransformer_rayen | 0.5811 | 0.4528 | 9.7182 | -0.0030 | -0.0002 | -0.0369 | 0.3076 | -0.6298 | -1.0622 | 338.7 | 1126.7 | 0.0010 | 2.8549 | 43 | 0 | 42 | 1 | 1478.3 | 0 | 99.5 | 100.6 | 2122.1 | 142.6 |
| itransformer_rayenfd+spt | 2.3222 | -1.8796 | 9.7182 | 16.0 | 1.0290 | 3.9274 | 50.0 | 32.3 | 53.2 | 78.1 | 287.9 | 0.0009 | 1.8699 | 180 | 0 | 178 | 2 | 1971.0 | 0 | 99.5 | 100.8 | 4670.3 | 156.4 |
| itransformer_task7+DR | 115.7 | -7878.5 | 9.7182 | -50.2 | -18.1 | -27.8 | -62.4 | -60.9 | -64.1 | 218.8 | 20658.1 | 0.0070 | 16.8 | 652 | 0 | 652 | 0 | 25446.4 | 0 | 100.0 | 89.0 | 834.5 | 144.3 |
| lstm_rayen | 0.5877 | 0.5100 | 9.7182 | -0.0033 | -0.0002 | -0.0018 | -0.1563 | 0.0438 | -0.2621 | 339.2 | 1123.8 | 0.0011 | 2.8447 | 41 | 0 | 41 | 0 | 1233.6 | 0 | 100.0 | 90.8 | 423.8 | 422.3 |
| lstm_task7+DR | 6.4629 | -38.1 | 9.7182 | 6.6077 | 0.5509 | 6.5016 | 4.3348 | -1.0756 | -7.0794 | 246.4 | 3267.3 | 0.0017 | 4.7874 | 345 | 0 | 340 | 5 | 1949.0 | 0 | 100.0 | 90.1 | 3049.6 | 427.7 |
| persistence | 0.4855 | 0.4770 | 9.7182 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 339.1 | 1127.3 | 0.0009 | 2.8518 | 41 | 0 | 41 | 0 | 1256.9 | 0 | 100.0 | 99.8 | 1268.8 | 4.9000 |
