# Study demand-scenario rollout — reshape_g4.85564

Scenario **reshape**, g=4.85564%, free window 11:00-14:00, demand cap 10783.7 MW, n=53568 steps. nd input = supply-delta (base nd + demand change; in-distribution). response_capture = (nd_scen - nd_base rollout mean delta) / (input nd mean delta) over the response region — 1.0 = fleet delivers the full simulated increase. track_free = |SIGN·pred − nd_scen_input| in the response region. Ramp counts split: free window / seam (first TF step after the window; a frozen model snaps here) / rest. SOC per calendar day + legacy whole-window swing (drift artifact, transparency only).

| model | base_WAPE | base_R2 | demand_in_pct | nd_resp_pct | response_capture | coal_resp_pct | hydro_resp_pct | ocgt_resp_pct | batt_dis_resp_pct | track_free_p50_mw | track_free_p95_mw | bal_own_max_mw | mismatch_in_pct | n_ramp | n_ramp_free | n_ramp_seam | n_ramp_tf | ramp_max_mw | n_neg | soc_day_feasible_pct | soc_worst_day_pct | soc_window_swing_pct | secs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_persistence | 0.5120 | 0.4823 | 45.6 | 75.4 | 1.0317 | 69.1 | 47.3 | 44.1 | 52.1 | 74.8 | 289.0 | 0.0016 | 2.1415 | 576 | 290 | 87 | 199 | 2915.7 | 0 | 100.0 | 95.7 | 828.7 | 14.7 |
| itransformer_rayen | 0.5811 | 0.4528 | 45.6 | -0.0236 | -0.0003 | 0.0803 | 1.9228 | 0.7242 | -18.8 | 1963.6 | 2932.0 | 0.0012 | 12.3 | 43 | 0 | 42 | 1 | 1396.3 | 0 | 99.5 | 100.2 | 1965.6 | 141.0 |
| itransformer_rayenfd+spt | 2.3222 | -1.8796 | 45.6 | 75.2 | 1.0293 | 11.3 | 287.2 | 239.9 | 305.1 | 75.3 | 302.8 | 725.1 | 2.1419 | 536 | 0 | 167 | 369 | 1423.6 | 0 | 93.0 | 126.2 | 3394.1 | 167.4 |
| itransformer_task7+DR | 115.7 | -7878.5 | 45.6 | -46.8 | -3.5877 | -23.0 | -60.6 | -57.5 | -65.3 | 345.1 | 19412.5 | 0.0063 | 29.0 | 633 | 0 | 532 | 101 | 22672.6 | 0 | 98.9 | 115.0 | 917.8 | 160.5 |
| lstm_rayen | 0.5877 | 0.5100 | 45.6 | -0.0144 | -0.0002 | -0.0080 | -0.7580 | 0.2405 | 3.2255 | 1961.5 | 2931.4 | 0.0012 | 12.2 | 41 | 0 | 41 | 0 | 1234.1 | 0 | 100.0 | 90.8 | 404.1 | 524.1 |
| lstm_task7+DR | 6.4629 | -38.1 | 45.6 | 52.1 | 0.9242 | 26.5 | 157.4 | 119.8 | 125.1 | 1100.7 | 3047.9 | 0.0018 | 9.1576 | 789 | 0 | 642 | 147 | 1865.9 | 0 | 100.0 | 95.8 | 4557.9 | 511.5 |
| persistence | 0.4855 | 0.4770 | 45.6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1962.8 | 2931.5 | 0.0009 | 12.3 | 41 | 0 | 41 | 0 | 1256.9 | 0 | 100.0 | 99.8 | 1268.8 | 5.0000 |
