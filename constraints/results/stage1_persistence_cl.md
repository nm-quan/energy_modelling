# Closed-loop: anchored persistence vs neural backbones

ar_free_rollout nd_mode=scenario, reb20/red10, free window 11:00-14:00; response-region metrics (per-target cols = cl WAPE; gas_steam omitted, all-zero midday).

| model | cl_WAPE | cl_R2 | hydro | coal | gas_ocgt | batt_chg | batt_dis | track p50 MW | nd_resp | hydro_resp | batt_dis_resp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence+anchor | 0.4749 | 0.3337 | 0.446 | 0.116 | 0.203 | 0.415 | 1.195 | 70 | +179.2% | +102.3% | +132.7% |
| itransformer+anchor | 0.7015 | 0.4270 | 0.663 | 0.113 | 0.383 | 0.351 | 1.998 | 70 | +179.2% | +118.3% | +20.6% |
| lstm_5min_mse (prod) | 0.4971 | 0.4240 | 0.432 | 0.071 | 0.633 | 0.389 | 0.960 | 143 | +134.7% | +1504.8% | +7057.5% |
