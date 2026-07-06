# Closed-loop (ar_free_rollout nd_mode=scenario) anchored vs bare

reb20/red10, demand-side net_demand, free window 11:00-14:00. cl_WAPE/cl_R2 = base-rollout accuracy vs actuals over the response region; track = |sum(SIGN*pred) - nd input| there.

| model | cl_WAPE | cl_R2 | track p50 MW | track p95 MW | nd_resp | coal_resp | hydro_resp | ocgt_resp | batt_dis_resp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm_5min_mse (prod) | 0.4971 | 0.4240 | 143 | 378 | +134.7% | +75.3% | +1504.8% | +45.3% | +7057.5% |
| itransformer (bare) | 0.6688 | 0.4797 | 341 | 1084 | +0.8% | +0.7% | +12.3% | -18.0% | +0.8% |
| itransformer+anchor | 0.7015 | 0.4270 | 70 | 222 | +179.2% | +156.1% | +118.3% | +113.3% | +20.6% |
| lstm_revin+anchor | 0.7812 | -0.0824 | 70 | 222 | +179.2% | +144.7% | +182.2% | +57.1% | +1676.5% |
