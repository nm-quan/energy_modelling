# Hist closed-loop stress episodes (ramp rate)

Autoregressive rollout over TEST-split stress episodes; models feed back their own dispatch, exogenous features stay actual. Ramps on the model's own trajectory (asymmetric data limits + 0.6 MW tol); ramp_excess_pct = 100*sum(MW beyond limit)/sum(|pred|). SOC per episode.

| model | cl_WAPE | n_ramp | ramp_excess_pct | bal_own_max_mw | n_demand_act | mismatch_act_pct | n_neg | soc_feasible_eps | soc_worst_ep_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 0.7589 | 0 | 0.0000 | 0.0000 | 4286 | 23.5 | 0 | 12 | 239.4 |
| lstm_rayen | 0.7988 | 0 | 0.0000 | 0.0010 | 4273 | 23.3 | 0 | 14 | 111.1 |
| itransformer_rayen | 0.7631 | 0 | 0.0000 | 0.0010 | 4282 | 23.5 | 0 | 13 | 240.7 |
| itransformer_rayenfd+spt | 1.6540 | 0 | 0.0000 | 0.0016 | 3920 | 1.8812 | 0 | 0 | 553.0 |
| lstm_task7 | 0.6335 | 0 | 0.0000 | 78.9 | 4070 | 3.5184 | 5648 | 15 | 23.7 |
| lstm_task7+DR | 3.2152 | 0 | 0.0000 | 0.0017 | 4231 | 31.6 | 0 | 9 | 125.7 |
| itransformer_task7 | 1.4330 | 106 | 0.0476 | 51099.6 | 4301 | 108.8 | 4448 | 15 | 86.3 |
| itransformer_task7+DR | 294.1 | 0 | 0.0000 | 0.1020 | 4320 | 3141.4 | 0 | 0 | 628.8 |
