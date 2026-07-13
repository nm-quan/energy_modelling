# Hist closed-loop stress episodes (ramp rate)

Autoregressive rollout over TEST-split stress episodes; models feed back their own dispatch, exogenous features stay actual. Ramps on the model's own trajectory (asymmetric data limits + 0.6 MW tol); ramp_excess_pct = 100*sum(MW beyond limit)/sum(|pred|). SOC per episode.

| model | cl_WAPE | n_ramp | ramp_excess_pct | bal_own_max_mw | n_demand_act | mismatch_act_pct | n_neg | soc_feasible_eps | soc_worst_ep_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 0.4832 | 0 | 0.0000 | 0.0000 | 699 | 7.5085 | 0 | 15 | 39.9 |
| lstm_rayen | 0.5319 | 0 | 0.0000 | 0.0008 | 696 | 7.5352 | 0 | 15 | 18.4 |
| itransformer_rayen | 0.4897 | 0 | 0.0000 | 0.0008 | 698 | 7.5135 | 0 | 15 | 40.0 |
| lstm_task7 | 0.5966 | 0 | 0.0000 | 78.9 | 663 | 2.2741 | 926 | 15 | 13.6 |
| lstm_task7+DR | 2.8864 | 0 | 0.0000 | 0.0012 | 675 | 13.1 | 0 | 15 | 21.9 |
| itransformer_task7 | 0.7500 | 0 | 0.0000 | 2850.0 | 708 | 14.8 | 865 | 15 | 21.9 |
| itransformer_task7+DR | 114.7 | 0 | 0.0000 | 0.0104 | 720 | 505.9 | 0 | 15 | 26.9 |
