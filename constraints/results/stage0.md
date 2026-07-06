# Stage 0 baselines: existing models through constraint_report

demand refs: `in` = nd(t-1) given to the model, `act` = actual nd(t); violation threshold 10 MW. SOC0 = feasible starting-charge window, % of 4736 MWh.

| model | WAPE | R2 | n_neg | n_cap | n_ramp | n_demand_in | mismatch_in_pct | n_demand_act | mismatch_act_pct | SOC0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actuals | 0.0000 | 1.0000 | 83 | 0 | 0 | 9489 | 1.81 | 0 | 0.00 | [44%, 57%] |
| persistence | 0.0956 | 0.9748 | 83 | 0 | 0 | 0 | 0.00 | 9489 | 1.81 | [44%, 57%] |
| lstm_5min_mse | 0.2024 | 0.9727 | 13267 | 0 | 0 | 9105 | 1.17 | 9585 | 1.83 | [45%, 57%] |
| lstm_revin | 0.1252 | 0.9747 | 16906 | 0 | 1 | 8934 | 1.10 | 9594 | 1.84 | None (swing 112%) |
| lstm_revin+clamp | 0.1185 | 0.9748 | 0 | 0 | 1 | 8924 | 1.09 | 9613 | 1.84 | None (swing 112%) |
| lstm_revin+anchor | 0.1188 | 0.9746 | 0 | 0 | 0 | 0 | 0.00 | 9489 | 1.81 | None (swing 100%) |
| itransformer | 0.1132 | 0.9765 | 19809 | 0 | 0 | 8770 | 0.96 | 9517 | 1.74 | [43%, 58%] |
| itransformer+clamp | 0.1068 | 0.9767 | 0 | 0 | 0 | 8723 | 0.95 | 9511 | 1.74 | [43%, 58%] |
| itransformer+anchor | 0.1075 | 0.9763 | 0 | 0 | 1 | 0 | 0.00 | 9489 | 1.81 | [50%, 58%] |
