# Hist teacher-forced test (demand balance)

Test = Jan-Jul 2026 (stride 1). @own = vs the model's own D_t (mechanism check, threshold 10.0 MW); @act = vs actual nd(t). nd_WAPE = D_t forecast quality. SOC per calendar day, eta_rt=0.834, cap 4736 MWh.

| model | WAPE | R2 | nd_WAPE | bal_own_max_mw | n_demand_own | n_demand_act | mismatch_act_pct | n_neg | n_ramp_vs_prev | soc_day_feasible_pct | soc_worst_day_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 0.1084 | 0.9669 | 0.0182 | 0.0004 | 0 | 47728 | 1.8158 | 0 | 0 | 100.0 | 79.4 |
| lstm_rayen | 0.1454 | 0.9668 | 0.0181 | 0.0013 | 0 | 47725 | 1.8102 | 0 | 0 | 100.0 | 79.7 |
| itransformer_rayen | 0.1289 | 0.9674 | 0.0182 | 0.0010 | 0 | 47768 | 1.8175 | 0 | 0 | 100.0 | 76.8 |
| itransformer_rayenfd+spt | 0.1123 | 0.9676 | 0.0182 | 0.0012 | 0 | 47728 | 1.8158 | 0 | 0 | 100.0 | 76.3 |
| lstm_task7 | 0.2700 | 0.9431 | 0.0248 | 155.7 | 16594 | 48296 | 2.5054 | 55162 | 731 | 100.0 | 61.9 |
| lstm_task7+DR | 0.3201 | 0.8992 | 0.0287 | 0.0018 | 0 | 49213 | 2.8714 | 0 | 0 | 100.0 | 71.9 |
| itransformer_task7 | 0.1313 | 0.9683 | 0.0211 | 254.6 | 38507 | 48773 | 1.9786 | 35131 | 0 | 100.0 | 76.2 |
| itransformer_task7+DR | 0.2692 | 0.9301 | 0.0271 | 0.0019 | 0 | 49960 | 2.7082 | 0 | 0 | 100.0 | 77.2 |
