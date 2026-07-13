# Hist teacher-forced test (demand balance)

Test = Jan-Jul 2026 (stride 500). @own = vs the model's own D_t (mechanism check, threshold 10.0 MW); @act = vs actual nd(t). nd_WAPE = D_t forecast quality. SOC per calendar day, eta_rt=0.834, cap 4736 MWh.

| model | WAPE | R2 | nd_WAPE | bal_own_max_mw | n_demand_own | n_demand_act | mismatch_act_pct | n_neg | n_ramp_vs_prev | soc_day_feasible_pct | soc_worst_day_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 0.1114 | 0.9646 | 0.0181 | 0.0003 | 0 | 99 | 1.8098 | 0 | 0 | 45.4 | 721.2 |
| lstm_rayen | 0.1319 | 0.9642 | 0.0180 | 0.0005 | 0 | 100 | 1.7996 | 0 | 0 | 45.4 | 694.4 |
| itransformer_rayen | 0.1210 | 0.9682 | 0.0181 | 0.0008 | 0 | 100 | 1.8095 | 0 | 0 | 48.1 | 721.3 |
| lstm_task7 | 0.2010 | 0.9614 | 0.0233 | 99.5 | 45 | 94 | 2.3619 | 109 | 0 | 47.2 | 659.0 |
| lstm_task7+DR | 0.2437 | 0.9380 | 0.0249 | 0.0007 | 0 | 97 | 2.4890 | 0 | 0 | 47.2 | 663.1 |
| itransformer_task7 | 0.1281 | 0.9711 | 0.0218 | 138.5 | 78 | 97 | 1.9017 | 74 | 0 | 47.2 | 683.5 |
| itransformer_task7+DR | 0.2520 | 0.9123 | 0.0271 | 0.0008 | 0 | 100 | 2.7108 | 0 | 0 | 44.4 | 695.1 |
