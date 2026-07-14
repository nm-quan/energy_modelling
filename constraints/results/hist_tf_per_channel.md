# Hist teacher-forced per-channel WAPE

Same run as hist_tf.md. delta_vs_persistence = macro WAPE minus the persistence row's. Persistence is the h=1 floor; the study gate asks for parity (<= +0.005 macro) with the residual gap isolated to the battery channels.

| model | hydro | coal_brown | gas_steam | gas_ocgt | battery_charging | battery_discharging | macro_WAPE | delta_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 0.0533 | 0.0076 | 0.0360 | 0.0343 | 0.2493 | 0.2697 | 0.1084 | +0.0000 |
| lstm_rayen | 0.0581 | 0.0077 | 0.2402 | 0.0495 | 0.2499 | 0.2668 | 0.1454 | +0.0370 |
| itransformer_rayen | 0.0544 | 0.0085 | 0.1489 | 0.0449 | 0.2453 | 0.2713 | 0.1289 | +0.0205 |
| itransformer_rayenfd+spt | 0.0562 | 0.0090 | 0.0360 | 0.0547 | 0.2449 | 0.2729 | 0.1123 | +0.0039 |
| lstm_task7 | 0.1165 | 0.0146 | 0.7576 | 0.1382 | 0.2953 | 0.2976 | 0.2700 | +0.1616 |
| lstm_task7+DR | 0.1883 | 0.0139 | 0.5786 | 0.1943 | 0.4811 | 0.4647 | 0.3201 | +0.2117 |
| itransformer_task7 | 0.0777 | 0.0097 | 0.0852 | 0.0624 | 0.2627 | 0.2902 | 0.1313 | +0.0229 |
| itransformer_task7+DR | 0.1711 | 0.0109 | 0.3372 | 0.1781 | 0.4183 | 0.4996 | 0.2692 | +0.1608 |
