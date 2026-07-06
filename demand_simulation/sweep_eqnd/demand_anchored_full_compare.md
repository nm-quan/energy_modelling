# Demand-anchored head, full unstrided convergence run

82080 train windows, demand input +101.2%

| model | R2 | WAPE | net_demand | coal | hydro | gas_ocgt | batt_dis | batt_chg | epochs | secs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm_revin_demandhead | 0.8929 | 0.3375 | +179.2% | +153.8% | +121.8% | +133.2% | +221.3% | -5.3% | 28 | 2349 |
