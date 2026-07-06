# Demand-anchored head vs RevIN/Dish-TS/EB-loss (MPS, plain MSE)

stride 12, 6840 train windows, early-stop patience 5, demand input +101.2%

| model | R2 | WAPE | net_demand | coal | hydro | gas_ocgt | batt_dis | batt_chg | epochs | secs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm_demandhead | 0.9320 | 0.2336 | +179.2% | +136.5% | +547.0% | +260.2% | +684.4% | -13.7% | 29 | 248 |
| lstm_revin_demandhead | 0.9499 | 0.2186 | +179.2% | +141.7% | +272.0% | +133.7% | +371.9% | -18.5% | 29 | 200 |
