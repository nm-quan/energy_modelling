# Inference-only demand anchoring on converged blind checkpoints

teacher-forced reb20/red10, demand input +101.2% (structural full response = +179.2%); per-target columns are WAPE

| model | R2 | WAPE | hydro | coal | gas_steam | gas_ocgt | batt_chg | batt_dis | nd_resp | coal_resp | hydro_resp | ocgt_resp | batt_dis_resp | batt_chg_resp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm_revin | 0.9747 | 0.1252 | 0.068 | 0.008 | 0.119 | 0.076 | 0.233 | 0.246 | +3.9% | +1.7% | +47.2% | -3.3% | +8.3% | -3.7% |
| lstm_revin+anchor | 0.9746 | 0.1530 | 0.064 | 0.012 | 0.294 | 0.076 | 0.229 | 0.242 | +179.2% | +144.7% | +205.2% | +91.9% | +156.5% | -3.7% |
| itransformer | 0.9765 | 0.1132 | 0.064 | 0.008 | 0.092 | 0.062 | 0.221 | 0.233 | +0.1% | +0.1% | +0.9% | -0.6% | +3.0% | +0.3% |
| itransformer+anchor | 0.9763 | 0.1420 | 0.060 | 0.011 | 0.262 | 0.071 | 0.219 | 0.229 | +179.2% | +147.6% | +107.2% | +99.1% | +157.5% | +0.3% |
