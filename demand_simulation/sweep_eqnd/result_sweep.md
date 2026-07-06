Rebound sweep -- lstm_5min_mse, DEMAND-SIDE net_demand (shifted, teacher-forced)

net_demand = demand_mw - wind - solar_utility - net_import (net_import = imports - exports).
reduction 0%, free window 11:00-14:00. Means over the response region. net_demand column = signed sum of predicted dispatch.
Caveat: model trained on dispatch-sum net_demand; demand-side net_demand is fed via the saved scaler.

| rebound | demand_mw in | net_demand | coal_brown | hydro | gas_ocgt | battery_dis |
|---|---|---|---|---|---|---|
| 10% | +9.7% | +13.3% | +10.9% | -27.2% | -57.6% | -20.7% |
| 20% | +19.4% | +27.4% | +21.3% | +5.9% | -58.2% | +39.9% |
| 30% | +29.2% | +41.4% | +30.9% | +70.2% | -51.4% | +181.0% |
| 40% | +38.9% | +55.0% | +39.8% | +167.2% | -45.5% | +372.9% |
| 50% | +48.6% | +68.1% | +47.8% | +305.7% | -36.8% | +642.7% |
| 60% | +58.3% | +80.9% | +54.8% | +481.9% | -30.5% | +1109.9% |
| 70% | +68.0% | +93.4% | +60.7% | +694.7% | -24.5% | +1886.9% |
| 80% | +77.8% | +105.6% | +65.8% | +920.8% | -14.8% | +3008.4% |
| 90% | +87.5% | +117.5% | +69.9% | +1156.6% | +7.7% | +4492.0% |
| 100% | +97.2% | +128.6% | +73.7% | +1354.7% | +37.5% | +6090.7% |
