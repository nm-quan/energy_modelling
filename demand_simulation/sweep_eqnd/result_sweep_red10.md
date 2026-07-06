Rebound sweep -- lstm_5min_mse, DEMAND-SIDE net_demand (shifted, teacher-forced)

net_demand = demand_mw - wind - solar_utility - net_import (net_import = imports - exports).
reduction 10%, free window 11:00-14:00. Means over the response region. net_demand column = signed sum of predicted dispatch.
Caveat: model trained on dispatch-sum net_demand; demand-side net_demand is fed via the saved scaler.

| rebound | demand_mw in | net_demand | coal_brown | hydro | gas_ocgt | battery_dis |
|---|---|---|---|---|---|---|
| 20% | +101.2% | +134.7% | +75.3% | +1504.8% | +45.3% | +7057.5% |
