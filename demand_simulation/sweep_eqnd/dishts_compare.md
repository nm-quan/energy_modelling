# Dish-TS vs RevIN (learned horizon level, MPS, plain MSE)

stride 12, 6840 train windows, early-stop patience 5, demand input +101%

| model | R2 | WAPE | net_demand_resp | coal_resp | batt_dis_resp | demand_sens | epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM +RevIN | 0.9418 | 0.2411 | +21.2% | +13.0% | +40.2% | 0.0% | 29 |
| Dish-TS own | 0.9516 | 0.2880 | +17.7% | +11.3% | +119.3% | 0.0% | 21 |
| Dish-TS cross | 0.9418 | 0.2916 | +3.5% | +1.3% | +14.5% | 0.0% | 23 |
| Dish-TS exo | 0.5633 | 2.0827 | +34.5% | +21.6% | +456.0% | 0.0% | 21 |
