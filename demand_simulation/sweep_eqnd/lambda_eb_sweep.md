# Energy-balance lambda sweep (lstm_selrevin + EB loss, MPS)

stride 12, 6840 train windows, early-stop patience 4, demand input +101%

| lambda | R2 | WAPE | net_demand_resp | coal_resp | batt_dis_resp | demand_sens | epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9395 | 0.2503 | +4.1% | +2.4% | +24.3% | 0.0% | 25 |
| 1 | 0.9373 | 0.2541 | +5.7% | +2.5% | +49.9% | 0.0% | 25 |
| 3 | 0.9334 | 0.2630 | +6.1% | +3.1% | +41.5% | 0.0% | 25 |
| 10 | 0.9190 | 0.2964 | +7.9% | +3.4% | +109.5% | 0.0% | 25 |
| 30 | 0.8903 | 0.3544 | +10.8% | +3.5% | +193.1% | 0.0% | 25 |
| 100 | 0.8241 | 0.4655 | +11.7% | +3.8% | +226.2% | 0.0% | 25 |
