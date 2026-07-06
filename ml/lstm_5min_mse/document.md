Data preprocessing:

1. VIC market + generation, 5-min native; interpolate small battery gaps.
2. Six non-negative targets: hydro, coal_brown, gas_steam, gas_ocgt, battery_charging, battery_discharging.
3. Input mode net_dispatch_totdem (17 columns): dispatchable history + net_demand + total demand_mw + price + 8 calendar.
4. Global StandardScaler fit on train only (NO per-window RevIN).
5. Chronological 80-10-10 split; sliding window 24h (288 steps); target 5min ahead (horizon 1 steps).

Model Use:

1. Plain 2-layer LSTM (hidden 128, dropout 0.2), linear head on the final step. No RevIN, so it sees absolute demand levels.
2. Loss = MSE only (no penalty).
3. AdamW, early stopping on validation total loss. Trained on CUDA.
4. Trainable parameters: 208,134. Stopped at epoch 51.

Hyperparameter:

1. Lookback 24h (288 steps), horizon 1 steps (5min).
2. Batch 128, lr 0.0001, weight decay 1e-05.
3. Max epochs 200, patience 20, seed 0, lambda 0.

Results:

Per-target metrics on the 5-min test set.

| Energy | MAE | RMSE | WAPE | R^2 |
|---|---|---|---|---|
| hydro | 20.66 | 40.52 | 0.0708 | 0.9893 |
| coal_brown | 32.85 | 44.63 | 0.0091 | 0.9954 |
| gas_steam | 2.27 | 5.63 | 0.5197 | 0.9792 |
| gas_ocgt | 5.85 | 11.34 | 0.1584 | 0.9925 |
| battery_charging | 31.66 | 55.65 | 0.2180 | 0.9379 |
| battery_discharging | 29.29 | 52.21 | 0.2385 | 0.9418 |
| average | 20.43 | 35.00 | 0.2024 | 0.9727 |
