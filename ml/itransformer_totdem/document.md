Data preprocessing:

1. Load VIC market and generation data; pivot generation long-to-wide.
2. Native 5-min resolution; interpolate small battery gaps.
3. Six non-negative targets: hydro, coal_brown, gas_steam, gas_ocgt, battery_charging, battery_discharging.
4. Input mode net_dispatch_totdem: dispatchable history + net_demand + total demand_mw + price + 8 calendar features (17 columns).
5. demand_mw (total operational demand) is the new feature vs the net_dispatch benchmark; it carries the wind+solar-driven gap that net_demand alone misses.
6. Chronological split 80-10-10: train up to 2026-02-28, val to 2026-04-11, test to 2026-05-18.
7. StandardScaler fit on train only; sliding window 24h (288 steps) -> next 5-min step.

Model Use:

1. iTransformer: each of the 17 variables is one token (whole 288-step series embedded); self-attention runs across variables, so demand_mw and price inform every target. RevIN normalisation, d_model=128, 3 layers.
2. AdamW optimiser, MSE loss, early stopping on val MSE.
3. Trained on CUDA (RTX 4060 Ti).
4. Trainable parameters: 436,737.
5. Stopped at epoch 36 (best val_mse=0.03063).

Hyperparameter:

1. Lookback 24h (288 steps), horizon 1 step (5 min).
2. Batch 128, learning rate 0.0001, weight decay 1e-05.
3. Max epochs 200, early-stop patience 20. Seed 0.

Results:

Per-target metrics on 5-min native test set (10,656 rows).

| Energy | MAE | RMSE | WAPE | R^2 |
|---|---|---|---|---|
| hydro | 18.70 | 40.20 | 0.0641 | 0.9895 |
| coal_brown | 27.69 | 38.81 | 0.0077 | 0.9966 |
| gas_steam | 0.40 | 3.12 | 0.0919 | 0.9936 |
| gas_ocgt | 2.29 | 8.62 | 0.0619 | 0.9957 |
| battery_charging | 32.06 | 54.91 | 0.2208 | 0.9396 |
| battery_discharging | 28.59 | 51.07 | 0.2328 | 0.9443 |
| average | 18.29 | 32.79 | 0.1132 | 0.9765 |
