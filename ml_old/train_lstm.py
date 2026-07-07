"""Train + evaluate the no-RevIN 5-min LSTM dispatch forecaster (lstm_5min_mse).

Pipeline -> LSTM -> evaluation, all from this clean folder. Reproduces the bundled
ml/lstm_5min_mse/ artifacts (weights, scalers, metrics, predictions, loss curve,
all-energy stacked graphs).

Why this model: the iTransformer wraps every channel in RevIN (per-window
normalisation), which makes it invariant to demand-level shifts and collapses in
closed loop. A plain LSTM has no RevIN, so it reads absolute demand and responds
to a demand shift -- which is what the demand simulation needs.

Spec: net_dispatch_totdem (17 features), 24h lookback (288 steps), 5-min horizon,
AdamW lr=1e-4 wd=1e-5 batch=128, MSE loss, early stop patience 20, seed 0.

Run:
    python ml/train_lstm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent           # ml/
ROOT = HERE.parent                                # energy_modelling/
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                       # noqa: E402
import models as M                    # noqa: E402
import evaluate as ev                 # noqa: E402
import plotting                       # noqa: E402
import stack_plots                    # noqa: E402

NAME = "lstm_5min_mse"
INPUT_MODE = "net_dispatch_totdem"
TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]


def write_document(dst, metrics, hist, n_params, n_feats):
    a = metrics["average"]
    lines = ["Data preprocessing:\n"]
    lines.append("1. VIC market + generation, 5-min native; interpolate small battery gaps.")
    lines.append("2. Six non-negative targets: hydro, coal_brown, gas_steam, gas_ocgt, battery_charging, battery_discharging.")
    lines.append(f"3. Input mode net_dispatch_totdem ({n_feats} columns): dispatchable history + net_demand + total demand_mw + price + 8 calendar.")
    lines.append("4. Global StandardScaler fit on train only (NO per-window RevIN).")
    lines.append("5. Chronological 80-10-10 split; sliding window 24h (288 steps); target 5min ahead (horizon 1 step).\n")
    lines.append("Model Use:\n")
    lines.append("1. Plain 2-layer LSTM (hidden 128, dropout 0.2), linear head on the final step. No RevIN, so it sees absolute demand levels.")
    lines.append("2. Loss = MSE only.")
    lines.append("3. AdamW, early stopping on validation MSE.")
    lines.append(f"4. Trainable parameters: {n_params:,}. Stopped at epoch {hist['epochs_run']}.\n")
    lines.append("Hyperparameter:\n")
    lines.append("1. Lookback 24h (288 steps), horizon 1 step (5min).")
    lines.append("2. Batch 128, lr 0.0001, weight decay 1e-05.")
    lines.append("3. Max epochs 200, patience 20, seed 0.\n")
    lines.append("Results:\n")
    lines.append("Per-target metrics on the 5-min test set.\n")
    rows = ["| Energy | MAE | RMSE | WAPE | R^2 |", "|---|---|---|---|---|"]
    for t, v in metrics["per_target"].items():
        rows.append(f"| {t} | {v['MAE']:.2f} | {v['RMSE']:.2f} | {v['WAPE']:.4f} | {v['R2']:.4f} |")
    rows.append(f"| average | {a['MAE']:.2f} | {a['RMSE']:.2f} | {a['WAPE']:.4f} | {a['R2']:.4f} |")
    lines.append("\n".join(rows) + "\n")
    (dst / "document.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data = pipeline.prepare(resolution="5min", lookback=24, horizon=1,
                            input_mode=INPUT_MODE, save=False)
    n_feats = data["Xtr"].shape[2]
    print(f"  Xtr {data['Xtr'].shape}  Xva {data['Xva'].shape}  Xte {data['Xte'].shape}")

    dst = HERE / NAME
    figdir = dst / "figure"
    figdir.mkdir(parents=True, exist_ok=True)

    ev._seed_all(0)
    model = M.make_neural("lstm", n_features=n_feats, n_targets=len(TARGETS))
    n_params = ev.count_params(model)
    print(f"  params={n_params:,}")
    spec = M.TrainSpec()
    model, hist = ev.train_neural(model, data, spec, device, NAME)

    true = data["y_scaler"].inverse_transform(data["Yte"])
    pred = data["y_scaler"].inverse_transform(ev.predict_neural(model, data["Xte"], device))
    metrics = ev.compute_metrics(true, pred, TARGETS)
    metrics.update({"arch": NAME, "epochs_run": hist["epochs_run"]})

    # ---- persist weights / scalers / metrics / predictions ----
    torch.save(model.state_dict(), dst / f"{NAME}.pt")
    xs, ys = data["x_scaler"], data["y_scaler"]
    np.savez(dst / "scalers.npz", x_mean=xs.mean_, x_scale=xs.scale_,
             y_mean=ys.mean_, y_scale=ys.scale_,
             feat_cols=np.array(data["feat_cols"]), targets=np.array(TARGETS))
    (dst / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (dst / "history.json").write_text(json.dumps(hist, indent=2))
    test_index = pd.DatetimeIndex(data["test_index"])
    out = pd.DataFrame({"interval": test_index.astype(str)})
    for i, t in enumerate(TARGETS):
        out[f"{t}_actual"] = true[:, i]; out[f"{t}_pred"] = pred[:, i]
    out.to_csv(dst / "predictions.csv", index=False)

    # ---- figures: loss curve + all-energy stacked (4-day window + full test) ----
    plotting.plot_loss(hist, figdir, tag=NAME)
    ren = stack_plots.load_renewables(test_index)
    days, pos = stack_plots.pick_full_days(test_index, n_days=4)
    stack_plots.fig_stacked_window(test_index[pos], true[pos], pred[pos], ren.iloc[pos],
                                   figdir / "stacked_all_energy_4day.png", NAME)
    stack_plots.fig_stacked_full(test_index, true, pred, ren,
                                 figdir / "stacked_all_energy_full.png", NAME)
    write_document(dst, metrics, hist, n_params, n_feats)

    a = metrics["average"]
    print(f"  [{NAME}] MAE={a['MAE']:.2f} RMSE={a['RMSE']:.2f} "
          f"WAPE={a['WAPE']*100:.2f}% R2={a['R2']:.4f}  -> {dst}")


if __name__ == "__main__":
    main()
