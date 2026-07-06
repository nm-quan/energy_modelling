"""Controlled test: is RevIN (not the architecture) the reason the model ignores
a demand shift? Train an LSTM *with* RevIN on the same pipeline, then compare its
demand response to the existing no-RevIN LSTM.

Logic: same backbone (2-layer LSTM, hidden 128), only RevIN differs. If the RevIN
LSTM forecasts well (comparable test R^2) but does NOT respond to the demand shift,
while the no-RevIN LSTM does, then RevIN -- not transformer/LSTM capacity or
undertraining -- is the cause.

Trains to early-stop (8 threads), saves weights to ml/lstm_revin_5min/, prints test
accuracy for both models and the one-step teacher-forced demand response (reb20/red10).

    python demand_simulation/test_lstm_revin.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402
import sim_common as sc         # noqa: E402
import models as M              # noqa: E402
import evaluate as ev           # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

FREE = (11, 14)
REBOUND, REDUCTION = 20.0, 10.0


def windows(frame, fc, xs, lb):
    X = xs.transform(frame[fc].values).astype(np.float32)
    Xw, _ = pipeline.make_windows(X, np.zeros((len(X), 6), np.float32), lb, 1)
    return Xw


def demand_response(model, base, scen, fc, xs, ys, lb, full_idx, ti, mask, device):
    Xb = windows(base.loc[full_idx], fc, xs, lb)
    Xs = windows(scen.loc[full_idx], fc, xs, lb)
    pb = sc.predict(model, Xb, ys, device)[mask]
    ps = sc.predict(model, Xs, ys, device)[mask]
    out = {}
    for i, t in enumerate(sc.TARGETS):
        m = pb[:, i].mean(); out[t] = 100 * (ps[:, i].mean() - m) / m if m else float("nan")
    out["net_demand"] = 100 * (sc.net_demand(ps).mean() - sc.net_demand(pb).mean()) / sc.net_demand(pb).mean()
    return out


def main():
    device = "cpu"
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]
    print(f"train windows {data['Xtr'].shape}, threads={torch.get_num_threads()}")

    # --- train LSTM + RevIN ---
    ev._seed_all(0)
    revin = M.make_neural("lstm_revin", len(fc), 6)
    spec = M.TrainSpec(epochs=40, patience=6)
    print("training lstm_revin (early stop, patience 6)...")
    revin, hist = ev.train_neural(revin, data, spec, device, "lstm_revin")
    dst = ROOT / "ml" / "lstm_revin_5min"; dst.mkdir(parents=True, exist_ok=True)
    torch.save(revin.state_dict(), dst / "lstm_revin_5min.pt")
    print(f"saved -> {dst}  (stopped epoch {hist['epochs_run']}, best val_mse {hist['best_val_mse']:.5f})")

    lstm = sc.load_lstm("lstm_5min_mse", len(fc), device)   # existing no-RevIN, converged

    # --- test accuracy (rule out undertraining) ---
    true = ys.inverse_transform(data["Yte"])
    for name, mdl in [("LSTM no-RevIN (converged)", lstm), ("LSTM +RevIN (new)", revin)]:
        pred = ys.inverse_transform(ev.predict_neural(mdl, data["Xte"], device))
        m = ev.compute_metrics(true, pred, sc.TARGETS)["average"]
        print(f"  accuracy {name:28s}: MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  R2={m['R2']:.4f}")

    # --- demand response (same as model_response_tf) ---
    df = pipeline.build_table("5min")
    inter = pd.read_parquet(pipeline.DATA_DIR / "vic_interconnector_last365.parquet")
    ni = inter.set_index("interval").sort_index()["net_import_mw"].reindex(df.index).interpolate(limit_direction="both")
    val = df[(df.index > pipeline.TRAIN_END) & (df.index <= pipeline.VAL_END)]
    test = df[df.index > pipeline.VAL_END]; ti = test.index
    full_idx = pd.concat([val.tail(lb), test]).index
    mask = sc.response_mask(ti, 1, FREE)
    def nd(f): return f["demand_mw"] - f["wind"] - f["solar_utility"] - ni.reindex(f.index)
    base = df.copy(); base["net_demand"] = nd(base)
    scen = FixedPercentageShift(REBOUND, REDUCTION, free_hours=FREE).transform(df); scen["net_demand"] = nd(scen)
    dem_in = 100 * (scen.loc[ti, "demand_mw"].to_numpy()[mask].mean() / base.loc[ti, "demand_mw"].to_numpy()[mask].mean() - 1)

    print(f"\none-step teacher-forced demand response, reb{REBOUND:g}%/red{REDUCTION:g}% "
          f"(free-window demand input +{dem_in:.0f}%)\n")
    hdr = f"{'model':28s} {'net_demand':>11s} {'coal':>8s} {'hydro':>9s} {'gas_ocgt':>9s} {'batt_dis':>9s}"
    print(hdr); print("-" * len(hdr))
    for name, mdl in [("LSTM no-RevIN (converged)", lstm), ("LSTM +RevIN (new)", revin)]:
        r = demand_response(mdl, base, scen, fc, xs, ys, lb, full_idx, ti, mask, device)
        print(f"{name:28s} {r['net_demand']:>10.1f}% {r['coal_brown']:>7.1f}% "
              f"{r['hydro']:>8.1f}% {r['gas_ocgt']:>8.1f}% {r['battery_discharging']:>8.1f}%")


if __name__ == "__main__":
    main()
