"""Selective-RevIN fix: keep RevIN's accuracy, restore the demand response.

Standard RevIN normalises EVERY channel per-window, erasing the absolute level
of the demand drivers (net_demand, demand_mw) -> the LSTM stops reacting to a
demand shift (response collapses ~16x). SelectiveRevIN (`lib/models.py`) passes
the two demand channels [6,7] through on the frozen global scaler (absolute level
preserved) while still RevIN-normalising the dispatch-history channels (the part
that buys accuracy).

This trains `lstm_selrevin` on the same pipeline/recipe as test_lstm_revin.py and
compares it head-to-head with the bundled no-RevIN LSTM and the +RevIN LSTM on:
  (1) test accuracy (R^2 / MAE / macro WAPE) — should match +RevIN, not no-RevIN,
  (2) one-step teacher-forced demand response (reb20/red10) — should be large.

    python demand_simulation/test_lstm_selrevin.py
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
# 5-min windows overlap by 287/288 steps, so adjacent windows are ~redundant.
# Train/val on hourly-strided windows to cut a multi-hour CPU job to ~20 min
# without changing model/lookback/data span. Test set stays full for fair metrics.
TRAIN_STRIDE = 12


def windows(frame, fc, xs, lb):
    X = xs.transform(frame[fc].values).astype(np.float32)
    Xw, _ = pipeline.make_windows(X, np.zeros((len(X), 6), np.float32), lb, 1)
    return Xw


def demand_response(model, base, scen, fc, xs, ys, lb, full_idx, mask, device):
    Xb = windows(base.loc[full_idx], fc, xs, lb)
    Xs = windows(scen.loc[full_idx], fc, xs, lb)
    pb = sc.predict(model, Xb, ys, device)[mask]
    ps = sc.predict(model, Xs, ys, device)[mask]
    out = {}
    for i, t in enumerate(sc.TARGETS):
        m = pb[:, i].mean(); out[t] = 100 * (ps[:, i].mean() - m) / m if m else float("nan")
    nb, ns = sc.net_demand(pb).mean(), sc.net_demand(ps).mean()
    out["net_demand"] = 100 * (ns - nb) / nb
    return out


def main():
    device = "cpu"
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]
    n_full = len(data["Xtr"])
    for k in ("Xtr", "Ytr", "Xva", "Yva"):
        data[k] = data[k][::TRAIN_STRIDE]
    print(f"train windows {n_full} -> {len(data['Xtr'])} (stride {TRAIN_STRIDE}), "
          f"threads={torch.get_num_threads()}")

    # --- train LSTM + selective RevIN (demand channels passthrough) ---
    ev._seed_all(0)
    sel = M.make_neural("lstm_selrevin", len(fc), 6)
    spec = M.TrainSpec(epochs=40, patience=6)
    print("training lstm_selrevin (early stop, patience 6)...")
    sel, hist = ev.train_neural(sel, data, spec, device, "lstm_selrevin")
    dst = ROOT / "ml" / "lstm_selrevin_5min"; dst.mkdir(parents=True, exist_ok=True)
    torch.save(sel.state_dict(), dst / "lstm_selrevin_5min.pt")
    print(f"saved -> {dst}  (stopped epoch {hist['epochs_run']}, best val_mse {hist['best_val_mse']:.5f})")

    # --- the two reference models (already trained) ---
    norevin = sc.load_lstm("lstm_5min_mse", len(fc), device)        # responsive, less accurate
    revin = M.make_neural("lstm_revin", len(fc), 6)
    revin.load_state_dict(torch.load(ROOT / "ml" / "lstm_revin_5min" / "lstm_revin_5min.pt",
                                     map_location=device))
    revin = revin.to(device).eval()

    models = [("LSTM no-RevIN", norevin),
              ("LSTM +RevIN", revin),
              ("LSTM selective-RevIN", sel)]

    # --- (1) test accuracy ---
    true = ys.inverse_transform(data["Yte"])
    print("\n(1) test accuracy")
    print(f"  {'model':22s} {'R2':>7s} {'MAE':>7s} {'WAPE':>8s}")
    for name, mdl in models:
        pred = ys.inverse_transform(ev.predict_neural(mdl, data["Xte"], device))
        m = ev.compute_metrics(true, pred, sc.TARGETS)["average"]
        print(f"  {name:22s} {m['R2']:>7.4f} {m['MAE']:>7.1f} {m['WAPE']:>8.4f}")

    # --- shift scenario inputs ---
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

    # --- (2) one-step teacher-forced demand response ---
    print(f"\n(2) one-step teacher-forced demand response, reb{REBOUND:g}%/red{REDUCTION:g}% "
          f"(free-window demand input +{dem_in:.0f}%)")
    hdr = f"  {'model':22s} {'net_demand':>11s} {'coal':>8s} {'hydro':>9s} {'gas_ocgt':>9s} {'batt_dis':>9s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, mdl in models:
        r = demand_response(mdl, base, scen, fc, xs, ys, lb, full_idx, mask, device)
        print(f"  {name:22s} {r['net_demand']:>10.1f}% {r['coal_brown']:>7.1f}% "
              f"{r['hydro']:>8.1f}% {r['gas_ocgt']:>8.1f}% {r['battery_discharging']:>8.1f}%")


if __name__ == "__main__":
    main()
