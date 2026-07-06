"""Train an iTransformer WITHOUT RevIN on the Mac GPU (MPS), then run the demand
scenario for the full 2x2 (architecture x RevIN).

NOTE: a plain transformer with no input normalization is unstable -- with the
default LSTM recipe (lr 1e-4, no clipping) it diverges (train_mse -> inf/nan).
RevIN was doubling as input normalization that stabilized training. So here the
no-RevIN iTransformer uses a gentler recipe: lower LR + linear warmup + gradient
clipping. This is the apples-to-apples "remove only RevIN" model.

Completes the controlled matrix:
            no-RevIN                 +RevIN
  LSTM      lstm_5min_mse            lstm_revin_5min
  iTrans    itransformer_norevin     itransformer_totdem

Saves to ml/itransformer_norevin_5min/, prints test accuracy + one-step teacher-
forced demand response (reb20/red10) for all 4.

    python demand_simulation/test_norevin_itransformer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

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


def load(arch, weights, n_feats, device):
    m = M.make_neural(arch, n_feats, 6)
    m.load_state_dict(torch.load(weights, map_location=device))
    return m.to(device).eval()


def train_stable(model, data, device, base_lr=3e-5, warmup=800, clip=1.0,
                 epochs=60, patience=8, batch=128):
    """AdamW + linear warmup + grad clipping (stabilizes a no-RevIN transformer)."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    tr = DataLoader(TensorDataset(torch.from_numpy(data["Xtr"]), torch.from_numpy(data["Ytr"])),
                    batch_size=batch, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(data["Xva"]), torch.from_numpy(data["Yva"])),
                    batch_size=batch * 4, shuffle=False)
    best, best_state, waited, step = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train(); tl = []
        for xb, yb in tr:
            step += 1
            for g in opt.param_groups:                       # linear warmup
                g["lr"] = base_lr * min(1.0, step / warmup)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); tl.append(loss.item())
        model.eval(); vl = []
        with torch.no_grad():
            for xb, yb in va:
                vl.append(loss_fn(model(xb.to(device)), yb.to(device)).item())
        tloss, vloss = float(np.mean(tl)), float(np.mean(vl))
        print(f"  [it_norevin] epoch {epoch:03d}  train_mse={tloss:.5f}  val_mse={vloss:.5f}")
        if vloss < best - 1e-6:
            best, best_state, waited = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                print(f"  early stop at epoch {epoch} (best val_mse={best:.5f})"); break
    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_val_mse": best, "epochs_run": epoch}


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nf = len(fc)

    ev._seed_all(0)
    it_nr = M.make_neural("itransformer_norevin", nf, 6)
    print("training itransformer_norevin (MPS; lr warmup + grad clip)...")
    it_nr, hist = train_stable(it_nr, data, device)
    dst = ROOT / "ml" / "itransformer_norevin_5min"; dst.mkdir(parents=True, exist_ok=True)
    torch.save(it_nr.state_dict(), dst / "itransformer_norevin_5min.pt")
    print(f"saved -> {dst}  (epoch {hist['epochs_run']}, best val_mse {hist['best_val_mse']:.5f})")

    models = {
        "LSTM no-RevIN":      load("lstm", ROOT/"ml"/"lstm_5min_mse"/"lstm_5min_mse.pt", nf, device),
        "LSTM +RevIN":        load("lstm_revin", ROOT/"ml"/"lstm_revin_5min"/"lstm_revin_5min.pt", nf, device),
        "iTransformer +RevIN": load("itransformer", ROOT/"ml"/"itransformer_totdem"/"itransformer_totdem.pt", nf, device),
        "iTransformer no-RevIN": it_nr.eval(),
    }

    true = ys.inverse_transform(data["Yte"])
    print("\naccuracy (test set):")
    for name, mdl in models.items():
        pred = ys.inverse_transform(ev.predict_neural(mdl, data["Xte"], device))
        m = ev.compute_metrics(true, pred, sc.TARGETS)["average"]
        print(f"  {name:24s} MAE={m['MAE']:6.1f}  RMSE={m['RMSE']:6.1f}  R2={m['R2']:.4f}")

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
    Xb, Xs = windows(base.loc[full_idx], fc, xs, lb), windows(scen.loc[full_idx], fc, xs, lb)
    dem_in = 100 * (scen.loc[ti, "demand_mw"].to_numpy()[mask].mean() / base.loc[ti, "demand_mw"].to_numpy()[mask].mean() - 1)

    print(f"\none-step teacher-forced demand response, reb{REBOUND:g}%/red{REDUCTION:g}% "
          f"(free-window demand input +{dem_in:.0f}%)\n")
    hdr = f"{'model':24s} {'net_demand':>11s} {'coal':>8s} {'hydro':>9s} {'gas_ocgt':>9s} {'batt_dis':>9s}"
    print(hdr); print("-" * len(hdr))
    for name, mdl in models.items():
        pb = sc.predict(mdl, Xb, ys, device)[mask]
        ps = sc.predict(mdl, Xs, ys, device)[mask]
        def pct(i):
            m = pb[:, i].mean(); return 100 * (ps[:, i].mean() - m) / m if m else float("nan")
        ndp = 100 * (sc.net_demand(ps).mean() - sc.net_demand(pb).mean()) / sc.net_demand(pb).mean()
        print(f"{name:24s} {ndp:>10.1f}% {pct(1):>7.1f}% {pct(0):>8.1f}% {pct(3):>8.1f}% {pct(5):>8.1f}%")


if __name__ == "__main__":
    main()
