"""Dish-TS vs RevIN: can a learned (predicted) horizon level keep WAPE while
responding to a demand shift? Trains on the Mac GPU (MPS).

RevIN copies the lookback mean onto the output, so the level is anchored to dispatch
history and a demand shift can't move it. Dish-TS (Fan et al., AAAI 2023) predicts the
horizon level with a learned coefficient net. We test what the level-predictor is
allowed to see:

  lstm_revin     RevIN baseline (copied level)
  lstm_dishts    Dish-TS 'own'   : level from each target's own history (channel-indep)
  lstm_dishtsx   Dish-TS 'cross' : level from ALL channels' recent mean (free to use demand)
  lstm_dishtsd   Dish-TS 'exo'   : level from NON-target channels only (forced through demand)

For each: test R^2 / WAPE (accuracy) and one-step teacher-forced demand response
(reb20/red10). Plain MSE training (the soft-constraint approach is abandoned).

    python demand_simulation/test_dishts.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402
import sim_common as sc         # noqa: E402
import models as M              # noqa: E402
import evaluate as ev           # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

MODELS = [("LSTM +RevIN", "lstm_revin"),
          ("Dish-TS own", "lstm_dishts"),
          ("Dish-TS cross", "lstm_dishtsx"),
          ("Dish-TS exo", "lstm_dishtsd")]
TRAIN_STRIDE = 12
EPOCHS, PATIENCE, BATCH = 30, 5, 64
FREE = (11, 14)
REBOUND, REDUCTION = 20.0, 10.0
OUT = HERE / "sweep_eqnd"


def windows(frame, fc, xs, lb):
    X = xs.transform(frame[fc].values).astype(np.float32)
    Xw, _ = pipeline.make_windows(X, np.zeros((len(X), 6), np.float32), lb, 1)
    return Xw


def demand_response(model, base, scen, fc, xs, ys, lb, full_idx, mask, device):
    pb = sc.predict(model, windows(base.loc[full_idx], fc, xs, lb), ys, device, batch=256)[mask]
    ps = sc.predict(model, windows(scen.loc[full_idx], fc, xs, lb), ys, device, batch=256)[mask]
    nb, ns = sc.net_demand(pb).mean(), sc.net_demand(ps).mean()
    coal = lambda a: a[:, 1].mean()
    return {"net_demand": 100 * (ns - nb) / nb,
            "coal": 100 * (coal(ps) - coal(pb)) / coal(pb),
            "batt_dis": 100 * (ps[:, 5].mean() - pb[:, 5].mean()) / pb[:, 5].mean()}


def _val_mse(model, Xva, Yva, device, batch=256):
    mse = torch.nn.MSELoss(reduction="sum"); tot, n = 0.0, len(Xva)
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.from_numpy(Xva[i:i + batch]).to(device)
            yb = torch.from_numpy(Yva[i:i + batch]).to(device)
            tot += mse(model(xb), yb).item()
    return tot / (n * Yva.shape[1])


def train_mse(model, Xtr, Ytr, Xva, Yva, device):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    mse = torch.nn.MSELoss(); n = len(Xtr)
    best_val, best_state, waited, ep = float("inf"), None, 0, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(Ytr[idx]).to(device)
            opt.zero_grad(); loss = mse(model(xb), yb); loss.backward(); opt.step()
        vmse = _val_mse(model, Xva, Yva, device)
        if vmse < best_val - 1e-6:
            best_val, best_state, waited = vmse, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
        print(f"    {model.xi_mode if hasattr(model,'xi_mode') else 'revin':6s} epoch {ep:02d}  val_mse={vmse:.5f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, ep


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]
    Xtr, Ytr = data["Xtr"][::TRAIN_STRIDE], data["Ytr"][::TRAIN_STRIDE]
    Xva, Yva = data["Xva"][::TRAIN_STRIDE], data["Yva"][::TRAIN_STRIDE]
    print(f"train windows {len(data['Xtr'])} -> {len(Xtr)} (stride {TRAIN_STRIDE})")

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

    true_te = ys.inverse_transform(data["Yte"])
    rows = []
    for name, arch in MODELS:
        t0 = time.time()
        ev._seed_all(0)
        model = M.make_neural(arch, len(fc), 6).to(device)
        print(f"  training {name} ({arch}) ...", flush=True)
        model, vmse, ep = train_mse(model, Xtr, Ytr, Xva, Yva, device)
        pred_te = ys.inverse_transform(ev.predict_neural(model, data["Xte"], device, batch=256))
        m = ev.compute_metrics(true_te, pred_te, sc.TARGETS)["average"]
        r = demand_response(model, base, scen, fc, xs, ys, lb, full_idx, mask, device)
        rows.append((name, m["R2"], m["WAPE"], r["net_demand"], r["coal"], r["batt_dis"], ep))
        print(f"  {name:16s} R2={m['R2']:.4f} WAPE={m['WAPE']:.4f} "
              f"net_demand={r['net_demand']:+.1f}% batt_dis={r['batt_dis']:+.1f}%  ({time.time()-t0:.0f}s, ep{ep})\n", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    hdr = ["model", "R2", "WAPE", "net_demand_resp", "coal_resp", "batt_dis_resp", "epochs"]
    lines = ["# Dish-TS vs RevIN (learned horizon level, MPS, plain MSE)\n",
             f"stride {TRAIN_STRIDE}, {len(Xtr)} train windows, early-stop patience {PATIENCE}, demand input +101%\n",
             "| " + " | ".join(hdr) + " |",
             "| " + " | ".join("---" for _ in hdr) + " |"]
    for name, r2, wape, ndr, coal, bd, ep in rows:
        lines.append(f"| {name} | {r2:.4f} | {wape:.4f} | {ndr:+.1f}% | {coal:+.1f}% | {bd:+.1f}% | {ep} |")
    (OUT / "dishts_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "dishts_compare.md")


if __name__ == "__main__":
    main()
