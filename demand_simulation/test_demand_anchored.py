"""Demand-anchored head: force sum(SIGN*pred) == net_demand by construction, so
the response is structural and cannot decay with training -- unlike RevIN,
SelectiveRevIN, Dish-TS, and the soft energy-balance loss, which all left the
model an escape hatch to hit low MSE without ever reading demand (saliency
stayed ~0%, see demand_simulation/findings.md). The backbone only has to learn
the fuel MIX (5 free degrees of freedom); DemandAnchoredHead (lib/models.py)
rescales the TOTAL after the fact to match net_demand read straight off the
input window. Trains on the Mac GPU (MPS).

  lstm_demandhead        DemandAnchoredHead wrapping a plain (no-RevIN) LSTM
  lstm_revin_demandhead  DemandAnchoredHead wrapping RevIN(LSTM) -- RevIN helps
                         the backbone learn shape; the head fixes RevIN's level

For each: test R^2 / WAPE (accuracy) and one-step teacher-forced demand
response (reb20/red10). Compare against the bundled no-RevIN LSTM (+62.1%
response, WAPE 0.2024) and +RevIN LSTM/iTransformer (~0% response, WAPE
~0.11-0.13) -- goal is WAPE ~0.10 AND response comparable to the no-RevIN LSTM.

    python demand_simulation/test_demand_anchored.py
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

MODELS = [("lstm_demandhead", "lstm"), ("lstm_revin_demandhead", "lstm_revin")]
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
    Xb = windows(base.loc[full_idx], fc, xs, lb)
    Xs = windows(scen.loc[full_idx], fc, xs, lb)
    pb = sc.predict(model, Xb, ys, device, batch=256)[mask]
    ps = sc.predict(model, Xs, ys, device, batch=256)[mask]
    out = {}
    for i, t in enumerate(sc.TARGETS):
        m = pb[:, i].mean(); out[t] = 100 * (ps[:, i].mean() - m) / m if m else float("nan")
    nb, ns = sc.net_demand(pb).mean(), sc.net_demand(ps).mean()
    out["net_demand"] = 100 * (ns - nb) / nb
    return out


def _val_mse(model, Xva, Yva, device, batch=256):
    """Batched no-grad val MSE (a single full-set forward would OOM on MPS)."""
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
        print(f"    epoch {ep:02d}  val_mse={vmse:.5f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, ep


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nd_idx = fc.index("net_demand")
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
    dem_in = 100 * (scen.loc[ti, "demand_mw"].to_numpy()[mask].mean() / base.loc[ti, "demand_mw"].to_numpy()[mask].mean() - 1)
    print(f"free-window demand input: {dem_in:+.1f}%")

    true_te = ys.inverse_transform(data["Yte"])
    rows = []
    for name, base_arch in MODELS:
        t0 = time.time()
        ev._seed_all(0)
        model = M.make_demand_anchored(base_arch, xs, ys, nd_feat_idx=nd_idx,
                                       n_features=len(fc), n_targets=6).to(device)
        print(f"  training {name} (base={base_arch}) ...", flush=True)
        model, vmse, ep = train_mse(model, Xtr, Ytr, Xva, Yva, device)
        dst = ROOT / "ml" / f"{name}_5min"; dst.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), dst / f"{name}_5min.pt")
        pred_te = ys.inverse_transform(ev.predict_neural(model, data["Xte"], device, batch=256))
        m = ev.compute_metrics(true_te, pred_te, sc.TARGETS)["average"]
        r = demand_response(model, base, scen, fc, xs, ys, lb, full_idx, mask, device)
        secs = time.time() - t0
        rows.append((name, m["R2"], m["WAPE"], r, ep, secs))
        print(f"  {name:24s} R2={m['R2']:.4f} WAPE={m['WAPE']:.4f} "
              f"net_demand={r['net_demand']:+.1f}% coal={r['coal_brown']:+.1f}% "
              f"hydro={r['hydro']:+.1f}% gas_ocgt={r['gas_ocgt']:+.1f}% "
              f"batt_dis={r['battery_discharging']:+.1f}% batt_chg={r['battery_charging']:+.1f}% "
              f"({secs:.0f}s, ep{ep})\n", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    hdr = ["model", "R2", "WAPE", "net_demand", "coal", "hydro", "gas_ocgt", "batt_dis", "batt_chg", "epochs", "secs"]
    lines = ["# Demand-anchored head vs RevIN/Dish-TS/EB-loss (MPS, plain MSE)\n",
             f"stride {TRAIN_STRIDE}, {len(Xtr)} train windows, early-stop patience {PATIENCE}, "
             f"demand input {dem_in:+.1f}%\n",
             "| " + " | ".join(hdr) + " |",
             "| " + " | ".join("---" for _ in hdr) + " |"]
    for name, r2, wape, r, ep, secs in rows:
        lines.append(f"| {name} | {r2:.4f} | {wape:.4f} | {r['net_demand']:+.1f}% | "
                     f"{r['coal_brown']:+.1f}% | {r['hydro']:+.1f}% | {r['gas_ocgt']:+.1f}% | "
                     f"{r['battery_discharging']:+.1f}% | {r['battery_charging']:+.1f}% | {ep} | {secs:.0f} |")
    (OUT / "demand_anchored_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "demand_anchored_compare.md")


if __name__ == "__main__":
    main()
