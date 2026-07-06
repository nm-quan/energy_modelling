"""Demand-anchored head, FULL (unstrided) convergence run.

The strided pilot (test_demand_anchored.py, 6840 windows / 29 epochs) showed:
  - net_demand response is IDENTICAL (+179.2%) regardless of base architecture --
    confirms the response is structural (set by the head), not learned.
  - lstm_revin_demandhead already beat the strided "LSTM +RevIN" baseline on
    BOTH accuracy (WAPE 0.2186 vs 0.2411, R^2 0.9499 vs 0.9418) and response
    (+179% vs +21%) at the SAME training budget.

But strided numbers understate accuracy badly in this project (e.g. converged
RevIN-LSTM WAPE 0.1252 vs its own strided 0.2411 -- see findings.md). This runs
both demand-anchored variants to convergence on the full 82k-window train set:

  lstm_revin_demandhead   base=lstm_revin   (recipe matches test_lstm_revin.py:
                          epochs=40, patience=6, for apples-to-apples accuracy
                          comparison against the bundled lstm_revin, WAPE 0.1252)
  itransformer_demandhead base=itransformer (RevIN on; best standalone WAPE of
                          any model tried so far was iTransformer+RevIN, 0.1132
                          -- if the head's structural response transfers to this
                          backbone too, it would combine the best accuracy with
                          full responsiveness)

Goal: WAPE ~0.10 AND demand response comparable to (or beyond) the no-RevIN
LSTM's +62.1%.

    python demand_simulation/test_demand_anchored_full.py
"""
from __future__ import annotations

import json
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

MODELS = [("lstm_revin_demandhead", "lstm_revin", 40, 6, 128),
          ("itransformer_demandhead", "itransformer", 60, 10, 64)]
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


def _val_mse(model, Xva, Yva, device, batch):
    """Batched no-grad val MSE (a single full-set forward would OOM on MPS)."""
    mse = torch.nn.MSELoss(reduction="sum"); tot, n = 0.0, len(Xva)
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.from_numpy(Xva[i:i + batch]).to(device)
            yb = torch.from_numpy(Yva[i:i + batch]).to(device)
            tot += mse(model(xb), yb).item()
    return tot / (n * Yva.shape[1])


def train_mse(model, Xtr, Ytr, Xva, Yva, device, epochs, patience, batch, ckpt_path=None):
    """AdamW + MSE with early stopping. Checkpoints (current weights, best-so-far
    weights, epoch, patience counter) after EVERY epoch to ckpt_path so a killed
    background job can resume instead of losing all progress -- a >30-epoch full
    (82k-window) run takes 40-90 min and was killed once with nothing saved."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    mse = torch.nn.MSELoss(); n = len(Xtr)
    best_val, best_state, waited, start_ep = float("inf"), None, 0, 1

    if ckpt_path is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        best_val, best_state, waited = ck["best_val"], ck["best_state"], ck["waited"]
        start_ep = ck["epoch"] + 1
        opt.load_state_dict(ck["opt_state"])
        print(f"    resumed from checkpoint at epoch {ck['epoch']} (best_val={best_val:.5f})", flush=True)

    ep = start_ep - 1
    for ep in range(start_ep, epochs + 1):
        t_ep = time.time()
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(Ytr[idx]).to(device)
            opt.zero_grad(); loss = mse(model(xb), yb); loss.backward(); opt.step()
        vmse = _val_mse(model, Xva, Yva, device, batch)
        stop = False
        if vmse < best_val - 1e-6:
            best_val, best_state, waited = vmse, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            stop = waited >= patience
        if ckpt_path is not None:
            torch.save({"epoch": ep, "best_val": best_val, "best_state": best_state,
                       "waited": waited, "model_state": model.state_dict(),
                       "opt_state": opt.state_dict()}, ckpt_path)
        tag = "  early stop" if stop else ""
        print(f"    epoch {ep:02d}  val_mse={vmse:.5f}  ({time.time()-t_ep:.0f}s){tag}", flush=True)
        if stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()
    return model, best_val, ep


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    models = [m for m in MODELS if m[0] == only] if only else MODELS
    if only and not models:
        raise SystemExit(f"unknown model name {only!r}; choices: {[m[0] for m in MODELS]}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nd_idx = fc.index("net_demand")
    Xtr, Ytr = data["Xtr"], data["Ytr"]
    Xva, Yva = data["Xva"], data["Yva"]
    print(f"train windows {len(Xtr)} (full, no stride)")

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
    OUT.mkdir(parents=True, exist_ok=True)
    for name, base_arch, epochs, patience, batch in models:
        t0 = time.time()
        ev._seed_all(0)
        model = M.make_demand_anchored(base_arch, xs, ys, nd_feat_idx=nd_idx,
                                       n_features=len(fc), n_targets=6).to(device)
        dst = ROOT / "ml" / f"{name}_5min"; dst.mkdir(parents=True, exist_ok=True)
        ckpt_path = dst / f"{name}_ckpt.pt"
        print(f"  training {name} (base={base_arch}, epochs={epochs}, patience={patience}, batch={batch}) ...",
              flush=True)
        model, vmse, ep = train_mse(model, Xtr, Ytr, Xva, Yva, device, epochs, patience, batch, ckpt_path)
        torch.save(model.state_dict(), dst / f"{name}_5min.pt")
        pred_te = ys.inverse_transform(ev.predict_neural(model, data["Xte"], device, batch=256))
        m = ev.compute_metrics(true_te, pred_te, sc.TARGETS)["average"]
        r = demand_response(model, base, scen, fc, xs, ys, lb, full_idx, mask, device)
        secs = time.time() - t0
        print(f"  {name:24s} R2={m['R2']:.4f} WAPE={m['WAPE']:.4f} "
              f"net_demand={r['net_demand']:+.1f}% coal={r['coal_brown']:+.1f}% "
              f"hydro={r['hydro']:+.1f}% gas_ocgt={r['gas_ocgt']:+.1f}% "
              f"batt_dis={r['battery_discharging']:+.1f}% batt_chg={r['battery_charging']:+.1f}% "
              f"(best val_mse={vmse:.5f}, {secs:.0f}s, ep{ep})\n", flush=True)

        # one JSON per model -- lets separate per-model runs (each its own short
        # background job, so a kill costs one model's progress, not all of them)
        # contribute to the same combined table without clobbering each other.
        result = {"model": name, "R2": m["R2"], "WAPE": m["WAPE"], "response": r,
                  "epochs": ep, "secs": secs, "dem_in": dem_in, "n_train": len(Xtr)}
        (OUT / f"demand_anchored_full_{name}.json").write_text(json.dumps(result, indent=2))
        _write_combined_table()

    print("wrote", OUT / "demand_anchored_full_compare.md")


def _write_combined_table():
    """Rebuild the combined markdown table from every per-model JSON on disk."""
    order = [m[0] for m in MODELS]
    results = []
    for name in order:
        p = OUT / f"demand_anchored_full_{name}.json"
        if p.exists():
            results.append(json.loads(p.read_text()))
    if not results:
        return
    hdr = ["model", "R2", "WAPE", "net_demand", "coal", "hydro", "gas_ocgt", "batt_dis", "batt_chg", "epochs", "secs"]
    lines = ["# Demand-anchored head, full unstrided convergence run\n",
             f"{results[0]['n_train']} train windows, demand input {results[0]['dem_in']:+.1f}%\n",
             "| " + " | ".join(hdr) + " |",
             "| " + " | ".join("---" for _ in hdr) + " |"]
    for res in results:
        rr = res["response"]
        lines.append(f"| {res['model']} | {res['R2']:.4f} | {res['WAPE']:.4f} | {rr['net_demand']:+.1f}% | "
                     f"{rr['coal_brown']:+.1f}% | {rr['hydro']:+.1f}% | {rr['gas_ocgt']:+.1f}% | "
                     f"{rr['battery_discharging']:+.1f}% | {rr['battery_charging']:+.1f}% | "
                     f"{res['epochs']} | {res['secs']:.0f} |")
    (OUT / "demand_anchored_full_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
