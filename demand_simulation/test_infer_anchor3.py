"""Inference-only anchoring, v3: ReLU positivity + nosteam rescale set.

test_infer_anchor2.py's softplus-only diagnostic showed most of the gas_steam
regression was the SOFTPLUS, not the rescale: steam is off for most of the test
set, so softplus's ~+0.7 MW floor on near-zero predictions is pure error on a
channel whose WAPE denominator is tiny. Softplus is only needed to TRAIN
through the head; for inference-only anchoring ReLU gives exact zeros (and is
what sc.predict applies downstream anyway).

Rows per base: bare unclamped (recorded baseline), bare clamped (fair
production baseline -- downstream sims clamp), all5+relu, nosteam+relu.
Goal: nosteam+relu ~= bare clamped WAPE with the full structural response.

    python demand_simulation/test_infer_anchor3.py
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

MODELS = [("lstm_revin", ROOT / "ml" / "lstm_revin_5min" / "lstm_revin_5min.pt"),
          ("itransformer", ROOT / "ml" / "itransformer_totdem" / "itransformer_totdem.pt")]
CONFIGS = [("all5", [0, 1, 2, 3, 5]), ("nosteam", [0, 1, 3, 5])]
FREE = (11, 14)
REBOUND, REDUCTION = 20.0, 10.0
OUT = HERE / "sweep_eqnd"


def windows(frame, fc, xs, lb):
    X = xs.transform(frame[fc].values).astype(np.float32)
    Xw, _ = pipeline.make_windows(X, np.zeros((len(X), 6), np.float32), lb, 1)
    return Xw


def demand_response(model, Xb, Xs, ys, mask, device):
    pb = sc.predict(model, Xb, ys, device, batch=256)[mask]
    ps = sc.predict(model, Xs, ys, device, batch=256)[mask]
    out = {}
    for i, t in enumerate(sc.TARGETS):
        m = pb[:, i].mean(); out[t] = 100 * (ps[:, i].mean() - m) / m if m else float("nan")
    nb, ns = sc.net_demand(pb).mean(), sc.net_demand(ps).mean()
    out["net_demand"] = 100 * (ns - nb) / nb
    return out


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nd_idx = fc.index("net_demand")

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

    Xb = windows(base.loc[full_idx], fc, xs, lb)
    Xs = windows(scen.loc[full_idx], fc, xs, lb)
    true_te = ys.inverse_transform(data["Yte"])

    y_st = true_te[:, sc.TARGETS.index("gas_steam")]
    print(f"gas_steam test actuals: mean|y|={np.abs(y_st).mean():.2f} MW, "
          f"on(>1 MW)={100 * (y_st > 1.0).mean():.1f}% of steps, max={y_st.max():.0f} MW")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, weights in MODELS:
        bare = M.make_neural(name, n_features=len(fc), n_targets=6)
        bare.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        bare = bare.to(device).eval()

        pred_bare = ys.inverse_transform(ev.predict_neural(bare, data["Xte"], device, batch=256))
        r_bare = demand_response(bare, Xb, Xs, ys, mask, device)
        rows.append((f"{name}", ev.compute_metrics(true_te, pred_bare, sc.TARGETS), r_bare))
        rows.append((f"{name}+clamp", ev.compute_metrics(true_te, np.maximum(pred_bare, 0.0), sc.TARGETS), r_bare))

        for cfg, ridx in CONFIGS:
            head = M.DemandAnchoredHead(bare, xs.mean_[nd_idx], xs.scale_[nd_idx],
                                        ys.mean_, ys.scale_, nd_feat_idx=nd_idx,
                                        rescale_idx=ridx, pos_fn="relu").to(device).eval()
            t0 = time.time()
            pred = ys.inverse_transform(ev.predict_neural(head, data["Xte"], device, batch=256))
            met = ev.compute_metrics(true_te, pred, sc.TARGETS)
            r = demand_response(head, Xb, Xs, ys, mask, device)
            rows.append((f"{name}+{cfg}_relu", met, r))
            print(f"  {name}+{cfg}_relu  R2={met['average']['R2']:.4f} WAPE={met['average']['WAPE']:.4f} "
                  f"net_demand={r['net_demand']:+.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    hdr = ["model", "R2", "WAPE", "hydro", "coal", "gas_steam", "gas_ocgt", "batt_chg", "batt_dis",
           "nd_resp", "coal_resp", "hydro_resp", "ocgt_resp", "batt_dis_resp"]
    lines = ["# Inference-only anchoring v3: ReLU positivity, nosteam rescale set\n",
             f"teacher-forced reb{REBOUND:.0f}/red{REDUCTION:.0f}, demand input {dem_in:+.1f}% "
             f"(structural full response = +179.2%); per-target columns are WAPE\n",
             "| " + " | ".join(hdr) + " |",
             "| " + " | ".join("---" for _ in hdr) + " |"]
    for tag, met, r in rows:
        p = met["per_target"]; avg = met["average"]
        lines.append(
            f"| {tag} | {avg['R2']:.4f} | {avg['WAPE']:.4f} | "
            f"{p['hydro']['WAPE']:.3f} | {p['coal_brown']['WAPE']:.3f} | {p['gas_steam']['WAPE']:.3f} | "
            f"{p['gas_ocgt']['WAPE']:.3f} | {p['battery_charging']['WAPE']:.3f} | {p['battery_discharging']['WAPE']:.3f} | "
            f"{r['net_demand']:+.1f}% | {r['coal_brown']:+.1f}% | {r['hydro']:+.1f}% | "
            f"{r['gas_ocgt']:+.1f}% | {r['battery_discharging']:+.1f}% |")
    (OUT / "infer_anchor3_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "infer_anchor3_compare.md")


if __name__ == "__main__":
    main()
