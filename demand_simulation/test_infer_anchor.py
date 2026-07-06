"""Inference-only demand anchoring: bolt DemandAnchoredHead onto the CONVERGED
blind checkpoints, no training through the head at all.

Rationale: DemandAnchoredHead is parameter-free -- "training through it" only
changes what the backbone converges to, and the full convergence run showed it
changes it for the worse (WAPE 0.1252 -> 0.3375, every channel 2-4x worse, even
batt_chg which the rescale never touches; its response also went NEGATIVE,
-5.3%). Leading explanation: `need = nd + chg_pred` makes batt_chg the only
free variable controlling the TOTAL, so MSE pressure distorts chg into a slack
absorber, and the 5 gen channels inherit the distorted total. None of that can
happen if the backbone is trained blind (plain MSE, no head) and the anchor is
applied only at eval: accuracy stays whatever the converged backbone earned,
and the response is still structurally exact (the head is an identity-enforcing
rescale at ANY weights).

Cost of anchoring at inference = the projection distance: each gen channel is
rescaled by need/gen_sum. Since actuals satisfy sum(SIGN*y) == net_demand
exactly (pipeline identity) and nd(t-1) ~ nd(t) at 5 min, the rescale moves
predictions TOWARD the truth manifold whenever the mix is roughly right.

Evaluates lstm_revin (WAPE 0.1252) and itransformer_totdem (WAPE 0.1132),
bare vs anchored: per-target WAPE, R^2, teacher-forced reb20/red10 response,
plus scale-factor and nd-ramp diagnostics.

    python demand_simulation/test_infer_anchor.py
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

    # nd-ramp diagnostic: the anchor uses nd(t-1) for the step-t identity, so the
    # 5-min net-demand ramp is the irreducible aggregate error it injects.
    nd_te = base.loc[ti, "net_demand"].to_numpy()
    ramp = np.abs(np.diff(nd_te))
    rel = ramp / np.abs(nd_te[1:]).clip(min=1.0)
    print(f"nd 5-min ramp MW:  p50={np.percentile(ramp, 50):.1f}  p95={np.percentile(ramp, 95):.1f}  "
          f"p99={np.percentile(ramp, 99):.1f}  max={ramp.max():.1f}")
    print(f"nd ramp % of |nd|: p50={100*np.percentile(rel, 50):.2f}%  p95={100*np.percentile(rel, 95):.2f}%  "
          f"p99={100*np.percentile(rel, 99):.2f}%")

    Xb = windows(base.loc[full_idx], fc, xs, lb)
    Xs = windows(scen.loc[full_idx], fc, xs, lb)
    true_te = ys.inverse_transform(data["Yte"])

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, weights in MODELS:
        t0 = time.time()
        bare = M.make_neural(name, n_features=len(fc), n_targets=6)
        bare.load_state_dict(torch.load(weights, map_location="cpu"))
        bare = bare.to(device).eval()
        anch = M.DemandAnchoredHead(bare, xs.mean_[nd_idx], xs.scale_[nd_idx],
                                    ys.mean_, ys.scale_, nd_feat_idx=nd_idx).to(device).eval()

        for tag, model in ((f"{name}", bare), (f"{name}+anchor", anch)):
            pred_te = ys.inverse_transform(ev.predict_neural(model, data["Xte"], device, batch=256))
            met = ev.compute_metrics(true_te, pred_te, sc.TARGETS)   # unclamped, matches recorded baselines
            r = demand_response(model, Xb, Xs, ys, mask, device)
            rows.append((tag, met, r))
            avg = met["average"]
            print(f"  {tag:22s} R2={avg['R2']:.4f} WAPE={avg['WAPE']:.4f} "
                  f"net_demand={r['net_demand']:+.1f}%  ({time.time()-t0:.0f}s)", flush=True)

        # scale-factor diagnostic: how far the rescale moves the bare prediction
        raw_mw = ys.inverse_transform(ev.predict_neural(bare, data["Xte"], device, batch=256))
        pos = np.log1p(np.exp(-np.abs(raw_mw))) + np.maximum(raw_mw, 0.0)          # stable softplus
        nd_last = data["Xte"][:, -1, nd_idx] * xs.scale_[nd_idx] + xs.mean_[nd_idx]
        need = np.maximum(nd_last + pos[:, M.TARGET_CHG_IDX], 1.0)
        gsum = pos[:, M.TARGET_GEN_IDX].sum(1) + 1.0
        s = need / gsum
        print(f"  {name} anchor scale: p1={np.percentile(s,1):.3f} p50={np.percentile(s,50):.3f} "
              f"p99={np.percentile(s,99):.3f} min={s.min():.3f} max={s.max():.3f}", flush=True)

    hdr = ["model", "R2", "WAPE", "hydro", "coal", "gas_steam", "gas_ocgt", "batt_chg", "batt_dis",
           "nd_resp", "coal_resp", "hydro_resp", "ocgt_resp", "batt_dis_resp", "batt_chg_resp"]
    lines = ["# Inference-only demand anchoring on converged blind checkpoints\n",
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
            f"{r['gas_ocgt']:+.1f}% | {r['battery_discharging']:+.1f}% | {r['battery_charging']:+.1f}% |")
    (OUT / "infer_anchor_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "infer_anchor_compare.md")


if __name__ == "__main__":
    main()
