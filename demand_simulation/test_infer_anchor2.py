"""Inference-only anchoring, v2: which channels should carry the correction?

test_infer_anchor.py showed inference-only anchoring keeps the converged
backbone's accuracy except for ONE channel: gas_steam WAPE 0.092 -> 0.262
(itransformer), which is ~100% of the average-WAPE regression (0.113 -> 0.142).
gas_steam is the tiny single-unit channel (Newport, 0-510 MW, usually off) and
its on-periods coincide with evening ramps -- exactly when the anchor's
correction factor deviates from 1 -- so the proportional rescale dumps error on
it. Here the rescale set is configurable (DemandAnchoredHead rescale_idx):
excluded units become softplus passthroughs whose MW is subtracted from `need`,
and the identity sum(SIGN*pred) == net_demand still holds exactly.

Variants (both converged bases, no training):
  softplus     no anchoring at all -- isolates the softplus effect on WAPE
  all5         rescale {hydro, coal, gas_steam, gas_ocgt, batt_dis}  (v1)
  nosteam      rescale {hydro, coal, gas_ocgt, batt_dis}, gas_steam passthrough
  flex3        rescale {hydro, gas_ocgt, batt_dis}, coal + gas_steam passthrough
               (most merit-order-like: baseload untouched, flexible units follow
               demand; risk: at high-solar midday need can hit the eps clamp)

    python demand_simulation/test_infer_anchor2.py
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
# TARGETS order: hydro=0 coal=1 steam=2 ocgt=3 chg=4 dis=5
CONFIGS = [("all5", [0, 1, 2, 3, 5]), ("nosteam", [0, 1, 3, 5]), ("flex3", [0, 3, 5])]
FREE = (11, 14)
REBOUND, REDUCTION = 20.0, 10.0
OUT = HERE / "sweep_eqnd"


class SoftplusOnly(torch.nn.Module):
    """Diagnostic: the head's softplus non-negativity step without any rescale."""

    def __init__(self, base, y_mean, y_scale):
        super().__init__()
        self.base = base
        self.register_buffer("_ym", torch.tensor(np.asarray(y_mean), dtype=torch.float32))
        self.register_buffer("_ys", torch.tensor(np.asarray(y_scale), dtype=torch.float32))

    def forward(self, x):
        pos = torch.nn.functional.softplus(self.base(x) * self._ys + self._ym)
        return (pos - self._ym) / self._ys


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
    nd_last = data["Xte"][:, -1, nd_idx] * xs.scale_[nd_idx] + xs.mean_[nd_idx]

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, weights in MODELS:
        bare = M.make_neural(name, n_features=len(fc), n_targets=6)
        bare.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        bare = bare.to(device).eval()

        # clamp-bind diagnostic inputs: softplus MW of the bare preds on test
        raw_mw = ys.inverse_transform(ev.predict_neural(bare, data["Xte"], device, batch=256))
        pos = np.log1p(np.exp(-np.abs(raw_mw))) + np.maximum(raw_mw, 0.0)

        variants = [(f"{name}", bare),
                    (f"{name}+softplus", SoftplusOnly(bare, ys.mean_, ys.scale_).to(device).eval())]
        for cfg, ridx in CONFIGS:
            head = M.DemandAnchoredHead(bare, xs.mean_[nd_idx], xs.scale_[nd_idx],
                                        ys.mean_, ys.scale_, nd_feat_idx=nd_idx,
                                        rescale_idx=ridx).to(device).eval()
            variants.append((f"{name}+{cfg}", head))
            pass_i = [i for i in M.TARGET_GEN_IDX if i not in ridx]
            need = nd_last + pos[:, M.TARGET_CHG_IDX] - pos[:, pass_i].sum(1)
            print(f"  {name}+{cfg:8s} need pre-clamp: min={need.min():.0f} MW, "
                  f"bind(<1 MW)={100 * (need < 1.0).mean():.2f}%", flush=True)

        for tag, model in variants:
            t0 = time.time()
            pred_te = ys.inverse_transform(ev.predict_neural(model, data["Xte"], device, batch=256))
            met = ev.compute_metrics(true_te, pred_te, sc.TARGETS)
            r = demand_response(model, Xb, Xs, ys, mask, device)
            rows.append((tag, met, r))
            avg = met["average"]
            print(f"  {tag:24s} R2={avg['R2']:.4f} WAPE={avg['WAPE']:.4f} "
                  f"net_demand={r['net_demand']:+.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    hdr = ["model", "R2", "WAPE", "hydro", "coal", "gas_steam", "gas_ocgt", "batt_chg", "batt_dis",
           "nd_resp", "coal_resp", "hydro_resp", "ocgt_resp", "batt_dis_resp"]
    lines = ["# Inference-only anchoring: rescale-set variants\n",
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
    (OUT / "infer_anchor2_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "infer_anchor2_compare.md")


if __name__ == "__main__":
    main()
