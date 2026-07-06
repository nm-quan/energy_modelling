"""Closed-loop validation of inference-only anchoring (the production path).

Teacher-forced results (test_infer_anchor3.py) gave itransformer+nosteam_relu
WAPE 0.1075 with the full structural +179.2% response. But the bundled sweep
(sweep_eqnd.py) runs CLOSED LOOP: ar_free_rollout with nd_mode="scenario",
demand-side net_demand, dispatch history fed back from the model's own
predictions inside the free window -- where the bare iTransformer collapses.
The anchor re-pins the aggregate to the scenario's net_demand every step, so it
should remove that failure mode. This verifies it, following the sweep protocol
exactly (reb20/red10, demand-side nd on both frames).

Per model: closed-loop accuracy of the BASE rollout against actuals over the
response region (does it stay sane in closed loop?), aggregate tracking error
|sum(SIGN*pred) - nd_input|, and the closed-loop scenario response.

    python demand_simulation/test_infer_anchor_cl.py
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

FREE = (11, 14)
REBOUND, REDUCTION = 20.0, 10.0
NOSTEAM = [0, 1, 3, 5]
OUT = HERE / "sweep_eqnd"


def load_bare(name, weights, n_feats, device):
    m = M.make_neural(name, n_features=n_feats, n_targets=6)
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return m.to(device).eval()


def main():
    # CPU: the rollout is a sequential batch-2 loop (no batching to win on GPU), and
    # the MPS unified-memory watermark OOMs on this 8 GB machine under system load.
    device = "cpu"
    print(f"device: {device}")
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nd_idx = fc.index("net_demand")
    for k in ("Xtr", "Ytr", "Xva", "Yva", "Xte", "Yte"):   # free ~2 GB of windows
        data[k] = None

    df = pipeline.build_table("5min")
    inter = pd.read_parquet(pipeline.DATA_DIR / "vic_interconnector_last365.parquet")
    ni = inter.set_index("interval").sort_index()["net_import_mw"].reindex(df.index).interpolate(limit_direction="both")
    val_df = df[(df.index > pipeline.TRAIN_END) & (df.index <= pipeline.VAL_END)]
    test_df = df[df.index > pipeline.VAL_END]; ti = test_df.index
    full_idx = pd.concat([val_df.tail(lb), test_df]).index
    free_test = np.asarray((ti.hour >= FREE[0]) & (ti.hour < FREE[1]))
    mask = sc.response_mask(ti, 1, FREE)

    def nd(f): return f["demand_mw"] - f["wind"] - f["solar_utility"] - ni.reindex(f.index)
    base = df.copy(); base["net_demand"] = nd(base)
    scen = FixedPercentageShift(REBOUND, REDUCTION, free_hours=FREE).transform(df)
    scen["net_demand"] = nd(scen)
    fs_base = xs.transform(base.loc[full_idx, fc].values).astype(np.float32)
    fs_scen = xs.transform(scen.loc[full_idx, fc].values).astype(np.float32)
    nd_in_b = base.loc[ti, "net_demand"].to_numpy()[mask]
    nd_in_s = scen.loc[ti, "net_demand"].to_numpy()[mask]
    print(f"free-window demand-side nd input: {100 * (nd_in_s.mean() / nd_in_b.mean() - 1):+.1f}%")

    actual = test_df[sc.TARGETS].to_numpy()[mask]

    def anchored(bare):
        return M.DemandAnchoredHead(bare, xs.mean_[nd_idx], xs.scale_[nd_idx],
                                    ys.mean_, ys.scale_, nd_feat_idx=nd_idx,
                                    rescale_idx=NOSTEAM, pos_fn="relu").to(device).eval()

    lstm_prod = load_bare("lstm", ROOT / "ml" / "lstm_5min_mse" / "lstm_5min_mse.pt", len(fc), device)
    lstm_revin = load_bare("lstm_revin", ROOT / "ml" / "lstm_revin_5min" / "lstm_revin_5min.pt", len(fc), device)
    itr = load_bare("itransformer", ROOT / "ml" / "itransformer_totdem" / "itransformer_totdem.pt", len(fc), device)
    models = [("lstm_5min_mse (prod)", lstm_prod),
              ("itransformer (bare)", itr),
              ("itransformer+anchor", anchored(itr)),
              ("lstm_revin+anchor", anchored(lstm_revin))]

    rows = []
    for tag, model in models:
        t0 = time.time()
        pb, ps = sc.ar_free_rollout(model, fs_base, fs_scen, lb, free_test,
                                    xs, ys, device, nd_mode="scenario")
        b, s = pb[mask], ps[mask]
        # nanmean: gas_steam is all-zero in the midday response region, so its
        # per-target WAPE/R2 are undefined and would poison a plain mean
        per = ev.compute_metrics(actual, b, sc.TARGETS)["per_target"]
        met = {"WAPE": float(np.nanmean([per[t]["WAPE"] for t in sc.TARGETS])),
               "R2": float(np.nanmean([per[t]["R2"] for t in sc.TARGETS]))}
        track = np.abs(sc.net_demand(b) - nd_in_b)
        r = {t: 100 * (s[:, i].mean() - b[:, i].mean()) / b[:, i].mean() for i, t in enumerate(sc.TARGETS)}
        ndr = 100 * (sc.net_demand(s).mean() - sc.net_demand(b).mean()) / sc.net_demand(b).mean()
        rows.append((tag, met, track, ndr, r))
        print(f"  {tag:22s} cl_WAPE={met['WAPE']:.4f} cl_R2={met['R2']:.4f} "
              f"|track| p50={np.percentile(track, 50):.0f} p95={np.percentile(track, 95):.0f} MW "
              f"nd_resp={ndr:+.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    hdr = ["model", "cl_WAPE", "cl_R2", "track p50 MW", "track p95 MW", "nd_resp",
           "coal_resp", "hydro_resp", "ocgt_resp", "batt_dis_resp"]
    lines = ["# Closed-loop (ar_free_rollout nd_mode=scenario) anchored vs bare\n",
             f"reb{REBOUND:.0f}/red{REDUCTION:.0f}, demand-side net_demand, free window "
             f"{FREE[0]}:00-{FREE[1]}:00. cl_WAPE/cl_R2 = base-rollout accuracy vs actuals over the "
             "response region; track = |sum(SIGN*pred) - nd input| there.\n",
             "| " + " | ".join(hdr) + " |",
             "| " + " | ".join("---" for _ in hdr) + " |"]
    for tag, met, track, ndr, r in rows:
        lines.append(f"| {tag} | {met['WAPE']:.4f} | {met['R2']:.4f} | "
                     f"{np.percentile(track, 50):.0f} | {np.percentile(track, 95):.0f} | {ndr:+.1f}% | "
                     f"{r['coal_brown']:+.1f}% | {r['hydro']:+.1f}% | {r['gas_ocgt']:+.1f}% | "
                     f"{r['battery_discharging']:+.1f}% |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "infer_anchor_cl_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "infer_anchor_cl_compare.md")


if __name__ == "__main__":
    main()
