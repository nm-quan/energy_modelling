"""Does the neural backbone earn its keep in CLOSED LOOP? (constraint_research.md)

Stage 0 showed 5-min persistence beats every learned model at one-step WAPE
(0.0956) and anchor(persistence) additionally gets the full structural demand
response -- so the only places a neural backbone can justify itself are the
closed-loop free-window rollout (persistence freezes its mix there; the anchor
only rescales the level) and horizon > 1. This runs the sweep-protocol rollout
(ar_free_rollout, nd_mode="scenario", reb20/red10, demand-side nd) for:

  persistence+anchor     frozen entry mix, level pinned to scenario nd
  itransformer+anchor    evolving learned mix, level pinned to scenario nd
  lstm_5min_mse (prod)   current production model, no anchor

and scores base-rollout accuracy vs actuals over the response region
(nanmean per-target WAPE/R2 -- gas_steam is all-zero midday), aggregate
tracking, and the scenario response.

    python3 constraints/stage1_persistence_cl.py
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
OUT = HERE / "results"


def main():
    device = "cpu"          # rollout is sequential batch-2; MPS watermark OOMs on 8 GB
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nd_idx = fc.index("net_demand")
    for k in ("Xtr", "Ytr", "Xva", "Yva", "Xte", "Yte"):
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
    actual = test_df[sc.TARGETS].to_numpy()[mask]

    def anchored(m):
        return M.DemandAnchoredHead(m, xs.mean_[nd_idx], xs.scale_[nd_idx],
                                    ys.mean_, ys.scale_, nd_feat_idx=nd_idx,
                                    rescale_idx=NOSTEAM, pos_fn="relu").to(device).eval()

    def load(arch, path):
        m = M.make_neural(arch, n_features=len(fc), n_targets=6)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return m.to(device).eval()

    pers = M.PersistenceForecaster(xs.mean_, xs.scale_, ys.mean_, ys.scale_).to(device).eval()
    itr = load("itransformer", ROOT / "ml" / "itransformer_totdem" / "itransformer_totdem.pt")
    prod = load("lstm", ROOT / "ml" / "lstm_5min_mse" / "lstm_5min_mse.pt")
    models = [("persistence+anchor", anchored(pers)),
              ("itransformer+anchor", anchored(itr)),
              ("lstm_5min_mse (prod)", prod)]

    rows = []
    for tag, model in models:
        t0 = time.time()
        pb, ps = sc.ar_free_rollout(model, fs_base, fs_scen, lb, free_test,
                                    xs, ys, device, nd_mode="scenario")
        b, s = pb[mask], ps[mask]
        per = ev.compute_metrics(actual, b, sc.TARGETS)["per_target"]
        wape = float(np.nanmean([per[t]["WAPE"] for t in sc.TARGETS]))
        r2 = float(np.nanmean([per[t]["R2"] for t in sc.TARGETS]))
        track = np.abs(sc.net_demand(b) - nd_in_b)
        ndr = 100 * (sc.net_demand(s).mean() - sc.net_demand(b).mean()) / sc.net_demand(b).mean()
        resp = {t: 100 * (s[:, i].mean() - b[:, i].mean()) / b[:, i].mean() for i, t in enumerate(sc.TARGETS)}
        rows.append((tag, wape, r2, track, ndr, resp,
                     {t: per[t]["WAPE"] for t in sc.TARGETS}))
        print(f"  {tag:22s} cl_WAPE={wape:.4f} cl_R2={r2:.4f} "
              f"|track| p50={np.percentile(track, 50):.0f} MW nd_resp={ndr:+.1f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)

    hdr = ["model", "cl_WAPE", "cl_R2", "hydro", "coal", "gas_ocgt", "batt_chg", "batt_dis",
           "track p50 MW", "nd_resp", "hydro_resp", "batt_dis_resp"]
    lines = ["# Closed-loop: anchored persistence vs neural backbones\n",
             f"ar_free_rollout nd_mode=scenario, reb{REBOUND:.0f}/red{REDUCTION:.0f}, free window "
             f"{FREE[0]}:00-{FREE[1]}:00; response-region metrics (per-target cols = cl WAPE; "
             "gas_steam omitted, all-zero midday).\n",
             "| " + " | ".join(hdr) + " |",
             "| " + " | ".join("---" for _ in hdr) + " |"]
    for tag, wape, r2, track, ndr, resp, pw in rows:
        lines.append(f"| {tag} | {wape:.4f} | {r2:.4f} | "
                     f"{pw['hydro']:.3f} | {pw['coal_brown']:.3f} | {pw['gas_ocgt']:.3f} | "
                     f"{pw['battery_charging']:.3f} | {pw['battery_discharging']:.3f} | "
                     f"{np.percentile(track, 50):.0f} | {ndr:+.1f}% | "
                     f"{resp['hydro']:+.1f}% | {resp['battery_discharging']:+.1f}% |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage1_persistence_cl.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "stage1_persistence_cl.md")


if __name__ == "__main__":
    main()
