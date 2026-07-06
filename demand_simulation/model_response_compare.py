"""Where RevIN actually ruins it: LSTM (no RevIN) vs iTransformer (RevIN) response
to the SAME demand shift, through the same closed-loop ar_free_rollout.

Both models share the net_dispatch_totdem pipeline + scaler. We feed each the same
shifted scenario (demand-side net_demand) and compare the response-region mean
dispatch, base vs scenario. The LSTM responds; the RevIN iTransformer barely moves
/ collapses -- that's the failure, and it's in the rollout, not the one-step input.

    python demand_simulation/model_response_compare.py --rebound 20 --reduction 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402
import sim_common as sc         # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

FREE = (11, 14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebound", type=float, default=20.0)
    ap.add_argument("--reduction", type=float, default=10.0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pipeline.build_table("5min")
    inter = pd.read_parquet(pipeline.DATA_DIR / "vic_interconnector_last365.parquet")
    ni = inter.set_index("interval").sort_index()["net_import_mw"].reindex(df.index).interpolate(limit_direction="both")
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]

    val = df[(df.index > pipeline.TRAIN_END) & (df.index <= pipeline.VAL_END)]
    test = df[df.index > pipeline.VAL_END]
    ti = test.index
    full_idx = pd.concat([val.tail(lb), test]).index
    free_test = np.asarray((ti.hour >= FREE[0]) & (ti.hour < FREE[1]))
    mask = sc.response_mask(ti, 1, FREE)

    def nd(f): return f["demand_mw"] - f["wind"] - f["solar_utility"] - ni.reindex(f.index)
    base = df.copy(); base["net_demand"] = nd(base)
    scen = FixedPercentageShift(args.rebound, args.reduction, free_hours=FREE).transform(df)
    scen["net_demand"] = nd(scen)
    fs_b = xs.transform(base.loc[full_idx, fc].values).astype(np.float32)
    fs_s = xs.transform(scen.loc[full_idx, fc].values).astype(np.float32)

    import models as M
    it = M.make_neural("itransformer", len(fc), 6).to(device)
    it.load_state_dict(torch.load(ROOT / "ml" / "itransformer_totdem" / "itransformer_totdem.pt",
                                  map_location=device))
    models = {
        "LSTM (no RevIN)": sc.load_lstm("lstm_5min_mse", len(fc), device),
        "iTransformer (RevIN)": it.eval(),
    }

    print(f"shift reb{args.rebound:g}% red{args.reduction:g}%  (response region mean, base -> scen)\n")
    hdr = f"{'model':22s} {'net_demand %':>13s} {'coal %':>9s} {'hydro %':>9s} {'batt_dis %':>11s}"
    print(hdr); print("-" * len(hdr))
    for name, mdl in models.items():
        print(f"  rolling out {name} ...", file=sys.stderr)
        pb, ps = sc.ar_free_rollout(mdl, fs_b, fs_s, lb, free_test, xs, ys, device, nd_mode="scenario")
        b, s = pb[mask], ps[mask]
        def pct(i):
            mb = b[:, i].mean(); return 100 * (s[:, i].mean() - mb) / mb if mb else float("nan")
        ndp = 100 * (sc.net_demand(s).mean() - sc.net_demand(b).mean()) / sc.net_demand(b).mean()
        print(f"{name:22s} {ndp:>12.1f}% {pct(1):>8.1f}% {pct(0):>8.1f}% {pct(5):>10.1f}%")


if __name__ == "__main__":
    main()
