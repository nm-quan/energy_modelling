"""Clean one-step (teacher-forced) demand response: LSTM vs iTransformer.

No closed loop -- isolates the pure input->output response to the demand shift from
the separate closed-loop collapse. For both models, one-shot predict the base and
shifted windows (real dispatch history in every window; only demand_mw / price /
demand-side net_demand differ), then compare the response-region mean dispatch.

This answers "does the iTransformer respond to demand at all?" without the rollout
confound. Expectation: the LSTM (global scaler, sees absolute level) responds; the
RevIN iTransformer responds to demand *shape* but its *level* is normalized out at
both ends, so it responds far less -- a property of RevIN, not transformer capacity.

    python demand_simulation/model_response_tf.py --rebound 20 --reduction 10
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
import models as M              # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

FREE = (11, 14)


def windows(frame, fc, xs, lb):
    X = xs.transform(frame[fc].values).astype(np.float32)
    Xw, _ = pipeline.make_windows(X, np.zeros((len(X), 6), np.float32), lb, 1)
    return Xw


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
    mask = sc.response_mask(ti, 1, FREE)

    def nd(f): return f["demand_mw"] - f["wind"] - f["solar_utility"] - ni.reindex(f.index)
    base = df.copy(); base["net_demand"] = nd(base)
    scen = FixedPercentageShift(args.rebound, args.reduction, free_hours=FREE).transform(df)
    scen["net_demand"] = nd(scen)

    Xb = windows(base.loc[full_idx], fc, xs, lb)
    Xs = windows(scen.loc[full_idx], fc, xs, lb)

    it = M.make_neural("itransformer", len(fc), 6).to(device)
    it.load_state_dict(torch.load(ROOT / "ml" / "itransformer_totdem" / "itransformer_totdem.pt",
                                  map_location=device))
    mdls = {"LSTM (no RevIN)": sc.load_lstm("lstm_5min_mse", len(fc), device),
            "iTransformer (RevIN)": it.eval()}

    dem_in = 100 * (scen.loc[ti, "demand_mw"].to_numpy()[mask].mean()
                    / base.loc[ti, "demand_mw"].to_numpy()[mask].mean() - 1)
    print(f"one-step teacher-forced, shift reb{args.rebound:g}% red{args.reduction:g}%")
    print(f"free-window demand_mw input change: +{dem_in:.1f}%\n")
    hdr = f"{'model':22s} {'net_demand':>11s} {'coal':>8s} {'hydro':>9s} {'gas_ocgt':>9s} {'batt_dis':>9s}"
    print(hdr); print("-" * len(hdr))
    for name, mdl in mdls.items():
        pb = sc.predict(mdl, Xb, ys, device)[mask]
        ps = sc.predict(mdl, Xs, ys, device)[mask]
        def pct(i):
            m = pb[:, i].mean(); return 100 * (ps[:, i].mean() - m) / m if m else float("nan")
        ndp = 100 * (sc.net_demand(ps).mean() - sc.net_demand(pb).mean()) / sc.net_demand(pb).mean()
        print(f"{name:22s} {ndp:>10.1f}% {pct(1):>7.1f}% {pct(0):>8.1f}% {pct(3):>8.1f}% {pct(5):>8.1f}%")


if __name__ == "__main__":
    main()
