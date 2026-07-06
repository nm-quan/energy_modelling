"""Concrete RevIN trace: exactly what RevIN does to one real lookback window
(base vs shifted) and across training windows.

RevIN (lib/models.py RevIN / _revin): for each window it subtracts that window's
per-channel mean and divides by its per-channel std (mean/std DETACHED), runs the
backbone, then RE-scales each predicted target by the same window mean/std of that
target's own channel. Because per-window normalization is affine-invariant, the global
x_scaler cancels -> RevIN's view depends only on the raw window.

We print, for a window whose target is in the free interval (11:00-14:00):
  INPUT side  : raw demand, the window mean/std RevIN computes, and the normalized
                demand the backbone actually sees -- base vs shifted.
  OUTPUT side : the window mean/std of the dispatch (target) channels that RevIN uses
                to set the output level -- base vs shifted (these set the MW level).
  Actual model outputs (MW) base vs shifted for LSTM+RevIN and LSTM no-RevIN.
TRAINING: the spread of per-window demand level across all training windows BEFORE vs
AFTER RevIN -- showing the absolute level is deleted, so it can't be learned.

    python demand_simulation/debug_revin_trace.py --target "2026-04-16 14:00"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import pipeline                 # noqa: E402
import sim_common as sc         # noqa: E402
import models as M              # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

OUT = Path(__file__).resolve().parent / "sweep_eqnd" / "figure"
FREE = (11, 14)
DISP_FEATS = ["hydro", "gas_steam", "gas_ocgt", "coal_brown",
              "battery_charging", "battery_discharging"]   # feature order 0..5


def revin_norm(v):
    return (v - v.mean()) / (v.std() + 1e-5)


def load(arch, w, nf, dev):
    m = M.make_neural(arch, nf, 6); m.load_state_dict(torch.load(w, map_location=dev))
    return m.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="2026-04-16 14:00")
    args = ap.parse_args()
    dev = "cpu"

    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    lb, fc, xs, ys = data["lookback_steps"], data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nf = len(fc)

    df = pipeline.build_table("5min")
    inter = pd.read_parquet(pipeline.DATA_DIR / "vic_interconnector_last365.parquet")
    ni = inter.set_index("interval").sort_index()["net_import_mw"].reindex(df.index).interpolate(limit_direction="both")
    def nd(f): return f["demand_mw"] - f["wind"] - f["solar_utility"] - ni.reindex(f.index)
    base = df.copy(); base["net_demand"] = nd(base)
    scen = FixedPercentageShift(20, 10, free_hours=FREE).transform(df); scen["net_demand"] = nd(scen)

    val = df[(df.index > pipeline.TRAIN_END) & (df.index <= pipeline.VAL_END)]
    test = df[df.index > pipeline.VAL_END]
    full_idx = pd.concat([val.tail(lb), test]).index
    tgt = pd.Timestamp(args.target, tz=df.index.tz)
    j = test.index.get_indexer([tgt])[0]
    win_idx = full_idx[j:j + lb]                       # the 288 raw rows of this window
    t_local = pd.DatetimeIndex(win_idx).tz_localize(None)

    dem_b = base.loc[win_idx, "demand_mw"].to_numpy()
    dem_s = scen.loc[win_idx, "demand_mw"].to_numpy()
    free_in_win = (t_local.hour >= FREE[0]) & (t_local.hour < FREE[1])

    print(f"=== window predicting {tgt}  (288 steps, {t_local[0]} -> {t_local[-1]}) ===")
    print(f"free-interval steps in window: {free_in_win.sum()} (11:00-14:00)\n")

    print("--- INPUT side: what RevIN feeds the backbone for the DEMAND channel ---")
    print(f"raw demand free-window mean:     base {dem_b[free_in_win].mean():8.1f} MW   "
          f"shifted {dem_s[free_in_win].mean():8.1f} MW   (+{100*(dem_s[free_in_win].mean()/dem_b[free_in_win].mean()-1):.0f}%)")
    print(f"RevIN window mean (subtracted):  base {dem_b.mean():8.1f} MW   shifted {dem_s.mean():8.1f} MW")
    print(f"RevIN window std  (divided by):  base {dem_b.std():8.1f}      shifted {dem_s.std():8.1f}")
    nb, nsh = revin_norm(dem_b), revin_norm(dem_s)
    print(f"NORMALIZED demand the backbone sees, free-window mean: "
          f"base {nb[free_in_win].mean():+.3f}   shifted {nsh[free_in_win].mean():+.3f}   "
          f"(bump partly survives: {100*(nsh[free_in_win].mean()-nb[free_in_win].mean()):.0f}% of a std)\n")

    print("--- OUTPUT side: window mean/std of the TARGET channels RevIN uses to set MW level ---")
    print("(dispatch history is NOT shifted, so these are identical base vs shifted -> output level is pinned)")
    for c in ["coal_brown", "hydro", "battery_discharging"]:
        b = base.loc[win_idx, c].to_numpy(); s = scen.loc[win_idx, c].to_numpy()
        print(f"  {c:20s} mean base {b.mean():8.2f} / shifted {s.mean():8.2f}   "
              f"std base {b.std():7.2f} / shifted {s.std():7.2f}")

    # actual model outputs on the scaled window
    Xb = xs.transform(base.loc[win_idx, fc].values).astype(np.float32)[None]
    Xs = xs.transform(scen.loc[win_idx, fc].values).astype(np.float32)[None]
    print("\n--- ACTUAL model output for this window (MW), net_demand = signed sum ---")
    for name, arch, w in [("LSTM no-RevIN", "lstm", "lstm_5min_mse/lstm_5min_mse.pt"),
                          ("LSTM +RevIN", "lstm_revin", "lstm_revin_5min/lstm_revin_5min.pt")]:
        m = load(arch, Path(".").resolve()/"ml"/w, nf, dev)
        ob = sc.net_demand(sc.predict(m, Xb, ys, dev))[0]
        os_ = sc.net_demand(sc.predict(m, Xs, ys, dev))[0]
        print(f"  {name:16s} net_demand  base {ob:8.1f}  shifted {os_:8.1f}   "
              f"({'+' if os_>=ob else ''}{100*(os_-ob)/ob:.1f}%)")

    # ---- TRAINING: per-window demand level deleted by RevIN ----
    Xtr = data["Xtr"]
    di = fc.index("demand_mw")
    # raw per-window demand mean across training windows
    raw_win_mean = Xtr[:, :, di] * xs.scale_[di] + xs.mean_[di]      # (N,288) raw MW
    win_means = raw_win_mean.mean(axis=1)                            # (N,) per-window mean MW
    post_revin = (Xtr[:, :, di] - Xtr[:, :, di].mean(axis=1, keepdims=True)).mean(axis=1)
    print("\n--- TRAINING: per-window demand level, before vs after RevIN ---")
    print(f"raw per-window demand mean across {len(win_means)} train windows: "
          f"min {win_means.min():.0f}  median {np.median(win_means):.0f}  max {win_means.max():.0f} MW  "
          f"(spread {win_means.max()-win_means.min():.0f} MW)")
    print(f"after RevIN (per-window mean subtracted): every window -> {post_revin.mean():.2e} "
          f"(std across windows {post_revin.std():.2e})  => absolute level DELETED, identical for every window")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax = axes[0][0]
    ax.plot(t_local, dem_b, color="#1f77b4", label="base"); ax.plot(t_local, dem_s, color="#d62728", label="shifted")
    ax.axvspan(t_local[free_in_win][0], t_local[free_in_win][-1], color="gold", alpha=0.2)
    ax.set_title("1. RAW demand in the window (MW)"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[0][1]
    ax.plot(t_local, nb, color="#1f77b4", label="base"); ax.plot(t_local, nsh, color="#d62728", label="shifted")
    ax.axvspan(t_local[free_in_win][0], t_local[free_in_win][-1], color="gold", alpha=0.2)
    ax.set_title("2. After RevIN: normalized demand the backbone sees\n(window mean removed -> only shape, level gone)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1][0]
    ax.hist(win_means, bins=60, color="#1f77b4", alpha=0.8)
    ax.set_title("3. TRAINING: raw per-window demand mean (MW)\nwide spread = real level info"); ax.grid(alpha=0.3)
    ax.set_xlabel("per-window mean demand (MW)")
    ax = axes[1][1]
    ax.hist(post_revin, bins=60, color="#d62728", alpha=0.8)
    ax.set_title("4. TRAINING after RevIN: per-window demand mean\nALL ~0 -> level deleted, never seen in training")
    ax.set_xlabel("per-window mean demand after RevIN"); ax.grid(alpha=0.3)
    fig.suptitle("RevIN trace: input level removed (in & training), output level pinned to dispatch channels")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "debug_revin_trace.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
