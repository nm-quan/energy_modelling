"""Visualize why RevIN damps the demand-shift signal on a real lookback window.

For one 24h (288-step) lookback window from the test set, plot the demand channel
in three spaces, base vs shifted (reb/red):

  1. raw MW                         -> the shift is plainly there (midday bumps)
  2. RevIN per-window (x-mu)/sigma  -> mu,sigma recompute from the window, so the
                                       bump is PARTIALLY cancelled (the "bug")
  3. fixed train StandardScaler     -> constant mu,sigma, bump fully survives

Per-window RevIN cancels a bump completely only if the whole window were lifted
(then mu,sigma move to cancel it exactly). Because only the 11:00-14:00 slice is
lifted inside a 24h window, mu,sigma move only a little, so the bump survives
partially -- but heavily damped vs the fixed scaler. That damping is the signal
loss that makes the RevIN iTransformer weak to a demand shift.

    python demand_simulation/revin_window_viz.py --rebound 20 --reduction 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

OUT = HERE / "sweep_eqnd" / "figure"
FREE_HOURS = (11, 14)
LB = 288   # 24h of 5-min steps (the bundled model's lookback)


def revin(w):
    """Per-window standardization, as in models._revin (population std + 1e-5)."""
    mu = w.mean()
    sd = w.std() + 1e-5
    return (w - mu) / sd, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebound", type=float, default=20.0)
    ap.add_argument("--reduction", type=float, default=10.0)
    ap.add_argument("--target-day", default="2026-04-16",
                    help="window ENDS at this day's 14:00 (covers prior 24h)")
    args = ap.parse_args()

    df = pipeline.build_table("5min")
    d0 = df["demand_mw"]
    shift = FixedPercentageShift(args.rebound, args.reduction, free_hours=FREE_HOURS)
    ds = shift.transform(df)["demand_mw"]

    # window ends at <target-day> 14:00 -> the 288 steps just before it
    end = pd.Timestamp(args.target_day + " 14:00", tz=df.index.tz)
    end_pos = df.index.get_indexer([end])[0]
    sl = slice(end_pos - LB, end_pos)
    idx = df.index[sl].tz_localize(None)
    base = d0.to_numpy()[sl]
    scen = ds.to_numpy()[sl]

    # fixed train scaler stats for demand_mw
    z = np.load(ROOT / "ml" / "lstm_5min_mse" / "scalers.npz", allow_pickle=True)
    fc = list(z["feat_cols"]); j = fc.index("demand_mw")
    mu_tr, sd_tr = float(z["x_mean"][j]), float(z["x_scale"][j])

    rb, mb, sb = revin(base)
    rs, ms, ss = revin(scen)
    gb = (base - mu_tr) / sd_tr
    gs = (scen - mu_tr) / sd_tr

    # quantify the midday peak in each space (mean over the free window in-window)
    fw = (idx.hour >= FREE_HOURS[0]) & (idx.hour < FREE_HOURS[1])
    def peak(a): return a[fw].mean()
    print(f"window {idx[0]} -> {idx[-1]}  ({len(idx)} steps)")
    print(f"RevIN stats: base mu={mb:.0f} sd={sb:.0f} | scen mu={ms:.0f} sd={ss:.0f}")
    print(f"free-window MEAN demand  raw : base {peak(base):.0f}  scen {peak(scen):.0f}  "
          f"(+{100*(peak(scen)-peak(base))/peak(base):.0f}%)")
    print(f"free-window MEAN demand RevIN: base {peak(rb):+.2f}  scen {peak(rs):+.2f}  "
          f"(delta {peak(rs)-peak(rb):+.2f})")
    print(f"free-window MEAN demand fixed: base {peak(gb):+.2f}  scen {peak(gs):+.2f}  "
          f"(delta {peak(gs)-peak(gb):+.2f})")
    d_rev = peak(rs) - peak(rb)
    d_fix = peak(gs) - peak(gb)
    print(f"signal retained by RevIN vs fixed scaler: {100*d_rev/d_fix:.0f}%")

    f, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    panels = [
        ("1. raw demand_mw (MW)", base, scen, "MW"),
        (f"2. RevIN per-window (x-mu)/sigma  [base mu={mb:.0f},sd={sb:.0f} -> "
         f"scen mu={ms:.0f},sd={ss:.0f}]  bump PARTLY cancelled", rb, rs, "z"),
        (f"3. fixed train scaler (x-{mu_tr:.0f})/{sd_tr:.0f}  bump SURVIVES", gb, gs, "z"),
    ]
    for ax, (title, b, s, ylab) in zip(axes, panels):
        ax.plot(idx, b, color="#1f77b4", lw=1.3, label="base (no shift)")
        ax.plot(idx, s, color="#d62728", lw=1.3, label="scenario (shift)")
        for day in np.unique(idx.normalize()):
            ax.axvspan(day + pd.Timedelta(hours=FREE_HOURS[0]),
                       day + pd.Timedelta(hours=FREE_HOURS[1]),
                       color="gold", alpha=0.18)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[2].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b%d %H:%M"))
    axes[2].set_xlabel("lookback window (free window 11:00-14:00 gold)")
    f.suptitle(f"RevIN damps the demand shift on a 24h lookback window "
               f"(reb{args.rebound:g}% red{args.reduction:g}%) — "
               f"retained {100*d_rev/d_fix:.0f}% of the fixed-scaler bump")
    f.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"revin_window_reb{args.rebound:g}_red{args.reduction:g}.png"
    f.savefig(out, dpi=130); plt.close(f)
    print("wrote", out)


if __name__ == "__main__":
    main()
