"""4-day all-energy stacked dispatch chart for the demand-shift scenario.

Three panels (actual / baseline no-shift / scenario shift) over a run of 4
consecutive full test days. Each panel stacks every energy fuel:

    coal_brown, gas_steam, gas_ocgt, hydro, battery_discharging,
    wind (+ wind curtailment, hatched), solar_utility (+ solar curtailment, hatched)

battery_charging is drawn below 0 (load). Renewables + curtailment are ACTUAL
(the LSTM only predicts the 6 dispatchable targets), so those layers are the
same in every panel; only the dispatchable layers respond to the shift. The
daily free window (11:00-14:00) is shaded gold.

Baseline / scenario dispatch come from the same closed-loop ar_free_rollout used
by sweep_eqnd.py (nd_mode="scenario", demand-side net_demand).

    python demand_simulation/stack_shift_4day.py --rebound 20 --reduction 10
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
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402
import sim_common as sc         # noqa: E402
import stack_plots as sp        # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

OUT = HERE / "sweep_eqnd" / "figure"
FREE_HOURS = (11, 14)


def panels_multiday(win_index, actual, base, scen, ren_win, out, tag):
    """3-panel multi-day all-energy stack: actual / baseline / scenario."""
    ti = {t: i for i, t in enumerate(sp.TARGETS)}
    t = pd.DatetimeIndex(win_index).tz_localize(None)
    f, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, sharey=True)
    for ax, (name, a) in zip(axes, [("actual", actual),
                                    ("baseline (no shift)", base),
                                    ("scenario (shift)", scen)]):
        sp._draw_full_stack(ax, t, a, ren_win, ti)
        for day in np.unique(t.normalize()):
            ax.axvspan(day + pd.Timedelta(hours=FREE_HOURS[0]),
                       day + pd.Timedelta(hours=FREE_HOURS[1]),
                       color="gold", alpha=0.15)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_ylabel("MW")
    axes[2].xaxis.set_major_locator(mdates.DayLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[2].xaxis.set_minor_locator(mdates.HourLocator(byhour=(6, 12, 18)))
    axes[2].set_xlabel("date")
    axes[0].legend(loc="upper left", ncol=5, fontsize=7, framealpha=0.9)
    f.suptitle(f"{tag} — all-energy dispatch stack (renewables + curtailment actual) — "
               f"{len(np.unique(t.date))} days {t[0].date()} to {t[-1].date()} "
               f"(free window gold, battery charging negative)")
    f.tight_layout()
    f.savefig(out, dpi=130)
    plt.close(f)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebound", type=float, default=20.0)
    ap.add_argument("--reduction", type=float, default=10.0)
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}; rebound={args.rebound:g}% reduction={args.reduction:g}%")

    df = pipeline.build_table("5min")
    inter = pd.read_parquet(pipeline.DATA_DIR / "vic_interconnector_last365.parquet")
    ni = inter.set_index("interval").sort_index()["net_import_mw"]
    ni = ni.reindex(df.index).interpolate(limit_direction="both")

    data = pipeline.prepare(resolution="5min", lookback=24, horizon=1,
                            input_mode="net_dispatch_totdem", save=False)
    lb = data["lookback_steps"]
    feat_cols, xs, ys = data["feat_cols"], data["x_scaler"], data["y_scaler"]
    model = sc.load_lstm("lstm_5min_mse", len(feat_cols), device)

    val_df = df[(df.index > pipeline.TRAIN_END) & (df.index <= pipeline.VAL_END)]
    test_df = df[df.index > pipeline.VAL_END]
    test_index = test_df.index
    full_idx = pd.concat([val_df.tail(lb), test_df]).index
    free_test = np.asarray((test_index.hour >= FREE_HOURS[0])
                           & (test_index.hour < FREE_HOURS[1]))

    def nd(frame):
        return frame["demand_mw"] - frame["wind"] - frame["solar_utility"] - ni.reindex(frame.index)

    base = df.copy(); base["net_demand"] = nd(base)
    shift = FixedPercentageShift(args.rebound, args.reduction, free_hours=FREE_HOURS)
    scen = shift.transform(df); scen["net_demand"] = nd(scen)

    fs_base = xs.transform(base.loc[full_idx, feat_cols].values).astype(np.float32)
    fs_scen = xs.transform(scen.loc[full_idx, feat_cols].values).astype(np.float32)
    print("rolling out (closed loop)...")
    pred_b, pred_s = sc.ar_free_rollout(model, fs_base, fs_scen, lb, free_test,
                                        xs, ys, device, nd_mode="scenario")
    actual = test_df[sp.TARGETS].to_numpy()                       # (N, 6) actual dispatch

    days, pos = sp.pick_full_days(pd.DatetimeIndex(test_index), n_days=args.days,
                                  seed=args.seed)
    print("days:", [str(d) for d in days])
    ren = sp.load_renewables(pd.DatetimeIndex(test_index))

    OUT.mkdir(parents=True, exist_ok=True)
    sfx = f"_reb{args.rebound:g}_red{args.reduction:g}"
    panels_multiday(test_index[pos], actual[pos], pred_b[pos], pred_s[pos],
                    ren.iloc[pos], OUT / f"stacked_all_energy_{args.days}day{sfx}.png",
                    tag=f"shift reb{args.rebound:g}% red{args.reduction:g}%")


if __name__ == "__main__":
    main()
