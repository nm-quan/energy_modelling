"""Demand before/after chart -- no model needed (demand_mw is a direct input,
not a prediction), so this is fast standalone. Reused by scenario_report.py so a
full scenario run produces this chart from the same base/scen frames.

    python demand_simulation/demand_chart.py --rebound 7 --reduction 7
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
import stack_plots as sp        # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

OUT = HERE / "sweep_eqnd" / "figure"
FREE_HOURS = (11, 14)


def plot_demand_before_after(test_index, dem_before, dem_after, free_hours, out, tag):
    """test_index/dem_before/dem_after are full test-set aligned; picks the same
    4-day window stack_shift_4day.py uses (seed=42) so figures line up."""
    days, pos = sp.pick_full_days(pd.DatetimeIndex(test_index), n_days=4, seed=42)
    t = pd.DatetimeIndex(test_index)[pos].tz_localize(None)
    b, a = np.asarray(dem_before)[pos], np.asarray(dem_after)[pos]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, b, label="before (base)", color="tab:blue", lw=1.3)
    ax.plot(t, a, label="after (scenario)", color="tab:red", lw=1.3)
    for day in np.unique(t.normalize()):
        ax.axvspan(day + pd.Timedelta(hours=free_hours[0]), day + pd.Timedelta(hours=free_hours[1]),
                   color="gold", alpha=0.18)
    ax.set_ylabel("demand_mw (MW)")
    ax.set_xlabel("date")
    ax.margins(x=0)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=(6, 12, 18)))
    ax.set_title(f"{tag} — demand before vs after shift — {len(days)} days "
                f"{t[0].date()} to {t[-1].date()} (free window gold)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebound", type=float, required=True)
    ap.add_argument("--reduction", type=float, required=True)
    args = ap.parse_args()

    df = pipeline.build_table("5min")
    val_df = df[(df.index > pipeline.TRAIN_END) & (df.index <= pipeline.VAL_END)]
    test_df = df[df.index > pipeline.VAL_END]
    test_index = test_df.index

    scen_df = FixedPercentageShift(args.rebound, args.reduction, free_hours=FREE_HOURS).transform(df)

    OUT.mkdir(parents=True, exist_ok=True)
    sfx = f"_reb{args.rebound:g}_red{args.reduction:g}"
    plot_demand_before_after(
        test_index, test_df["demand_mw"], scen_df.loc[test_index, "demand_mw"],
        FREE_HOURS, OUT / f"demand_before_after{sfx}.png",
        tag=f"shift reb{args.rebound:g}% red{args.reduction:g}%")


if __name__ == "__main__":
    main()
