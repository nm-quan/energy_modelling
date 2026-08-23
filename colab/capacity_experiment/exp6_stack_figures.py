"""EXPERIMENT 6 figures -- stacked dispatch over a WHOLE imputed day.

Reads exp6_stack_data.npz (written by exp6_oneday_gap.py) and draws the same all-energy
stack the notebook uses (cell 20), so the panels are directly comparable to exp5's.

Five panels: actual; the day-long optimization; the imputer given that optimization as
an input; the unconstrained imputer; and the unconstrained imputer projected by planB's
hardnet head.

NOTE on the optimization panel: the QP decides only the SIX DISPATCHABLE channels.
wind, solar_utility and both curtailment channels are passed through from linear
interpolation, bit-identical -- so that panel is a hybrid, and its renewable errors equal
interpolation's by construction. Making them genuine decisions needs renewable
AVAILABILITY inside the gap (design.md sec.4.3 W_bar/V_bar), which is exactly what is
masked, so it requires imputing wind/solar first and feeding nd_free to the QP.

Day selection (one figure each -- the ranking of the arms is NOT the same across them):
    --select normal    ordinary days: coal total variation closest to the test median
    --select demand    the highest-demand days
    --select coalramp  the days coal moved most, by total variation sum|d coal| over the
                       day. These are the days a flat-line fill fails hardest, so they
                       are the discriminating cases rather than the biggest ones.

    python colab/capacity_experiment/exp6_stack_figures.py --select all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (CAL, OBS, STEPS_PER_DAY, TARGETS,  # noqa: E402
                      load_table, make_days)

HERE = Path(__file__).resolve().parent
COLORS = {"coal_brown": "saddlebrown", "gas_steam": "#d62728", "gas_ocgt": "#ff7f0e",
          "hydro": "royalblue", "battery_discharging": "#9467bd",
          "battery_charging": "dimgray"}
WIND, SOLAR = "#2e8b40", "#f4c20d"
STACK_POS = ["coal_brown", "gas_steam", "gas_ocgt", "hydro", "battery_discharging"]
DISP = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
        "battery_charging", "battery_discharging"]
NDAY = 6                              # highest-demand test days to draw
ti = {t: i for i, t in enumerate(TARGETS)}

ap = argparse.ArgumentParser()
ap.add_argument("--select", default="all",
                choices=["all", "normal", "demand", "coalramp"])
ap.add_argument("--ndays", type=int, default=NDAY)
args = ap.parse_args()

d = np.load(HERE / "exp6_stack_data.npz")
Y = d["actual"]
ci = TARGETS.index("coal_brown")
tv = np.abs(np.diff(Y[:, :, ci], axis=1)).sum(1)          # total variation, MW/day
mx = np.abs(np.diff(Y[:, :, ci], axis=1)).max(1)

def pick(mode):
    """(indices, filename tag, human description) for one day-selection mode."""
    if mode == "coalramp":
        return np.sort(np.argsort(-tv)[:args.ndays]), "coalramp", \
            "largest coal total variation"
    if mode == "demand":
        return np.sort(d["t20"])[:args.ndays], "demand", "highest demand"
    med = np.median(tv)
    return np.sort(np.argsort(np.abs(tv - med))[:args.ndays]), "normal", \
        "coal total variation nearest the median (ordinary days)"


MODES = ["normal", "demand", "coalramp"] if args.select == "all" else [args.select]
print(f"coal total variation across the 60 test days: "
      f"min {tv.min():.0f}  median {np.median(tv):.0f}  max {tv.max():.0f} MW/day")

# recover the real calendar date of each imputed day, so the x axis reads as clock time
# rather than step index. Mirrors exp6_oneday_gap.py's window construction exactly.
_, _days = make_days(load_table(), TARGETS + OBS + CAL)
_ok = np.array([i for i in range(len(_days) - 2)
                if (_days[i + 1] - _days[i]).days == 1
                and (_days[i + 2] - _days[i + 1]).days == 1])
_mid = _days[_ok + 1]                       # the masked (middle) day of each window
_N = len(_ok)
_te = np.arange(_N - 60, _N)                # exp6: VAL_DAYS=45, TEST_DAYS=60
DATES = _mid[_te]
def flat(a, sel):
    return a[sel].reshape(-1, len(TARGETS))


def draw_stack(ax, A):
    x = np.arange(len(A))
    base = np.zeros(len(A))
    for k in STACK_POS:
        top = base + np.clip(A[:, ti[k]], 0, None)
        ax.fill_between(x, base, top, color=COLORS[k], alpha=0.9, label=k)
        base = top
    top = base + np.clip(A[:, ti["wind"]], 0, None)
    ax.fill_between(x, base, top, color=WIND, alpha=0.55, label="wind")
    base = top
    top = base + np.clip(A[:, ti["curtailment_wind"]], 0, None)
    ax.fill_between(x, base, top, facecolor="none", edgecolor=WIND, hatch="////",
                    lw=0.0, label="wind curt")
    base = top
    top = base + np.clip(A[:, ti["solar_utility"]], 0, None)
    ax.fill_between(x, base, top, color=SOLAR, alpha=0.8, label="solar_utility")
    base = top
    top = base + np.clip(A[:, ti["curtailment_solar"]], 0, None)
    ax.fill_between(x, base, top, facecolor="none", edgecolor=SOLAR, hatch="////",
                    lw=0.0, label="solar curt")
    ax.fill_between(x, 0, -np.clip(A[:, ti["battery_charging"]], 0, None),
                    color=COLORS["battery_charging"], alpha=0.6,
                    label="battery_charging (load)")
    ax.axhline(0, color="k", lw=0.6)
    ax.margins(x=0)
    ax.grid(True, axis="y", alpha=0.3)
    for i in range(1, len(A) // STEPS_PER_DAY):
        ax.axvline(i * STEPS_PER_DAY, color="k", lw=0.6, alpha=0.4)


PANELS = [("actual", "actual"),
          ("(1) optimization -- QP on the 6 dispatchables; wind/solar/curtailment are "
           "INTERPOLATED, not optimized", "opt"),
          ("(2) optimization as input + BiLSTM imputation (all 10 channels)", "opt_in"),
          ("(3) BiLSTM only -- unconstrained", "base"),
          ("(4) BiLSTM + hardnet projection (planB/heads.py)", "hardnet")]
ARR = {"actual": Y, "opt": d["opt"], "opt_in": d["opt_in"],
       "base": d["base"], "hardnet": d["hardnet"], "interp": d["interp"]}
HRS = [0, 6, 12, 18]
summary = {}

for mode in MODES:
    sel, tag, what = pick(mode)
    print(f"\n=== {mode}: {what} ===")
    print(f"  coal total variation: {tv[sel].round(0).astype(int).tolist()} MW/day")
    print(f"  dates: {[str(DATES[i].date()) for i in sel]}")

    A0 = flat(Y, sel)
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(16, 3.2 * len(PANELS)),
                             sharex=True, sharey=True)
    for ax, (nm, key) in zip(axes, PANELS):
        A = flat(ARR[key], sel)
        draw_stack(ax, A)
        err = "" if key == "actual" else f"   mean |err| {np.abs(A - A0).mean():.1f} MW"
        ax.set_title(f"{nm}{err}", loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("MW")

    # clock time on the x axis: 6-hourly ticks per day, real date under each block
    ticks = [j * STEPS_PER_DAY + h * 12 for j in range(len(sel)) for h in HRS]
    labels = [f"{h:02d}" for _ in range(len(sel)) for h in HRS]
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xticks([j * STEPS_PER_DAY for j in range(len(sel) + 1)], minor=True)
        ax.tick_params(axis="x", which="minor", length=0)
    for j, i in enumerate(sel):
        axes[-1].text((j + 0.5) * STEPS_PER_DAY, -0.13,
                      DATES[i].strftime("%a %-d %b %Y"),
                      transform=axes[-1].get_xaxis_transform(), ha="center", va="top",
                      fontsize=9, fontweight="bold")
    axes[-1].set_xlabel(f"hour of day  —  {len(sel)} test days, {what}, "
                        f"whole day imputed", labelpad=26)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"One-day gap, {mode.upper()} days: optimization, "
                 f"and optimization as an input",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(HERE / f"exp6_stack_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote exp6_stack_{tag}.png")

    summary[mode] = {}
    for nm, key in [("(1) optimization", "opt"), ("(2) optim + BiLSTM", "opt_in"),
                    ("(3) BiLSTM only", "base"), ("(4) BiLSTM + hardnet", "hardnet"),
                    ("linear interpolation", "interp")]:
        e = np.abs(ARR[key][sel] - Y[sel])
        summary[mode][nm] = (e.mean(), e[:, :, ci].mean())

print("\n" + "=" * 78)
print("MAE on the drawn days (all channels / coal only), MW")
print("=" * 78)
print(f"  {'arm':24s} " + "".join(f"{m:>17s}" for m in MODES))
for nm in ("(1) optimization", "(2) optim + BiLSTM", "(3) BiLSTM only",
           "(4) BiLSTM + hardnet", "linear interpolation"):
    cells = "".join(f"{summary[m][nm][0]:8.1f} /{summary[m][nm][1]:7.1f}" for m in MODES)
    print(f"  {nm:24s} " + cells)
