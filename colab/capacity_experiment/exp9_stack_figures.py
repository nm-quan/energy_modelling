"""EXPERIMENT 9 figures -- the constrained model's imputed day against the recorded one.

Reads exp8_stack_data.npz (written by exp8_constrained_imputer.py) and draws the
all-energy dispatch stack in the same style as exp5/exp6 and lib/stack_plots.py, so the
panels are directly comparable to the figures already in the repo.

WHAT EACH FIGURE IS FOR

  window  N full 3-DAY windows, recorded vs model, the imputed middle day shaded gold.
          This is the one to read first. Two things are visible in it that no table can
          show: the stack inside the gap sums to the dashed net-demand line (that IS the
          balance constraint), and the stack is CONTINUOUS across both gold edges -- no
          seam -- because the fill is pinned to the observed steps either side.

  span    the whole 60-day test period as daily means. At 5-minute resolution 60 days is
          17,280 points and unreadable; the daily mean shows the entire simulated period
          honestly, and a systematic bias (too much coal, say) shows up as a persistent
          offset rather than hiding inside an average.

  peak    the highest-demand test days -- the stress slice, where a flat-line fill fails
          hardest and where the capacity and ramp bounds actually bite.

ONE SEED. exp8 trains three seeds for the tables but keeps no weights, so the model panels
are SEED 0's predictions, not the 3-seed mean. Stated on every figure.

A NOTE ON THE PALETTE, since it is inherited rather than chosen. These are the repo's fuel
colours (lib/stack_plots.py, exp5, exp6, behaviour/) and they are kept for consistency, but
run through a CVD validator two adjacent pairs fail: coal_brown vs gas_steam (dE 3.4 under
protanopia) and hydro vs battery_discharging (dE 12.8 even with normal vision). Brown and
red are the same colour to a red-blind reader and no brown fixes it. Mitigated here with a
surface gap between bands, direct labels on the large ones and hatching on gas_steam; a
real fix means re-stepping the palette across every figure in the repo, which is a
repo-wide decision rather than this script's to make.

    python colab/capacity_experiment/exp9_stack_figures.py [--select all]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402

HERE = Path(__file__).resolve().parent
SURFACE = "#fcfcfb"

# the repo's fuel colours -- lib/stack_plots.py / exp6_stack_figures.py
COLORS = {"coal_brown": "saddlebrown", "gas_steam": "#d62728", "gas_ocgt": "#ff7f0e",
          "hydro": "royalblue", "battery_discharging": "#9467bd",
          "battery_charging": "dimgray"}
HATCH = {"gas_steam": "///"}            # secondary encoding for the brown/red CVD pair
STACK_POS = ["coal_brown", "gas_steam", "gas_ocgt", "hydro", "battery_discharging"]
CHG = "battery_charging"
LABEL_MIN_MW = 250.0                    # only label a band thick enough to hold text


def stack(ax, x, dat, ch, gap=None, nd=None, title="", label=True, legend=False):
    """One dispatch stack. Positive fuels stacked in merit order, charging below zero.

    The 1pt surface-coloured edge on every band is not decoration: adjacent fuels in this
    palette are hard to separate (see the module docstring), and a gap between fills is the
    cheapest secondary encoding there is.
    """
    idx = {c: i for i, c in enumerate(ch)}
    base = np.zeros_like(x, dtype=float)
    for name in STACK_POS:
        v = dat[:, idx[name]]
        top = base + v
        ax.fill_between(x, base, top, facecolor=COLORS[name], alpha=0.92, linewidth=1.0,
                        edgecolor=SURFACE, hatch=HATCH.get(name), label=name)
        if label and v.mean() > LABEL_MIN_MW:
            j = int(np.argmax(v))
            ax.text(x[j], base[j] + v[j] / 2, name.replace("_", " "), fontsize=6.5,
                    ha="center", va="center", color="white", weight="bold")
        base = top
    ax.fill_between(x, 0, -dat[:, idx[CHG]], facecolor=COLORS[CHG], alpha=0.65,
                    linewidth=1.0, edgecolor=SURFACE, label=CHG + " (load)")
    if nd is not None:
        ax.plot(x, nd, color="0.15", lw=1.4, ls="--", label="net demand", zorder=5)
    if gap is not None:
        ax.axvspan(x[gap[0]], x[gap[1] - 1], color="gold", alpha=0.16, zorder=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(axis="y", color="0.88", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend:
        ax.legend(loc="upper left", fontsize=6.5, ncol=4, framealpha=0.9)


def mae(a, b):
    return float(np.abs(a - b).mean())


def fig_windows(d, pick, arms, out, suptitle):
    """Rows = days, cols = recorded then each model arm. Full 3-day window each."""
    ch = list(d["channels"])
    g0, g1 = d["gap"]
    n, m = len(pick), len(arms) + 1
    fig, axes = plt.subplots(n, m, figsize=(6.2 * m, 2.5 * n), squeeze=False, sharey="row")
    for r, i in enumerate(pick):
        x = np.arange(d["actual"].shape[1]) / 12.0                 # hours from window start
        nd = np.full(len(x), np.nan)
        nd[g0:g1] = d["nd_true"][i]
        stack(axes[r][0], x, d["actual"][i], ch, gap=(g0, g1), nd=nd,
              title=f"recorded   {d['days'][i]}", legend=(r == 0))
        for c, arm in enumerate(arms, start=1):
            e = mae(d[arm][i, g0:g1], d["actual"][i, g0:g1])
            stack(axes[r][c], x, d[arm][i], ch, gap=(g0, g1), nd=nd,
                  title=f"{arm}   {d['days'][i]}   gap MAE {e:.0f} MW")
        for c in range(m):
            axes[r][c].set_xlim(0, x[-1])
            if r == n - 1:
                axes[r][c].set_xlabel("hours (middle day = the 24 h gap, shaded)",
                                      fontsize=8)
        axes[r][0].set_ylabel("MW", fontsize=8)
    fig.suptitle(suptitle, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_span(d, arms, out):
    """The whole test period as daily means -- one row per arm, recorded first."""
    ch = list(d["channels"])
    g0, g1 = d["gap"]
    names = ["actual"] + list(arms)
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.6 * len(names)), sharey=True,
                             squeeze=False)
    x = np.arange(d["actual"].shape[0])
    for r, nm in enumerate(names):
        day = d[nm][:, g0:g1].mean(axis=1)                          # (days, channels)
        ttl = "recorded" if nm == "actual" else nm
        if nm != "actual":
            ttl += f"   whole-period MAE {mae(d[nm][:, g0:g1], d['actual'][:, g0:g1]):.1f} MW"
        stack(axes[r][0], x, day, ch, title=ttl, label=(r == 0), legend=(r == 0))
        axes[r][0].set_xlim(0, x[-1])
        axes[r][0].set_ylabel("MW (daily mean)", fontsize=8)
    step = max(1, len(x) // 12)
    axes[-1][0].set_xticks(x[::step])
    axes[-1][0].set_xticklabels([d["days"][i] for i in x[::step]], rotation=45,
                                ha="right", fontsize=7)
    fig.suptitle(f"Whole simulated period, {d['days'][0]} .. {d['days'][-1]} "
                 f"({len(x)} imputed days, daily means) — seed {int(d['seed'][0])}",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default="all",
                    choices=["all", "window", "span", "peak"])
    ap.add_argument("--ndays", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--arms", default="model,interp",
                    help="comma-separated npz keys to draw beside the recorded panel")
    a = ap.parse_args()

    d = dict(np.load(HERE / f"exp8_stack_data{a.tag}.npz", allow_pickle=True))
    arms = [x for x in a.arms.split(",") if x in d]
    g0, g1 = d["gap"]
    print(f"{d['actual'].shape[0]} test days {d['days'][0]} .. {d['days'][-1]}   "
          f"arms: {arms}   seed {int(d['seed'][0])}")

    res = HERE / "exp8_constrained_results.json"
    if res.exists():
        j = json.loads(res.read_text())
        print(f"  table says +SOC MAE {j['+SOC']['MAE']:.2f} (3 seeds); "
              f"this figure is seed {int(d['seed'][0])}, "
              f"MAE {mae(d['model'][:, g0:g1], d['actual'][:, g0:g1]):.2f}")

    peak = np.argsort(-d["demand"][:, g0:g1].max(axis=1))
    # ordinary days: coal total variation closest to the median (the exp6 convention)
    ci = list(d["channels"]).index("coal_brown")
    tv = np.abs(np.diff(d["actual"][:, g0:g1, ci], axis=1)).sum(1)
    normal = np.argsort(np.abs(tv - np.median(tv)))

    if a.select in ("all", "window"):
        fig_windows(d, normal[:a.ndays], arms, HERE / f"exp9_stack_window{a.tag}.png",
                    f"Ordinary test days — recorded vs imputed (seed {int(d['seed'][0])}). "
                    f"Gold = the 24 h gap the model fills; dashed = net demand.")
    if a.select in ("all", "peak"):
        fig_windows(d, peak[:a.ndays], arms, HERE / f"exp9_stack_peak{a.tag}.png",
                    f"Highest-demand test days — recorded vs imputed "
                    f"(seed {int(d['seed'][0])}).")
    if a.select in ("all", "span"):
        fig_span(d, arms, HERE / f"exp9_stack_span{a.tag}.png")


if __name__ == "__main__":
    main()
