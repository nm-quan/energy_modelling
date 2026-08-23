"""EXPERIMENT 5 figures -- stacked dispatch in the 3h gap.

Reads exp5_stack_data.npz (written by exp5_optimization_input.py) and draws the same
all-energy stack the notebook uses (cell 20): dispatchables stacked positive, wind and
solar above them with hatched curtailment, battery charging as a negative band.

Two figures:
  exp5_stack_top20.png  actual / optimization / imputer-with-optimization-input,
                        over the 20 highest-demand test days
  exp5_stack_all.png    the same three plus interpolation and the plain imputer

    python colab/capacity_experiment/exp5_stack_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import GAP, TARGETS  # noqa: E402

HERE = Path(__file__).resolve().parent
COLORS = {"coal_brown": "saddlebrown", "gas_steam": "#d62728", "gas_ocgt": "#ff7f0e",
          "hydro": "royalblue", "battery_discharging": "#9467bd",
          "battery_charging": "dimgray"}
WIND, SOLAR = "#2e8b40", "#f4c20d"
STACK_POS = ["coal_brown", "gas_steam", "gas_ocgt", "hydro", "battery_discharging"]
ti = {t: i for i, t in enumerate(TARGETS)}

d = np.load(HERE / "exp5_stack_data.npz")
t20 = np.sort(d["t20"])


def flat(a):
    """(days, GAP, 10) -> (days*GAP, 10) over the top-20 days, chronological."""
    return a[t20].reshape(-1, len(TARGETS))


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
    for i in range(1, len(A) // GAP):
        ax.axvline(i * GAP, color="k", lw=0.4, alpha=0.3)


def figure(panels, fname, title):
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.4 * n), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    A0 = flat(d["actual"])
    for ax, (nm, key) in zip(axes, panels):
        A = A0 if key == "actual" else flat(d[key])
        draw_stack(ax, A)
        err = "" if key == "actual" else (
            f"   mean |err| {np.abs(flat(d[key]) - A0).mean():.1f} MW")
        ax.set_title(f"{nm}{err}", loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("MW")
    axes[-1].set_xlabel(f"20 highest-demand test days, 11:00-14:00 gap "
                        f"({GAP} x 5 min each), chronological")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(HERE / fname, dpi=130, bbox_inches="tight")
    print(f"wrote {fname}")


figure([("actual", "actual"),
        ("(1) optimization -- constrained dispatch QP, no network", "opt"),
        ("(2) BiLSTM imputer with the optimization as input", "opt_in")],
       "exp5_stack_top20.png",
       "Filling the 3h dispatch gap: optimization, and optimization as an input")

figure([("actual", "actual"),
        ("linear interpolation", "interp"),
        ("(1) optimization -- constrained dispatch QP", "opt"),
        ("BiLSTM imputer (control)", "base"),
        ("(2) BiLSTM imputer + optimization as input", "opt_in")],
       "exp5_stack_all.png",
       "All arms, 20 highest-demand test days")
