"""Deliverable 1+2 — average-day dispatch stacks.

  monthly_<year>.png   12 panels, one per calendar month: the average day of that
                       month, stacked by fuel. Shows the seasonal shape.
  weekly_<year>.png    4 panels, one per week of that year's highest-demand FULL
                       month: the average day of each week.

Both are averages over days at each 5-minute slot, so the peak is the TYPICAL
peak, not a single extreme day.

    python3 dispatch_study/profiles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (load, mean_day, draw_day_stack, FUELS, RENEW, OUT,   # noqa: E402
                    INK, MUTED)

COLS = FUELS + RENEW + ["battery_charging", "demand_mw", "net_demand"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly(t, year):
    sub = t[t.index.year == year]
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharey=True)
    ymax = 0
    profs = {}
    for m in range(1, 13):
        d = sub[sub.index.month == m]
        if len(d) < 288:
            continue
        profs[m] = mean_day(d, COLS)
        ymax = max(ymax, float(profs[m][FUELS + RENEW].sum(1).max()))
    ylim = (-350, ymax * 1.08)
    for i, m in enumerate(range(1, 13)):
        ax = axes[i // 4][i % 4]
        if m not in profs:
            ax.text(0.5, 0.5, f"{MONTHS[m-1]} {year}\nno data", ha="center",
                    va="center", fontsize=9, color=MUTED, transform=ax.transAxes)
            ax.set_axis_off(); continue
        nd = len(sub[sub.index.month == m]) / 288
        draw_day_stack(ax, profs[m], f"{MONTHS[m-1]}  ({nd:.0f} days)",
                       legend=(i == 0), label_bands=False, ylim=ylim)
        if i % 4 == 0:
            ax.set_ylabel("MW", fontsize=8, color=MUTED)
    fig.suptitle(f"Victoria {year} — average day by month, dispatch stacked by fuel "
                 f"(shaded band = evening peak 17:00–21:00)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.005, "Mean MW at each 5-minute slot across all days of the month. "
             "Dashed = total demand, solid = net demand (demand − wind − solar − curtailment). "
             "Battery charging drawn below zero as a load.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.955])
    fp = OUT / f"monthly_{year}.png"
    fig.savefig(fp, dpi=140, facecolor="white"); plt.close(fig)
    print(f"wrote {fp}")


def weekly(t, year):
    sub = t[t.index.year == year]
    full = [m for m in range(1, 13) if len(sub[sub.index.month == m]) >= 28 * 288]
    if not full:
        print(f"  [weekly] no full month in {year}"); return
    m = max(full, key=lambda mm: sub[sub.index.month == mm]["demand_mw"].mean())
    d = sub[sub.index.month == m]
    print(f"  {year}: highest-demand full month = {MONTHS[m-1]} "
          f"({d['demand_mw'].mean():.0f} MW mean demand)")
    dom = d.index.day
    weeks = [(1, 7), (8, 14), (15, 21), (22, 31)]
    profs = [mean_day(d[(dom >= a) & (dom <= b)], COLS) for a, b in weeks]
    ymax = max(float(p[FUELS + RENEW].sum(1).max()) for p in profs)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for i, ((a, b), p) in enumerate(zip(weeks, profs)):
        n = len(d[(dom >= a) & (dom <= b)]) / 288
        draw_day_stack(axes[i], p, f"{MONTHS[m-1]} {a}–{b}  ({n:.0f} days)",
                       legend=(i == 0), label_bands=(i == 0), ylim=(-350, ymax * 1.08))
        if i == 0:
            axes[i].set_ylabel("MW", fontsize=8, color=MUTED)
    fig.suptitle(f"Victoria {year} — average day by week of {MONTHS[m-1]} "
                 f"(the year's highest-demand month)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fp = OUT / f"weekly_{year}.png"
    fig.savefig(fp, dpi=140, facecolor="white"); plt.close(fig)
    print(f"wrote {fp}")


def main():
    t = load()
    for year in (2025, 2026):
        monthly(t, year)
        weekly(t, year)


if __name__ == "__main__":
    main()
