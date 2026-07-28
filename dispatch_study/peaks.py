"""Deliverable 3 — who serves the evening peak, and how often.

Writes peaks.md (tables) and peaks.png (yearly average evening peak, stacked).

Four things per fuel:
  running level   mean MW in the peak window vs the midday trough vs all day
  peak share      % of the energy dispatched during 17:00-21:00
  appearance      % of days on which the fuel is active (>10 MW) at any point in
                  the peak, and % of peak STEPS on which it is active
  peak ramp       mean(peak) - mean(trough): how much the fuel is turned UP to
                  serve the evening rise, and its share of the total turn-up.
                  This is the number that matters for a demand-shift study --
                  running level tells you who is on, ramp tells you who responds.

    python3 dispatch_study/peaks.py
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
from common import (load, slot, in_hours, mean_day, draw_day_stack, FUELS,   # noqa: E402
                    RENEW, ALL_DISPATCH, PEAK_HOURS, TROUGH_HOURS, ACTIVE_MW,
                    OUT, LABEL, INK, MUTED, DT)

COLS = FUELS + RENEW + ["battery_charging", "demand_mw", "net_demand"]


def table(t: pd.DataFrame) -> pd.DataFrame:
    pk = in_hours(t.index, *PEAK_HOURS)
    tr = in_hours(t.index, *TROUGH_HOURS)
    day = t.index.normalize()
    rows = []
    peak_e = t.loc[pk, ALL_DISPATCH].clip(lower=0).sum() * DT
    for f in ALL_DISPATCH:
        x = t[f].values
        mp, mt, ma = x[pk].mean(), x[tr].mean(), x.mean()
        # active on a day = at least one peak step above ACTIVE_MW
        act_day = t.loc[pk].groupby(day[pk])[f].max() > ACTIVE_MW
        rows.append({
            "fuel": LABEL[f],
            "mean all-day (MW)": ma,
            "mean trough 11-14 (MW)": mt,
            "mean peak 17-21 (MW)": mp,
            "% of peak energy": 100 * peak_e[f] / peak_e.sum(),
            "% days active in peak": 100 * act_day.mean(),
            "% peak steps active": 100 * (x[pk] > ACTIVE_MW).mean(),
            "peak ramp (MW)": mp - mt,
        })
    d = pd.DataFrame(rows).set_index("fuel")
    # battery charging is a LOAD: reducing it serves the peak, so its contribution
    # to the turn-up is the REDUCTION (trough - peak), i.e. minus its ramp.
    contrib = d["peak ramp (MW)"].copy()
    contrib[LABEL["battery_charging"]] *= -1
    d["% of total turn-up"] = 100 * contrib / contrib.clip(lower=0).sum()
    return d


def md(d: pd.DataFrame, cols, title, fmt) -> list[str]:
    L = [f"### {title}", "",
         "| fuel | " + " | ".join(cols) + " |",
         "| --- |" + " ---: |" * len(cols)]
    for f, r in d.iterrows():
        L.append(f"| {f} | " + " | ".join(fmt[c](r[c]) for c in cols) + " |")
    return L + [""]


def main():
    t = load()
    OUT.mkdir(exist_ok=True)
    f1 = lambda v: f"{v:,.1f}"
    f0 = lambda v: f"{v:,.0f}"
    pc = lambda v: f"{v:.1f}%"

    L = ["# Who serves the Victorian evening peak", "",
         f"Source: `data/preprocessed/hist/5min/net_dispatch_ren/table.parquet`, "
         f"{len(t):,} five-minute intervals, {t.index[0].date()} to {t.index[-1].date()}. "
         "All local (AEST) time.", "",
         f"**Evening peak** = {PEAK_HOURS[0]}:00–{PEAK_HOURS[1]}:00, chosen from the data "
         f"(mean demand exceeds 5,900 MW across exactly these hours and tops out 18:00–19:00). "
         f"**Midday trough** = {TROUGH_HOURS[0]}:00–{TROUGH_HOURS[1]}:00, the daily minimum. "
         f"**Active** = output above {ACTIVE_MW:.0f} MW.", ""]

    d = table(t)
    L += md(d, ["mean all-day (MW)", "mean trough 11-14 (MW)", "mean peak 17-21 (MW)",
                "% of peak energy"],
            "Running level — who is ON during the peak",
            {"mean all-day (MW)": f1, "mean trough 11-14 (MW)": f1,
             "mean peak 17-21 (MW)": f1, "% of peak energy": pc})
    L += md(d, ["% days active in peak", "% peak steps active"],
            "Appearance — how often each fuel shows up in the peak at all",
            {"% days active in peak": pc, "% peak steps active": pc})
    L += md(d, ["peak ramp (MW)", "% of total turn-up"],
            "Peak ramp — who is turned UP to serve the evening rise",
            {"peak ramp (MW)": lambda v: f"{v:+,.1f}", "% of total turn-up": pc})
    L += ["_Battery charging is a load: it contributes to the turn-up by "
          "*reducing*, so its ramp is negative and its turn-up share is the "
          "magnitude of that reduction._", ""]

    # ---- per-year evening peak ----
    L += ["## Yearly evening peak", "",
          "| year | days | mean peak demand (MW) | mean peak net demand (MW) | "
          "peak hour | trough→peak net-demand rise (MW) |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for y, sub in t.groupby(t.index.year):
        pk = in_hours(sub.index, *PEAK_HOURS); tr = in_hours(sub.index, *TROUGH_HOURS)
        prof = sub.groupby(sub.index.hour)["demand_mw"].mean()
        L.append(f"| {y} | {len(sub)//288} | {sub.loc[pk,'demand_mw'].mean():,.0f} | "
                 f"{sub.loc[pk,'net_demand'].mean():,.0f} | {prof.idxmax():02d}:00 | "
                 f"{sub.loc[pk,'net_demand'].mean()-sub.loc[tr,'net_demand'].mean():,.0f} |")
    L += ["", "_2021 excluded (Oct–Dec only); 2026 runs to 5 July._", ""]

    # ---- per-year turn-up composition ----
    L += ["## Peak turn-up composition by year (% of the evening turn-up)", "",
          "| year | " + " | ".join(LABEL[f] for f in ALL_DISPATCH) + " |",
          "| --- |" + " ---: |" * len(ALL_DISPATCH)]
    for y, sub in t.groupby(t.index.year):
        dy = table(sub)
        L.append(f"| {y} | " + " | ".join(
            f"{dy.loc[LABEL[f], '% of total turn-up']:.1f}%" for f in ALL_DISPATCH) + " |")
    L.append("")

    (OUT / "peaks.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT/'peaks.md'}")

    # ---- figure: average day per year ----
    years = sorted(t.index.year.unique())
    fig, axes = plt.subplots(1, len(years), figsize=(3.6 * len(years), 4.4), sharey=True)
    profs = [mean_day(t[t.index.year == y], COLS) for y in years]
    ymax = max(float(p[FUELS + RENEW].sum(1).max()) for p in profs)
    for i, (y, p) in enumerate(zip(years, profs)):
        nd = len(t[t.index.year == y]) // 288
        draw_day_stack(axes[i], p, f"{y}  ({nd} days)", legend=(i == 0),
                       label_bands=(i == 0), ylim=(-350, ymax * 1.08))
        if i == 0:
            axes[i].set_ylabel("MW", fontsize=8, color=MUTED)
    fig.suptitle("Victoria — average day by year (shaded band = evening peak 17:00–21:00)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "peaks.png", dpi=140, facecolor="white"); plt.close(fig)
    print(f"wrote {OUT/'peaks.png'}")


if __name__ == "__main__":
    main()
