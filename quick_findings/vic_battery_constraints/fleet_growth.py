"""When VIC battery capacity was added, 2025-2026.

Daily maximum discharge (grey) and the running maximum (blue). The running
maximum only moves when the fleet does something it could not do before, so the
date it stops moving is the date the build-out finished.

    python3 quick_findings/vic_battery_constraints/fleet_growth.py
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

STEP_MW = 20.0     # a rise smaller than this is jitter, not new capacity
INK, MUTED, GRID, SURF = "#0b0b0b", "#8a8880", "#e6e5e1", "#fcfcfb"
BLUE, GREY = "#2a78d6", "#c9c8c2"

T = pd.read_parquet("data/preprocessed/hist/5min/net_dispatch_ren/table.parquet")
T.index = pd.to_datetime(T.index)
W = T.loc["2025-01-01":]
daily = W.battery_discharging.resample("D").max()
run = daily.cummax()

rises = run.index[run.diff().fillna(0.0) > STEP_MW]
end = rises[-1]                      # last time new capacity showed up
start = daily.index[0]

mpl.rcParams.update({"font.size": 10, "text.color": INK})
fig, ax = plt.subplots(figsize=(10.5, 4.4))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

ax.axvspan(start, end, color=BLUE, alpha=0.07, lw=0, zorder=1)
ax.axvline(end, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=5)

ax.plot(daily.index, daily.values, color=GREY, lw=1.0, zorder=2)
ax.plot(run.index, run.values, color=BLUE, lw=2.6, zorder=4)

top = daily.max() * 1.16
ax.annotate(f"{run.iloc[0]:,.0f} MW", xy=(start, run.iloc[0]),
            xytext=(8, 10), textcoords="offset points", fontsize=10, color=INK)
ax.annotate(f"{run.iloc[-1]:,.0f} MW", xy=(end, run.iloc[-1]),
            xytext=(10, 6), textcoords="offset points", fontsize=10, color=INK)
mid = start + (end - start) / 2
ax.text(mid, top * 0.97, "capacity being added", ha="center", va="top",
        fontsize=10, color=INK)
ax.text(end + pd.Timedelta(days=12), top * 0.97, "flat", ha="left", va="top",
        fontsize=10, color=INK)
ax.text(end, top * 0.80, f"  {end.strftime('%d %b %Y')}", ha="left", va="top",
        fontsize=9.5, color=MUTED)

ax.set_ylabel("MW")
ax.set_ylim(0, top)
ax.set_xlim(start, daily.index[-1])
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.tick_params(labelsize=9, colors=MUTED, length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
fig.tight_layout()
out = "quick_findings/vic_battery_constraints/fleet_growth.png"
fig.savefig(out, dpi=170, facecolor=SURF)
print("wrote", out)
print(f"build-out ends {end.date()}: {run.iloc[0]:,.0f} MW -> {run.iloc[-1]:,.0f} MW")
