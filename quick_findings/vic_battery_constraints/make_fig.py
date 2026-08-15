"""Four diagnostic panels for the VIC battery constraint derivation."""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1"
SURF = "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "font.size": 9, "legend.frameon": False, "legend.fontsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.7,
})

T = pd.read_parquet("data/preprocessed/hist/5min/net_dispatch_totdem/table.parquet")
T.index = pd.to_datetime(T.index)
DT = 5 / 60.0
W = T.loc["2025-12-01":"2026-07-05"]
chg, dis = W.battery_charging.to_numpy(), W.battery_discharging.to_numpy()
idx = W.index
day = idx.floor("D")
rt = dis.sum() / chg.sum()
eta = np.sqrt(rt)

fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
fig.suptitle("VIC battery — constraints inferred from Dec 2025 – Jul 2026 dispatch",
             fontsize=12, fontweight="bold", x=0.055, ha="left", y=0.985)

# ---- A: hour-of-day charge vs discharge -------------------------------------
ax = axes[0, 0]
h = idx.hour
mc = pd.Series(chg, index=h).groupby(level=0).mean()
md = pd.Series(dis, index=h).groupby(level=0).mean()
ax.bar(mc.index - 0.21, -mc.values, width=0.4, color=BLUE, label="charging", zorder=3)
ax.bar(md.index + 0.21, md.values, width=0.4, color=RED, label="discharging", zorder=3)
ax.axhline(0, color=MUTED, lw=0.8, zorder=4)
ax.set_title("A · Mean power by hour — one solar-soak / evening-peak cycle a day", loc="left")
ax.set_ylabel("MW  (charge below zero)")
ax.set_xlabel("hour of day")
ax.set_xticks(range(0, 24, 3))
ax.grid(axis="y", zorder=0)
ax.legend(loc="upper left", ncols=2)
ax.set_xlim(-0.8, 23.8)

# ---- B: mean implied SOC by hour --------------------------------------------
ax = axes[0, 1]
E = pd.Series(np.cumsum((chg * eta - dis / eta) * DT), index=idx)
soc = E.groupby(day).transform(lambda v: v - v.min())
ms = pd.Series(soc.to_numpy(), index=h).groupby(level=0).mean()
ax.fill_between(ms.index, 0, ms.values, color=BLUE, alpha=0.14, zorder=2)
ax.plot(ms.index, ms.values, color=BLUE, lw=2, zorder=3)
ax.axhline(3966, color=INK2, lw=1.2, ls=(0, (5, 3)), zorder=4)
ax.text(0.3, 3966 * 0.955, "E_max = 3,966 MWh  (largest daily swing)",
        fontsize=8, color=INK2, va="top")
ax.set_title("B · Mean implied state of charge, anchored at each day's minimum", loc="left")
ax.set_ylabel("MWh above the day's low")
ax.set_xlabel("hour of day")
ax.set_xticks(range(0, 24, 3))
ax.set_ylim(0, 4300)
ax.grid(axis="y", zorder=0)
ax.set_xlim(-0.8, 23.8)

# ---- C: continuous SOC path drifts ------------------------------------------
ax = axes[1, 0]
for rtv, c, lab in [(0.82, ORANGE, "η_rt = 0.82"), (rt, BLUE, f"η_rt = {rt:.3f}  (zero drift)"),
                    (0.86, AQUA, "η_rt = 0.86")]:
    e = np.sqrt(rtv)
    p = np.cumsum((chg * e - dis / e) * DT)
    ax.plot(idx, p - p[0], color=c, lw=1.8, label=lab, zorder=3)
ax.axhspan(0, 3966, color=INK2, alpha=0.07, zorder=1)
ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
ax.set_title("C · Run the SOC recursion continuously and it walks out of the reservoir", loc="left")
ax.set_ylabel("cumulative implied SOC (MWh)")
ax.grid(axis="y", zorder=0)
ax.legend(loc="lower left")
ax.text(idx[int(len(idx) * 0.52)], 2600, "reservoir band  0 – 3,966 MWh",
        fontsize=8, color=INK2)
for lb in ax.get_xticklabels():
    lb.set_rotation(0)

# ---- D: swing needed vs horizon ---------------------------------------------
ax = axes[1, 1]
hz = [(12, "1h"), (36, "3h"), (72, "6h"), (144, "12h"), (288, "24h"),
      (576, "48h"), (2016, "7d"), (8640, "30d")]
xs = np.arange(len(hz))
p50, p99, mx = [], [], []
for n, _ in hz:
    r = (E.rolling(n).max() - E.rolling(n).min()).dropna()
    p50.append(r.quantile(.5)); p99.append(r.quantile(.99)); mx.append(r.max())
for ys, c, lab in [(p50, AQUA, "median window"), (p99, ORANGE, "99th pct window"),
                   (mx, BLUE, "worst window")]:
    ax.plot(xs, ys, color=c, lw=2, marker="o", ms=5, label=lab, zorder=3,
            markeredgecolor=SURF, markeredgewidth=1.2)
ax.axhline(3966, color=INK2, lw=1.2, ls=(0, (5, 3)), zorder=4)
ax.text(0.05, 4270, "E_max = 3,966 MWh", fontsize=8, color=INK2, va="bottom")
ax.axvline(1, color=RED, lw=1.2, ls=(0, (2, 2)), zorder=2)
ax.text(1.12, 5150, "planB gap = 3h", fontsize=8, color=RED)
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in hz])
ax.set_title("D · Reservoir the recursion needs, by window length", loc="left")
ax.set_ylabel("SOC swing (MWh)")
ax.set_xlabel("window length")
ax.grid(axis="y", zorder=0)
ax.legend(loc="lower right")

for ax in axes.ravel():
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(rect=(0, 0, 1, 0.955))
out = "quick_findings/vic_battery_constraints/battery_constraints.png"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=160)
print("wrote", out)
