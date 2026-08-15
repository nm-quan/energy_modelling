"""NSW1: how the 11:00-14:00 window changed between May and July 2026.

    python3 script/nsw_window_compare.py --year 2026
    python3 script/nsw_window_compare.py --year 2025

For every day, average each series over the 36 five-minute intervals from 11:00
to 14:00 (the same free window planB uses), then compare the two months.

DEFINITIONS, because "net demand" is ambiguous and the answer depends on it.

  demand      operational demand as the API reports it. VERIFIED to be ALREADY
              net of rooftop solar: it falls to 7,020 MW at noon while rooftop
              peaks at 3,152 MW, and adding rooftop back gives a load shape that
              does not collapse at midday. So rooftop is inside this number.
  net demand  demand - wind - solar_utility. What the dispatchable fleet must
              serve after grid-scale variable renewables. Matches the demand-side
              convention used elsewhere in this repo.

May and July are both non-summer months in NSW, so this is a winter-deepening
comparison, not a seasonal-peak one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "dispatch_study"
W0, W1 = 11, 14
VRE = ["wind", "solar_utility"]
FUELS = ["coal_black", "gas_ccgt", "gas_ocgt", "hydro", "battery_discharging",
         "battery_charging", "wind", "solar_utility", "solar_rooftop",
         "bioenergy_biomass", "distillate", "pumps"]


def load(gen_p, mkt_p):
    g = pd.read_parquet(gen_p)
    m = pd.read_parquet(mkt_p).set_index("interval").sort_index()
    w = (g.pivot_table(index="interval", columns="fueltech", values="power_mw",
                       aggfunc="mean").sort_index())
    d = m.join(w, how="inner").sort_index()
    d = d.interpolate(limit=3).ffill().bfill()          # <200 gaps in 26,496 rows
    for c in FUELS:
        if c not in d:
            d[c] = 0.0
    d["net_demand_mw"] = d["demand_mw"] - d[VRE].sum(axis=1)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--region", default="nsw")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    Y = args.year
    pre = args.region
    gen_p = DATA / f"{pre}_generation_{Y}0501_{Y}0801.parquet"
    mkt_p = DATA / f"{pre}_market_{Y}0501_{Y}0801.parquet"
    for f_ in (gen_p, mkt_p):
        if not f_.exists():
            raise SystemExit(f"missing {f_.name} -- run script/pull_nsw.py "
                             f"--start {Y}-05-01 --end {Y}-08-01")
    PERIODS = {f"May 1 – Jun 1 {Y}": (f"{Y}-05-01", f"{Y}-06-01"),
               f"Jul 1 – Aug 1 {Y}": (f"{Y}-07-01", f"{Y}-08-01")}
    tag = args.suffix or f"_{Y}"
    d = load(gen_p, mkt_p)
    win = d[(d.index.hour >= W0) & (d.index.hour < W1)]
    daily = win.groupby(win.index.normalize()).mean()      # one row per day
    print(f"{len(d):,} five-minute rows, {d.index[0]} .. {d.index[-1]}")
    print(f"{len(daily)} days with a {W0:02d}:00-{W1:02d}:00 window\n")

    cols = ["demand_mw", "net_demand_mw", "price_aud_per_mwh"] + FUELS
    sub = {}
    for name, (a, b) in PERIODS.items():
        mask = (daily.index >= pd.Timestamp(a, tz=daily.index.tz)) & \
               (daily.index < pd.Timestamp(b, tz=daily.index.tz))
        sub[name] = daily.loc[mask, cols]
    (kA, kB) = list(PERIODS)
    A, B = sub[kA], sub[kB]
    _h = d.groupby(d.index.hour)[["demand_mw", "solar_rooftop"]].mean()
    _noon, _mid, _rt = _h.demand_mw.loc[12], _h.demand_mw.loc[0], _h.solar_rooftop.max()

    L = [f"# {args.region.upper()}1 — the {W0:02d}:00-{W1:02d}:00 window, "
         f"{kB} vs {kA}", "",
         f"Daily average over the 36 five-minute intervals from {W0:02d}:00 to "
         f"{W1:02d}:00, then averaged across the days of each month. "
         f"{kA}: {len(A)} days. {kB}: {len(B)} days. Source: OpenElectricity v4, "
         f"{args.region.upper()}1, 5-minute.", "",
         f"`demand` is operational demand and is **already net of rooftop solar** "
         f"— verified on this data: it falls to {_noon:,.0f} MW at noon versus "
         f"{_mid:,.0f} MW at midnight while rooftop peaks at {_rt:,.0f} MW. "
         f"`net demand` subtracts grid-scale wind and solar on top of that.", ""]

    L += ["## The headline", "",
          f"| series | {kA} | {kB} | change | % |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for c, lab, unit in (("demand_mw", "demand", "MW"),
                         ("net_demand_mw", "net demand", "MW"),
                         ("price_aud_per_mwh", "price", "$/MWh")):
        a, b = A[c].mean(), B[c].mean()
        L.append(f"| {lab} ({unit}) | {a:,.0f} | {b:,.0f} | {b - a:+,.0f} | "
                 f"{100 * (b - a) / abs(a):+.1f}% |")
    L += ["",
          f"_Wind + grid solar in the window went "
          f"{A[VRE].sum(axis=1).mean():,.0f} -> {B[VRE].sum(axis=1).mean():,.0f} MW, "
          f"which is why demand and net demand move by different amounts._", ""]

    L += ["## Spread across days (not just the mean)", "",
          f"| series | period | mean | sd | min | median | max |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for c, lab in (("demand_mw", "demand"), ("net_demand_mw", "net demand"),
                   ("price_aud_per_mwh", "price")):
        for k, S in ((kA, A), (kB, B)):
            v = S[c]
            L.append(f"| {lab} | {k} | {v.mean():,.1f} | {v.std():,.1f} | "
                     f"{v.min():,.1f} | {v.median():,.1f} | {v.max():,.1f} |")
    L += [""]

    L += ["## Fuel mix in the window (mean MW)", "",
          f"| fuel | {kA} | {kB} | change | % |", "| --- | ---: | ---: | ---: | ---: |"]
    for c in FUELS:
        a, b = A[c].mean(), B[c].mean()
        if abs(a) < 0.5 and abs(b) < 0.5:
            continue
        pc = f"{100 * (b - a) / abs(a):+.1f}%" if abs(a) > 1e-6 else "n/a"
        L.append(f"| {c} | {a:,.1f} | {b:,.1f} | {b - a:+,.1f} | {pc} |")
    L += [""]

    (OUT / f"nsw_window_may_vs_jul{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ---------------- figures ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(f"{args.region.upper()}1 {Y}", fontsize=12, x=0.005, ha="left")
    for k, S, col in ((kA, A, "#3b6ea5"), (kB, B, "#c1553b")):
        ax[0].plot(np.arange(len(S)) + 1, S["demand_mw"], lw=1.6, color=col,
                   marker="o", ms=3, label=f"{k}  demand")
        ax[0].plot(np.arange(len(S)) + 1, S["net_demand_mw"], lw=1.4, color=col,
                   ls="--", marker="s", ms=2.6, label=f"{k}  net demand")
    ax[0].set_xlabel("day of month"); ax[0].set_ylabel("MW")
    ax[0].set_title(f"Daily average, {W0:02d}:00-{W1:02d}:00", loc="left", fontsize=11)
    ax[0].legend(fontsize=8, frameon=False); ax[0].grid(alpha=0.2)

    for k, (a, b), col in ((kA, PERIODS[kA], "#3b6ea5"), (kB, PERIODS[kB], "#c1553b")):
        mask = (d.index >= pd.Timestamp(a, tz=d.index.tz)) & \
               (d.index < pd.Timestamp(b, tz=d.index.tz))
        p = d.loc[mask].groupby(d.loc[mask].index.hour)[["demand_mw", "net_demand_mw"]].mean()
        ax[1].plot(p.index, p["demand_mw"], lw=2, color=col, label=f"{k}  demand")
        ax[1].plot(p.index, p["net_demand_mw"], lw=1.6, ls="--", color=col,
                   label=f"{k}  net demand")
    ax[1].axvspan(W0, W1, color="#000", alpha=0.08)
    ax[1].set_xlabel("hour"); ax[1].set_xticks([0, 6, 11, 14, 18, 23])
    ax[1].set_title("Diurnal shape (shaded = the window)", loc="left", fontsize=11)
    ax[1].legend(fontsize=8, frameon=False); ax[1].grid(alpha=0.2)

    show = ["coal_black", "gas_ccgt", "gas_ocgt", "hydro", "wind", "solar_utility",
            "solar_rooftop", "battery_discharging", "battery_charging"]
    x = np.arange(len(show))
    ax[2].bar(x - 0.2, [A[c].mean() for c in show], 0.4, label=kA, color="#3b6ea5")
    ax[2].bar(x + 0.2, [B[c].mean() for c in show], 0.4, label=kB, color="#c1553b")
    ax[2].set_xticks(x); ax[2].set_xticklabels(show, rotation=40, ha="right", fontsize=8)
    ax[2].set_ylabel("mean MW in the window")
    ax[2].set_title("Fuel mix in the window", loc="left", fontsize=11)
    ax[2].legend(fontsize=8, frameon=False); ax[2].grid(axis="y", alpha=0.2)
    for a_ in ax:
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"nsw_window_may_vs_jul{tag}.png", dpi=150, facecolor="white")
    print(f"\nwrote {OUT}/nsw_window_may_vs_jul{tag}.md and .png")


if __name__ == "__main__":
    main()
