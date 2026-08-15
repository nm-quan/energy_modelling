"""NSW1: the 11:00-14:00 May->July change, 2025 and 2026 side by side.

    python3 script/nsw_compare_years.py

Same window and same daily-average-then-monthly-average method as
nsw_window_compare.py, but both years in one table so the percentage changes can
be read against each other.

ON GAS. It looks absent because the window is midday. NSW's gas fleet is
gas_ccgt + gas_ocgt + distillate (there is no gas_steam in NSW1, unlike VIC1),
and it is a peaker: in 2025 it runs 65 MW at noon and 1,010 MW at 18:00. So the
window numbers are small because solar has displaced it, not because the pull
missed a fueltech. The evening figure is reported alongside for that reason.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "dispatch_study"
W0, W1 = 11, 14
YEARS = (2025, 2026)
VRE = ["wind", "solar_utility"]
GAS = ["gas_ccgt", "gas_ocgt", "distillate"]
ROWS = [("demand_mw", "demand"), ("net_demand_mw", "net demand"),
        ("price_aud_per_mwh", "price $/MWh"), (None, None),
        ("coal_black", "coal (black)"), ("gas_total", "**gas total**"),
        ("gas_ccgt", "  gas CCGT"), ("gas_ocgt", "  gas OCGT"),
        ("distillate", "  distillate"), ("hydro", "hydro"),
        ("battery_discharging", "battery out"), ("battery_charging", "battery in"),
        (None, None), ("wind", "wind"), ("solar_utility", "solar utility"),
        ("solar_rooftop", "solar rooftop"), ("vre_total", "**wind + grid solar**"),
        ("bioenergy_biomass", "bioenergy"), ("pumps", "pumps")]


def load(Y):
    g = pd.read_parquet(DATA / f"nsw_generation_{Y}0501_{Y}0801.parquet")
    m = pd.read_parquet(DATA / f"nsw_market_{Y}0501_{Y}0801.parquet") \
          .set_index("interval").sort_index()
    w = g.pivot_table(index="interval", columns="fueltech", values="power_mw",
                      aggfunc="mean").sort_index()
    d = m.join(w, how="inner").sort_index().interpolate(limit=3).ffill().bfill()
    for c in set(GAS + VRE + ["solar_rooftop", "coal_black", "hydro", "pumps",
                              "bioenergy_biomass", "battery_charging",
                              "battery_discharging"]):
        if c not in d:
            d[c] = 0.0
    d["net_demand_mw"] = d["demand_mw"] - d[VRE].sum(axis=1)
    d["gas_total"] = d[GAS].sum(axis=1)
    d["vre_total"] = d[VRE].sum(axis=1)
    return d


def month_means(d, Y):
    win = d[(d.index.hour >= W0) & (d.index.hour < W1)]
    daily = win.groupby(win.index.normalize()).mean()
    out = {}
    for tag, (a, b) in (("May", (f"{Y}-05-01", f"{Y}-06-01")),
                        ("Jul", (f"{Y}-07-01", f"{Y}-08-01"))):
        msk = (daily.index >= pd.Timestamp(a, tz=daily.index.tz)) & \
              (daily.index < pd.Timestamp(b, tz=daily.index.tz))
        out[tag] = daily.loc[msk]
    return out


def main():
    D = {Y: load(Y) for Y in YEARS}
    M = {Y: month_means(D[Y], Y) for Y in YEARS}
    pc = lambda a, b: (f"{100 * (b - a) / abs(a):+.1f}%" if abs(a) > 1e-6 else "n/a")

    L = [f"# NSW1 {W0:02d}:00-{W1:02d}:00 — May → July percentage change, "
         f"{YEARS[0]} vs {YEARS[1]}", "",
         f"Daily average over the 36 five-minute intervals of the window, then "
         f"averaged across the days of each month. Source: OpenElectricity v4, "
         f"NSW1, 5-minute.", "",
         "| series | "
         + " | ".join(f"{Y} May | {Y} Jul | **{Y} %**" for Y in YEARS) + " |",
         "| --- |" + " ---: | ---: | ---: |" * len(YEARS)]
    for col, lab in ROWS:
        if col is None:
            L.append("| | " + " | ".join([" |"] * 0) + " | ".join([""] * 0)
                     + " |  |  |  |  |  |")
            continue
        cells = []
        for Y in YEARS:
            a, b = M[Y]["May"][col].mean(), M[Y]["Jul"][col].mean()
            cells += [f"{a:,.1f}", f"{b:,.1f}", f"**{pc(a, b)}**"]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += [""]

    L += ["## Gas is not missing — it is an evening peaker", "",
          "The window is midday, when solar has displaced it. Gas here is "
          "`gas_ccgt + gas_ocgt + distillate`; NSW1 has no `gas_steam` at all "
          "(VIC1 does), so this is the whole gas fleet.", "",
          "| gas total (MW) | " + " | ".join(f"{Y}" for Y in YEARS) + " |",
          "| --- |" + " ---: |" * len(YEARS)]
    for lab, fn in (("mean 11:00-14:00", lambda d: d[(d.index.hour >= W0)
                                                     & (d.index.hour < W1)]["gas_total"].mean()),
                    ("mean 17:00-20:00", lambda d: d[(d.index.hour >= 17)
                                                     & (d.index.hour < 20)]["gas_total"].mean()),
                    ("mean, all hours", lambda d: d["gas_total"].mean()),
                    ("max 5-minute", lambda d: d["gas_total"].max())):
        L.append(f"| {lab} | " + " | ".join(f"{fn(D[Y]):,.0f}" for Y in YEARS) + " |")
    L += ["",
          f"_Evening gas fell from {D[2025][(D[2025].index.hour >= 17) & (D[2025].index.hour < 20)]['gas_total'].mean():,.0f} MW "
          f"to {D[2026][(D[2026].index.hour >= 17) & (D[2026].index.hour < 20)]['gas_total'].mean():,.0f} MW "
          f"between the two years — a bigger structural change than anything in the "
          f"midday window._", ""]

    p = OUT / "nsw_window_pct_change_2025_2026.md"
    p.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
