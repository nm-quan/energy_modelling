"""Shared loading, definitions and plotting for the actual-grid dispatch study.

Everything here is DESCRIPTIVE: what the VIC fleet actually did, 2022-2026, from
data/preprocessed/hist/5min/net_dispatch_ren/table.parquet. No model involved.
These numbers are the empirical target any counterfactual should be judged against.

DEFINITIONS (fixed here so every script in the study agrees)
  evening peak   17:00-21:00 local (AEST). Chosen from the data: the mean daily
                 demand profile is above 5,900 MW across exactly those hours and
                 tops out at 18:00-19:00.
  midday trough  11:00-14:00 local. The daily minimum of both demand and net
                 demand, and the same window the imputation work uses.
  peak ramp      mean(peak) - mean(trough) per fuel, i.e. how much a fuel is
                 turned UP to serve the evening rise, not how much it runs.
  active         output > ACTIVE_MW. A fuel is "dispatched" at a step if it is
                 above this; below it the unit is effectively off.
  net demand     demand - wind - solar_utility - curtailments. This is what the
                 six dispatchable fuels actually have to cover.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "table.parquet"
OUT = Path(__file__).resolve().parent

FUELS = ["coal_brown", "hydro", "gas_ocgt", "gas_steam", "battery_discharging"]
ALL_DISPATCH = FUELS + ["battery_charging"]
GEN = FUELS                    # the five that SUPPLY; charging is a load, kept out
                               # of any "% of energy" denominator
RENEW = ["wind", "solar_utility"]

# Windows, all local AEST, all picked from the mean hourly profile rather than
# assumed. Demand: overnight min 4,957 MW at 03:00; morning local max 5,574 MW at
# 06:00 (hours 6-8 are within 95% of it); midday min 4,333 MW at 12:00; evening
# max 6,461 MW at 18:00 (hours 18-20 within 95%, 17 at 94.7%).
NIGHT_HOURS = (2, 5)           # overnight trough  -> the base the MORNING peak rises from
MORNING_HOURS = (6, 9)         # morning peak
TROUGH_HOURS = (11, 14)        # midday trough     -> the base the EVENING peak rises from
PEAK_HOURS = (17, 21)          # evening peak
ACTIVE_MW = 10.0
DT = 5.0 / 60.0

# Stack order is bottom -> top and is ALSO the colour order: it interleaves warm
# and cool hues so no two adjacent bands are perceptually close. Validated with
# the dataviz palette checker (lightness / chroma / CVD / normal-vision all PASS;
# the two contrast WARNs are relieved by direct labels + the report tables).
COLORS = {
    "coal_brown":          "#A0661F",   # brown  - baseload
    "hydro":               "#2277BB",   # blue   - mid-merit, peak-following
    "gas_ocgt":            "#E9A020",   # amber  - peaker
    "gas_steam":           "#C0392B",   # red    - rare peaker
    "battery_discharging": "#8E44AD",   # purple - peak only
    "wind":                "#2E8B3D",   # green
    "solar_utility":       "#C9A227",   # gold
    "battery_charging":    "#6E6E6E",   # grey   - drawn below zero (a load)
}
LABEL = {"coal_brown": "coal", "hydro": "hydro", "gas_ocgt": "gas OCGT",
         "gas_steam": "gas steam", "battery_discharging": "battery out",
         "battery_charging": "battery in (load)", "wind": "wind",
         "solar_utility": "solar"}
INK, MUTED = "#22252A", "#6B7280"


def load(year_min: int = 2022) -> pd.DataFrame:
    """5-min table from year_min onward. 2021 is dropped (Oct-Dec only)."""
    t = pd.read_parquet(TABLE)
    return t[t.index.year >= year_min].copy()


def slot(idx: pd.DatetimeIndex) -> np.ndarray:
    """5-minute slot of day, 0..287, in LOCAL time (the index is tz-aware; use
    .hour/.minute, never .values, which silently converts to UTC)."""
    return idx.hour * 12 + idx.minute // 5


def in_hours(idx: pd.DatetimeIndex, lo: int, hi: int) -> np.ndarray:
    return (idx.hour >= lo) & (idx.hour < hi)


def mean_day(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Average daily profile: mean over days at each of the 288 slots."""
    return df.groupby(slot(df.index))[list(cols)].mean()


def draw_day_stack(ax, prof: pd.DataFrame, title: str, legend: bool = False,
                   label_bands: bool = True, ylim=None):
    """Stacked average-day profile. prof is indexed 0..287 (5-min slots).

    Mark spec: 1px surface-coloured separator between bands (the 2px gap of the
    marks spec, at this density), recessive grid, demand and net-demand as thin
    lines on the SAME MW axis (never a second scale), selective direct labels on
    bands large enough to hold one.
    """
    x = prof.index.values / 12.0                       # hours of day
    base = np.zeros(len(prof))
    for f in FUELS:
        top = base + prof[f].values
        ax.fill_between(x, base, top, color=COLORS[f], lw=0.6,
                        edgecolor="#FFFFFF", zorder=2)
        if label_bands:
            # selective direct labels: only a band thick enough to hold text
            # without colliding with its neighbours. Identity for the thin bands
            # comes from the legend, not from crowding four labels together.
            k = int(np.argmax(top - base))
            if (top - base)[k] > 700:
                ax.text(x[k], (base[k] + top[k]) / 2, LABEL[f], ha="center",
                        va="center", fontsize=7.5, color="#FFFFFF", zorder=5,
                        fontweight="bold")
        base = top
    for r in RENEW:
        top = base + prof[r].values
        ax.fill_between(x, base, top, color=COLORS[r], lw=0.6,
                        edgecolor="#FFFFFF", alpha=0.9, zorder=2)
        base = top
    ax.fill_between(x, 0, -prof["battery_charging"].values,
                    color=COLORS["battery_charging"], lw=0.6,
                    edgecolor="#FFFFFF", zorder=2)
    ax.plot(x, prof["demand_mw"].values, color=INK, lw=1.6, ls="--",
            zorder=4, label="demand")
    ax.plot(x, prof["net_demand"].values, color="#111111", lw=1.6,
            zorder=4, label="net demand")
    ax.axvspan(PEAK_HOURS[0], PEAK_HOURS[1], color="#000000", alpha=0.05, zorder=1)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["0", "6", "12", "18", "24"], fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.18, lw=0.6, zorder=0)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=9, color=INK, pad=4)
    if legend:
        handles = [plt_patch(COLORS[f], LABEL[f]) for f in FUELS + RENEW
                   + ["battery_charging"]]
        ax.legend(handles=handles, loc="upper left", ncol=4, fontsize=7,
                  frameon=False, handlelength=1.1, columnspacing=1.0)


def plt_patch(color, label):
    from matplotlib.patches import Patch
    return Patch(facecolor=color, label=label, edgecolor="none")
