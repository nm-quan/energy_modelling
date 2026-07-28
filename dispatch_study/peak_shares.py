"""Fuel proportions during the two daily peaks — morning AND evening.

Victoria has two demand peaks, not one:
  morning  06:00-09:00, local max 5,574 MW at 06:00
  evening  17:00-21:00, global max 6,461 MW at 18:00

They are NOT the same event and are not served the same way. The morning peak
rises out of the overnight trough (02:00-05:00) before solar exists; the evening
peak rises out of the midday trough (11:00-14:00) as solar collapses. So each
peak is measured against its own base.

"Proportion" here = share of DISPATCHED GENERATION in the window. Battery
charging is a load and is reported separately, never inside the denominator.

Writes peak_shares.md and peak_shares.png.

    python3 dispatch_study/peak_shares.py
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
from common import (load, in_hours, GEN, ALL_DISPATCH, ACTIVE_MW, OUT,   # noqa: E402
                    COLORS, LABEL, INK, MUTED, DT,
                    NIGHT_HOURS, MORNING_HOURS, TROUGH_HOURS, PEAK_HOURS)

WINDOWS = {
    "overnight trough": NIGHT_HOURS,
    "MORNING peak": MORNING_HOURS,
    "midday trough": TROUGH_HOURS,
    "EVENING peak": PEAK_HOURS,
}
# each peak is measured against the trough it rises out of
BASE = {"MORNING peak": "overnight trough", "EVENING peak": "midday trough"}


def shares(t, hours):
    """% of generation energy by fuel in the window, plus mean MW and the load."""
    m = in_hours(t.index, *hours)
    mw = t.loc[m, ALL_DISPATCH].mean()
    e = t.loc[m, GEN].clip(lower=0).sum() * DT
    return mw, 100 * e / e.sum()


def main():
    t = load()
    rows_mw, rows_sh = {}, {}
    for name, hrs in WINDOWS.items():
        rows_mw[name], rows_sh[name] = shares(t, hrs)

    # both peaks pooled
    both = in_hours(t.index, *MORNING_HOURS) | in_hours(t.index, *PEAK_HOURS)
    e_both = t.loc[both, GEN].clip(lower=0).sum() * DT
    sh_both = 100 * e_both / e_both.sum()

    L = ["# Fuel proportions during the two daily peaks", "",
         f"Source: {len(t):,} five-minute intervals, {t.index[0].date()} to "
         f"{t.index[-1].date()}, local AEST.", "",
         "Victoria has **two** demand peaks. They are different events and are not "
         "served the same way, so each is measured against the trough it rises out of.", "",
         "| window | hours | mean demand (MW) | mean net demand (MW) |",
         "| --- | --- | ---: | ---: |"]
    for name, (a, b) in WINDOWS.items():
        m = in_hours(t.index, a, b)
        L.append(f"| {name} | {a:02d}:00–{b:02d}:00 | {t.loc[m,'demand_mw'].mean():,.0f} | "
                 f"{t.loc[m,'net_demand'].mean():,.0f} |")
    L.append("")

    # ---------- 1. proportion of generation ----------
    L += ["## 1. Proportion of generation in each peak", "",
          "Share of the five dispatchable generators' energy in the window. "
          "Battery charging is a load and sits outside the denominator (its mean "
          "MW is listed for reference).", "",
          "| fuel | MORNING peak 06–09 | EVENING peak 17–21 | both peaks pooled |",
          "| --- | ---: | ---: | ---: |"]
    for f in GEN:
        L.append(f"| {LABEL[f]} | {rows_sh['MORNING peak'][f]:.1f}% | "
                 f"{rows_sh['EVENING peak'][f]:.1f}% | {sh_both[f]:.1f}% |")
    L += [f"| _{LABEL['battery_charging']} (load, excluded)_ | "
          f"_{rows_mw['MORNING peak']['battery_charging']:,.0f} MW_ | "
          f"_{rows_mw['EVENING peak']['battery_charging']:,.0f} MW_ | _—_ |", ""]

    # ---------- 2. mean MW ----------
    L += ["## 2. Mean output by window (MW)", "",
          "| fuel | " + " | ".join(WINDOWS) + " |",
          "| --- |" + " ---: |" * len(WINDOWS)]
    for f in ALL_DISPATCH:
        L.append(f"| {LABEL[f]} | " +
                 " | ".join(f"{rows_mw[w][f]:,.1f}" for w in WINDOWS) + " |")
    L.append("")

    # ---------- 3. turn-up ----------
    L += ["## 3. Turn-up — who is switched ON for each peak", "",
          "Peak mean minus the mean of the trough that peak rises out of. Battery "
          "charging contributes by *reducing*, so its turn-up is the size of that "
          "reduction.", "",
          "| fuel | morning turn-up (MW) | share | evening turn-up (MW) | share |",
          "| --- | ---: | ---: | ---: | ---: |"]
    tu = {}
    for pk, base in BASE.items():
        d = rows_mw[pk] - rows_mw[base]
        d["battery_charging"] *= -1                 # a reduction supplies power
        tu[pk] = d
    for f in ALL_DISPATCH:
        m, e = tu["MORNING peak"][f], tu["EVENING peak"][f]
        sm = 100 * m / tu["MORNING peak"].clip(lower=0).sum()
        se = 100 * e / tu["EVENING peak"].clip(lower=0).sum()
        L.append(f"| {LABEL[f]} | {m:+,.1f} | {sm:.1f}% | {e:+,.1f} | {se:.1f}% |")
    L += ["", f"_Total turn-up: morning {tu['MORNING peak'].clip(lower=0).sum():,.0f} MW, "
          f"evening {tu['EVENING peak'].clip(lower=0).sum():,.0f} MW._", ""]

    # ---------- 4. appearance ----------
    L += ["## 4. Appearance — % of steps active (>10 MW) in each peak", "",
          "| fuel | morning | evening |", "| --- | ---: | ---: |"]
    for f in ALL_DISPATCH:
        a = 100 * (t.loc[in_hours(t.index, *MORNING_HOURS), f] > ACTIVE_MW).mean()
        b = 100 * (t.loc[in_hours(t.index, *PEAK_HOURS), f] > ACTIVE_MW).mean()
        L.append(f"| {LABEL[f]} | {a:.1f}% | {b:.1f}% |")
    L.append("")

    # ---------- 5. by year ----------
    for pk, hrs in (("MORNING peak", MORNING_HOURS), ("EVENING peak", PEAK_HOURS)):
        L += [f"## 5{'a' if pk.startswith('MORN') else 'b'}. {pk} — generation share by year", "",
              "| year | " + " | ".join(LABEL[f] for f in GEN) + " |",
              "| --- |" + " ---: |" * len(GEN)]
        for y, sub in t.groupby(t.index.year):
            _, s = shares(sub, hrs)
            L.append(f"| {y} | " + " | ".join(f"{s[f]:.1f}%" for f in GEN) + " |")
        L.append("")

    (OUT / "peak_shares.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    order = ["MORNING peak", "EVENING peak"]
    # 1. generation share, grouped bars
    w = 0.38
    for i, pk in enumerate(order):
        vals = [rows_sh[pk][f] for f in GEN]
        ax[0].bar(np.arange(len(GEN)) + (i - 0.5) * w, vals, w,
                  color=[COLORS[f] for f in GEN], alpha=1.0 if i else 0.55,
                  edgecolor="white", lw=1.2,
                  label="evening" if i else "morning")
    ax[0].set_xticks(range(len(GEN)))
    ax[0].set_xticklabels([LABEL[f] for f in GEN], fontsize=7.5, rotation=20, ha="right")
    ax[0].set_ylabel("% of generation", fontsize=8, color=MUTED)
    ax[0].set_title("Share of generation in each peak\n(pale = morning, solid = evening)",
                    loc="left", fontsize=9.5, color=INK)

    # 2. turn-up share
    for i, pk in enumerate(order):
        vals = [100 * tu[pk][f] / tu[pk].clip(lower=0).sum() for f in ALL_DISPATCH]
        ax[1].bar(np.arange(len(ALL_DISPATCH)) + (i - 0.5) * w, vals, w,
                  color=[COLORS[f] for f in ALL_DISPATCH], alpha=1.0 if i else 0.55,
                  edgecolor="white", lw=1.2)
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set_xticks(range(len(ALL_DISPATCH)))
    ax[1].set_xticklabels([LABEL[f] for f in ALL_DISPATCH], fontsize=7.5,
                          rotation=20, ha="right")
    ax[1].set_ylabel("% of turn-up", fontsize=8, color=MUTED)
    ax[1].set_title("Share of the turn-up into each peak\n(pale = morning, solid = evening)",
                    loc="left", fontsize=9.5, color=INK)

    # 3. evening-peak share by year, stacked
    yrs = sorted(t.index.year.unique())
    base = np.zeros(len(yrs))
    for f in GEN:
        v = np.array([shares(t[t.index.year == y], PEAK_HOURS)[1][f] for y in yrs])
        ax[2].bar(yrs, v, bottom=base, color=COLORS[f], edgecolor="white", lw=1.2,
                  label=LABEL[f])
        base += v
    ax[2].set_ylim(0, 100); ax[2].set_xticks(yrs)
    ax[2].set_ylabel("% of generation", fontsize=8, color=MUTED)
    ax[2].set_title("Evening-peak generation share by year", loc="left",
                    fontsize=9.5, color=INK)
    ax[2].legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    for a in ax:
        a.grid(True, axis="y", alpha=0.18, lw=0.6)
        a.tick_params(labelsize=8, colors=MUTED, length=0)
        for s in ("top", "right", "left", "bottom"):
            a.spines[s].set_visible(False)
    fig.suptitle("Victoria 2022–2026 — morning vs evening peak composition",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUT / "peak_shares.png", dpi=140, facecolor="white"); plt.close(fig)
    print(f"\nwrote {OUT/'peak_shares.md'} and {OUT/'peak_shares.png'}")


if __name__ == "__main__":
    main()
