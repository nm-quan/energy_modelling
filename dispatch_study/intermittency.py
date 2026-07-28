"""How often is each fuel simply ABSENT, and for how long at a stretch?

Prompted by an observation that held up: there are long periods -- months, not
days -- where hydro and the battery carry the flexible work and gas never appears
at all. This matters for modelling, because a channel that is exactly zero most of
the time is not the same estimation problem as one that is always on.

Writes intermittency.md.

    python3 dispatch_study/intermittency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load, ALL_DISPATCH, ACTIVE_MW, LABEL, OUT   # noqa: E402


def longest_run(mask: np.ndarray):
    """(length, start_i) of the longest run of True."""
    best = cur = bi = ci = 0
    for i, v in enumerate(mask):
        if v:
            if cur == 0:
                ci = i
            cur += 1
            if cur > best:
                best, bi = cur, ci
        else:
            cur = 0
    return best, bi


def main():
    t = load()
    idx = t.index
    day = idx.normalize()
    ym = idx.strftime("%Y-%m")
    ndays = len(np.unique(day))

    L = ["# How often is each fuel absent?", "",
         f"Source: {len(t):,} five-minute intervals, {idx[0].date()} to "
         f"{idx[-1].date()}. **Absent** = at or below {ACTIVE_MW:.0f} MW.", "",
         "## Longest continuous absence", "",
         "| fuel | longest gap | from | to |", "| --- | ---: | --- | --- |"]
    for f in ALL_DISPATCH:
        n, i = longest_run(t[f].values <= ACTIVE_MW)
        L.append(f"| {LABEL[f]} | **{n*5/60/24:,.1f} days** | {idx[i].date()} | "
                 f"{idx[i+n-1].date()} |")
    both = (t["gas_steam"].values <= ACTIVE_MW) & (t["gas_ocgt"].values <= ACTIVE_MW)
    n, i = longest_run(both)
    L += [f"| **no gas at all** | **{n*5/60/24:,.1f} days** | {idx[i].date()} | "
          f"{idx[i+n-1].date()} |", ""]

    L += ["## Share of days on which the fuel never runs", "",
          "| fuel | days never active | share |", "| --- | ---: | ---: |"]
    dmax = t.groupby(day)[ALL_DISPATCH].max()
    for f in ALL_DISPATCH:
        z = (dmax[f] <= ACTIVE_MW).sum()
        L.append(f"| {LABEL[f]} | {z:,} / {ndays:,} | {100*z/ndays:.1f}% |")
    nog = ((dmax["gas_steam"] <= ACTIVE_MW) & (dmax["gas_ocgt"] <= ACTIVE_MW))
    L += [f"| **no gas at all** | {nog.sum():,} / {ndays:,} | **{100*nog.mean():.1f}%** |", ""]

    L += ["## Full calendar months in which gas_steam never once ran", ""]
    mmax = t.groupby(ym)[["gas_steam", "gas_ocgt"]].max()
    full = [m for m in mmax.index if (ym == m).sum() >= 28 * 288]
    zs = [m for m in full if mmax.loc[m, "gas_steam"] <= ACTIVE_MW]
    L += ["- " + ", ".join(f"`{m}`" for m in zs), "",
          f"That is **{len(zs)} of {len(full)} full months**.", ""]

    nd = t.groupby(day)["net_demand"].max()
    L += ["## What separates a gas day from a no-gas day", "",
          "| | mean daily peak net demand |", "| --- | ---: |",
          f"| days with no gas | {nd[nog].mean():,.0f} MW |",
          f"| days with gas | {nd[~nog].mean():,.0f} MW |", "",
          "Gas is a **threshold** phenomenon, not a daily one. It is the fuel of "
          "high-net-demand days, and on an ordinary day it simply does not exist.", ""]

    L += ["## Why this matters for the imputation model", "",
          f"`gas_steam` is at or below {ACTIVE_MW:.0f} MW on "
          f"{100*(dmax['gas_steam']<=ACTIVE_MW).mean():.0f}% of days and has gone "
          f"{longest_run(t['gas_steam'].values <= ACTIVE_MW)[0]*5/60/24:.0f} consecutive "
          "days without running. The correct prediction for that channel is *exactly "
          "zero*, most of the time.", "",
          "An MSE loss on z-scored targets cannot express that: it will settle on a "
          "small positive number every step, which is wrong on the ~72% of days the "
          "channel is off and barely penalised on the rest. This is the direct cause "
          "of the 248% gas_steam MRE in the interpolation benchmark — the denominator "
          "is near zero because the truth is near zero, while the prediction is not.", "",
          "By contrast hydro and battery discharge have never been absent for more "
          "than about two days. They are genuinely always-on channels and an "
          "always-on estimator suits them.", "",
          "Practical consequence: `gas_steam` (and to a lesser degree `gas_ocgt`) "
          "wants zero-inflated treatment — predict P(on) and the level separately — "
          "or at minimum a metric that does not reward a small constant.", ""]

    (OUT / "intermittency.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT/'intermittency.md'}")


if __name__ == "__main__":
    main()
