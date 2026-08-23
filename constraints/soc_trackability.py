"""Is the VIC fleet battery reservoir observable, and can it be tracked from dispatch?

Answers the question the SOC constraint work has assumed "no" to since
planB/SOC_DESIGN.md 3 ("the data gives you power in and power out, it never gives
you E").  It does now: script/pull_soc_hist.py -> data/updata/vic_battery_soc.csv
carries measured per-unit MWh from 2025-06-30.

    python3 constraints/soc_trackability.py

Three things are measured, all on the aggregate of the 12 units with a stable
fleet over 2026-02-01..2026-08-13 (see WINDOW / UNITS below for why):

  1. observability -- coverage of the fleet level E(t) and its E_max
  2. trackability  -- how far an open-loop recursion driven only by dispatch
                      drifts away from the measured level, over 3h / 6h / 24h
  3. tightness     -- how much the LEVEL form (0 <= E <= E_max, E_0 measured)
                      binds compared with the SWING form the heads implement now
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOC_CSV = "data/updata/vic_battery_soc.csv"
PWR_CSV = "data/updata/vic_battery_power.csv"
GEN_PQ = "data/updata/vic_generation_20250101_20260814.parquet"

DT = 1.0 / 12.0                      # h per 5-min interval
ETA = float(np.sqrt(0.834))          # repo canon one-side efficiency (imputation/constraints.py)
WINDOW = ("2026-02-01", "2026-08-13 23:55")

# MLB01 is frozen telemetry (89% of intervals |dE| < 0.01 MWh on a 710 MWh
# reservoir) and only starts 2026-03; MRNBESS1/TRGBESS1 do not enter the feed
# until 2026-07-30.  Including any of them changes the fleet mid-window, which
# is a composition change masquerading as a reservoir change.
UNITS = ["BALB1", "BULBES1", "GANNB1", "HBESS1", "KESSB1", "LVES1",
         "MREHA1", "MREHA2", "MREHA3", "PIBESS1", "RANGEB1", "VBB1"]

# power is timestamped at the END of its dispatch interval, SOC is an
# instantaneous snapshot at its label, so power leads SOC by one interval.
LAG = -1


def load():
    """Measured level, dispatch power and the fueltech rollup on ONE 5-min grid.

    The SOC csv is missing ~9% of its intervals outright (107,687 rows over a
    118,080-interval span), so every shift has to happen on a reindexed grid or
    it silently straddles a hole.
    """
    soc = pd.read_csv(SOC_CSV, parse_dates=["interval"], index_col="interval")
    mwh = soc[[c for c in soc.columns if c.endswith("_mwh")]]
    mwh.columns = [c[:-4] for c in mwh.columns]
    pwr = pd.read_csv(PWR_CSV, parse_dates=["interval"], index_col="interval")
    pwr.columns = [c[:-3] for c in pwr.columns]

    grid = pd.date_range(*WINDOW, freq="5min")
    mwh, pwr = mwh.reindex(grid)[UNITS], pwr.reindex(grid)[UNITS]

    gen = pd.read_parquet(GEN_PQ)
    gen["interval"] = pd.to_datetime(gen["interval"]).dt.tz_localize(None)
    roll = (gen[gen.fueltech.isin(["battery_charging", "battery_discharging"])]
            .pivot_table(index="interval", columns="fueltech",
                         values="power_mw", aggfunc="sum").reindex(grid))
    return mwh, pwr, roll


def trapezoid(net):
    """MWh leaving the reservoir over each interval, aligned to the SOC label.

    OpenElectricity power is INSTANTANEOUS, so energy is the trapezoid
    (P_i + P_i-1)/2 * dt, not P_i * dt.  Sign: net is + discharging.
    """
    p = net.shift(LAG)
    return (p + p.shift(1)) / 2.0 * DT


def open_loop(E, dE, horizon, emax):
    """Max |error| of a recursion anchored on the measured level `horizon` steps back."""
    d = pd.DataFrame({"E": E, "dE": dE}).dropna()
    v, p = d["E"].to_numpy(), d["dE"].to_numpy()
    errs = [np.abs(v[i] + np.cumsum(p[i + 1:i + horizon + 1])
                   - v[i + 1:i + horizon + 1]).max()
            for i in range(0, len(v) - horizon, max(1, horizon // 6))]
    e = np.asarray(errs)
    return np.median(e), np.percentile(e, 90), 100 * np.median(e) / emax, \
        100 * np.percentile(e, 90) / emax, len(e)


def main():
    mwh, pwr, roll = load()
    have = mwh.notna().all(axis=1)
    E = mwh.sum(axis=1).where(have)
    emax = mwh.max().sum()

    print(f"=== 1. observability === {WINDOW[0]}..{WINDOW[1]}, {len(UNITS)} units")
    print(f"  fleet level E(t) present on {100 * have.mean():.1f}% of the 5-min grid")
    print(f"  E_max (sum of per-unit observed maxima) = {emax:.0f} MWh")
    print(f"  E ranges {100 * E.min() / emax:.1f}%..{100 * E.max() / emax:.1f}% of E_max; "
          f"within 5% of an edge on {100 * ((E / emax < .05) | (E / emax > .95)).mean():.2f}% of intervals")

    print("\n=== 2. trackability from dispatch alone ===")
    dE_act = E.diff()
    # split each unit on sign BEFORE summing: a unit charging while another
    # discharges must not net out, or the efficiency term sees the wrong throughput
    ok_p = pwr.notna().all(axis=1)
    e_chg = trapezoid((-pwr.clip(upper=0)).sum(axis=1).where(ok_p))
    e_dis = trapezoid((pwr.clip(lower=0)).sum(axis=1).where(ok_p))

    # THE LOSS HAS TWO COMPONENTS AND BOTH ARE REAL. A constant draw alone and an
    # efficiency alone fit almost equally well at step level, because each can
    # absorb the other's mean -- which is how this was first got wrong. What
    # separates them is the DAY-TO-DAY VARIATION of the gap: regressing the daily
    # (fuel - dSOC) gap on daily charge energy gives a slope of 0.161, i.e. losses
    # are 16% of throughput, R2 0.31, where a constant explains 0 by construction.
    grid_eta = np.arange(0.85, 1.0001, 0.0005)
    sse = [np.nansum(((e_chg * e - e_dis / e) - dE_act) ** 2) for e in grid_eta]
    eta_fit = float(grid_eta[int(np.argmin(sse))])

    dE = e_chg - e_dis
    loss = np.nanmean((dE - dE_act)[have])        # constant draw alone, MWh/step
    dE_eta = e_chg * eta_fit - e_dis / eta_fit
    loss_r = np.nanmean((dE_eta - dE_act)[have])  # residual constant AFTER eta

    r_chg, r_dis = trapezoid(roll["battery_charging"]), trapezoid(roll["battery_discharging"])
    variants = {
        "units, frictionless": dE,
        "units, constant draw only": dE - loss,
        f"units, eta={eta_fit:.3f} only": dE_eta,
        f"units, eta={eta_fit:.3f} + constant": dE_eta - loss_r,
        f"fueltech rollup, eta={eta_fit:.3f} + constant":
            r_chg * eta_fit - r_dis / eta_fit - loss_r,
        f"fueltech, eta={ETA:.3f}, no lag/trapezoid (CONSTRAINT.md as used today)":
            (roll["battery_charging"] * ETA - roll["battery_discharging"] / ETA) * DT,
    }

    print(f"  fitted one-way eta = {eta_fit:.4f}  ->  eta_rt = {eta_fit ** 2:.4f} "
          f"(repo canon 0.834)")
    print(f"  residual constant draw after eta = {loss_r:.3f} MWh/5min = "
          f"{loss_r * 288:.0f} MWh/day; constant-only fit would be {loss * 288:.0f} MWh/day")
    print(f"  {'driver':36s} {'step R2':>8s}   " +
          "  ".join(f"{h:>16s}" for h in ("3h max|err|", "6h max|err|", "24h max|err|")))
    for name, d in variants.items():
        ok = dE_act.notna() & d.notna() & have
        a, b = dE_act[ok], d[ok]
        r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
        cells = []
        for h in (36, 72, 288):
            med, p90, pm, pp, _ = open_loop(E, d, h, emax)
            cells.append(f"{pm:5.1f}% /{pp:5.1f}%")
        print(f"  {name:36s} {r2:8.3f}   " + "  ".join(f"{c:>16s}" for c in cells))
    print("  (cells are median / p90 of the worst error inside the horizon, as % of E_max)")

    print("\n=== 3. how hard does it bind: SWING form vs LEVEL form ===")
    v = E.to_numpy()
    good = ~np.isnan(v)
    for h, lab in ((36, "3h"), (72, "6h"), (288, "24h")):
        idx = [i for i in range(0, len(v) - h, max(1, h // 6)) if good[i:i + h + 1].all()]
        seg = np.array([v[i:i + h + 1] for i in idx])
        e0 = seg[:, 0]
        swing = seg.max(1) - seg.min(1)
        up = (seg.max(1) - e0) / np.maximum(emax - e0, 1.0)   # charge headroom used
        dn = (e0 - seg.min(1)) / np.maximum(e0, 1.0)          # discharge headroom used
        print(f"  {lab:>3s} (n={len(idx)}): swing/E_max med {100 * np.median(swing) / emax:5.1f}% "
              f"max {100 * swing.max() / emax:5.1f}%  ->  swing form binds "
              f"{100 * np.mean(swing > .95 * emax):.2f}% of windows")
        print(f"         level form: charge headroom used med {100 * np.median(up):5.1f}% "
              f"(>95% in {100 * np.mean(up > .95):5.2f}% of windows), "
              f"discharge med {100 * np.median(dn):5.1f}% (>95% in {100 * np.mean(dn > .95):5.2f}%)")
    print(f"\n  the level form admits {emax:.0f} MWh of excursion from a given E_0 "
          f"(E_0 down, E_max-E_0 up); the swing form admits {2 * emax:.0f} MWh. "
          "Knowing E_0 halves the admissible set.")


if __name__ == "__main__":
    main()
