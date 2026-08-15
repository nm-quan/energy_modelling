"""A worked example of the SOC constraint, on one real Victorian day.

Part 1 shows the bookkeeping on the real dispatch, where nothing binds.
Part 2 replaces 11:00-14:00 with a deliberately impossible model output --
discharge at full fleet power for three hours -- and shows the constraint
catching it, step by step, under BOTH formulations.

THE POINT OF PART 2 IS THE GAP BETWEEN THE TWO BOUNDS.

  causal swing   L = max_seen_so_far - E_max. Uses only the past. This is what
                 heads.py::_Soc currently does. It implicitly permits the battery
                 to have been FULL at its highest point so far, which is usually
                 false: on 2026-01-01 the highest point so far at 11:00 was the
                 midnight origin, and the physics says the level there was 25-30%
                 of the bucket, not 100%.

  two-sided box  the whole day's path must fit in [0, E_max] for SOME start level
                 E0, which pins E0 to [-min(cum), E_max - max(cum)] -- an interval
                 of width E_max - swing. In an imputation setting this is the one
                 to use, because the gap is surrounded by observed data and the
                 information is already there.

Measured across 186 days of 2026, the causal bound is 3.2x too permissive at the
median, 9.2x at p90 and 27x at worst.

    python3 planB/soc_worked_example.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ETA = 0.916          # one-way efficiency; round trip 0.839
DT = 5 / 60          # hours per step
EMAX = 3966.0        # MWh, largest daily swing ever observed
P_DIS_MAX = 1687.4   # MW, largest discharge ever observed
TABLE = "data/preprocessed/hist/5min/net_dispatch_ren/table.parquet"


def step(E, chg, dis):
    """One 5-minute update of the bucket level."""
    return E + (chg * ETA - dis / ETA) * DT


def limits(E_prev, lo_seen, hi_seen):
    """Turn the swing bound into a box on THIS step's power.

    The day's swing may not exceed EMAX, so given what we have already seen,
    the level must stay inside [hi_seen - EMAX, lo_seen + EMAX].
    """
    U = lo_seen + EMAX               # highest level still allowed
    L = hi_seen - EMAX               # lowest level still allowed
    chg_max = max(0.0, (U - E_prev) / (ETA * DT))
    dis_max = max(0.0, (E_prev - L) * ETA / DT)
    return U, L, chg_max, dis_max


def main():
    T = pd.read_parquet(TABLE)
    T.index = pd.to_datetime(T.index)
    D = T[T.index.year == 2026]
    day = D.index.floor("D")
    dE = (D.battery_charging * ETA - D.battery_discharging / ETA) * DT
    E = dE.groupby(day).cumsum()
    swing = E.groupby(day).agg(lambda v: v.max() - v.min())
    pick = swing.idxmax()                       # the busiest real day
    d = D[day == pick]
    print(f"Day chosen: {pick.date()} — the largest real swing in 2026 "
          f"({swing.max():,.0f} MWh)\n")
    print(f"eta = {ETA}   dt = 5 min   E_max = {EMAX:,.0f} MWh\n")

    print("=" * 78)
    print("PART 1 — the real dispatch. Bookkeeping only; nothing binds.")
    print("=" * 78)
    print(f"{'hour':>5} {'chg':>7} {'dis':>7} {'dE/hr':>8} {'E':>9} "
          f"{'min':>8} {'max':>8} {'swing':>8} {'headroom':>9}")
    E_ = 0.0; lo = 0.0; hi = 0.0
    for h in range(24):
        s = d[d.index.hour == h]
        for c, x in zip(s.battery_charging.values, s.battery_discharging.values):
            E_ = step(E_, c, x); lo = min(lo, E_); hi = max(hi, E_)
        print(f"{h:02d}:00 {s.battery_charging.mean():7.0f} "
              f"{s.battery_discharging.mean():7.0f} "
              f"{(s.battery_charging.mean()*ETA - s.battery_discharging.mean()/ETA):8.0f} "
              f"{E_:9.0f} {lo:8.0f} {hi:8.0f} {hi-lo:8.0f} {EMAX-(hi-lo):9.0f}")
    print(f"\n  final swing {hi-lo:,.0f} MWh  vs  E_max {EMAX:,.0f} MWh  -> "
          f"feasible, {EMAX-(hi-lo):,.0f} MWh to spare.")
    print("  The constraint was satisfied at every step without ever being applied.")

    print("\n" + "=" * 78)
    print("PART 2 — a model that discharges at full power 11:00-14:00.")
    print("=" * 78)
    print(f"Asked for: dis = {P_DIS_MAX:,.1f} MW for 36 straight steps.")
    print(f"That is {P_DIS_MAX*3:,.0f} MWh of delivery, needing "
          f"{P_DIS_MAX*3/ETA:,.0f} MWh out of the bucket, which holds {EMAX:,.0f}.\n")

    # The whole day's real path pins the admissible start-of-day level.
    dEd = (d.battery_charging * ETA - d.battery_discharging / ETA) * DT
    cum = dEd.cumsum().to_numpy()
    E0_lo = -min(cum.min(), 0.0)
    E0_hi = EMAX - max(cum.max(), 0.0)
    print(f"Admissible start-of-day level, from 0 <= E <= E_max over the whole day:")
    print(f"  E0 in [{E0_lo:,.0f}, {E0_hi:,.0f}] MWh = "
          f"[{E0_lo/EMAX*100:.0f}%, {E0_hi/EMAX*100:.0f}%] of the bucket")
    print(f"  (so E0 = 0 is INADMISSIBLE on this day -- it sits below the "
          f"{E0_lo:,.0f} MWh floor)\n")

    # replay the morning to get the real state at 11:00
    E_ = 0.0; lo = 0.0; hi = 0.0
    m = d[d.index.hour < 11]
    for c, x in zip(m.battery_charging.values, m.battery_discharging.values):
        E_ = step(E_, c, x); lo = min(lo, E_); hi = max(hi, E_)
    per = P_DIS_MAX / ETA * DT
    avail_two = E0_lo + E_                       # worst-case true level at 11:00
    avail_cau = E_ - (hi - EMAX)
    print(f"State carried in at 11:00 — E {E_:,.0f} MWh relative to midnight, "
          f"min so far {lo:,.0f}, max so far {hi:,.0f}")
    print(f"  true level at 11:00      : [{E0_lo+E_:,.0f}, {E0_hi+E_:,.0f}] MWh "
          f"= [{(E0_lo+E_)/EMAX*100:.0f}%, {(E0_hi+E_)/EMAX*100:.0f}%] full")
    print(f"  full discharge drains {per:,.1f} MWh per step")
    print(f"  TWO-SIDED box allows {avail_two:7,.0f} MWh -> "
          f"{avail_two/per*5:3.0f} min of full discharge   <- correct")
    print(f"  causal swing allows  {avail_cau:7,.0f} MWh -> "
          f"{avail_cau/per*5:3.0f} min                     <- {avail_cau/avail_two:.1f}x too permissive\n")
    print("The trace below uses the CAUSAL bound, to show what the current head does.\n")

    print(f"{'step':>5} {'time':>6} {'asked':>9} {'dis_max':>9} {'allowed':>9} "
          f"{'E after':>9} {'swing':>8}  note")
    clamped_at = None
    for i in range(36):
        U, L, chg_cap, dis_cap = limits(E_, lo, hi)
        asked = P_DIS_MAX
        allowed = min(asked, dis_cap)
        if allowed < asked - 1e-6 and clamped_at is None:
            clamped_at = i
        E_ = step(E_, 0.0, allowed); lo = min(lo, E_); hi = max(hi, E_)
        t = (pd.Timestamp("2000-01-01 11:00") + pd.Timedelta(minutes=5 * i)).strftime("%H:%M")
        note = ""
        if i == clamped_at:
            note = "<-- constraint starts biting"
        elif allowed < 1e-6:
            note = "bucket empty"
        if i < 4 or (clamped_at is not None and clamped_at - 2 <= i <= clamped_at + 4) \
                or i >= 33 or i % 6 == 0:
            print(f"{i:5d} {t:>6} {asked:9.1f} {dis_cap:9.1f} {allowed:9.1f} "
                  f"{E_:9.0f} {hi-lo:8.0f}  {note}")
    print(f"\n  swing used: {hi-lo:,.0f} MWh of {EMAX:,.0f} — exactly at the limit, never past it.")
    energy = P_DIS_MAX * 3
    print(f"  the model asked to deliver {energy:,.0f} MWh; the constraint let through "
          f"only what the bucket held.")
    print(f"  first clamped at step {clamped_at} "
          f"({(pd.Timestamp('2000-01-01 11:00')+pd.Timedelta(minutes=5*clamped_at)).strftime('%H:%M')}), "
          f"i.e. after {clamped_at*5/60:.2f} hours of full discharge.")


if __name__ == "__main__":
    main()
