"""EXPERIMENT 1 -- is a single static cap the right bound when the fleet changes?

The alternative to a scalar cap is C(t): the installed capacity that actually existed at
each moment, read straight from the MONTHLY table
data/updata/capacity_VIC1_by_fueltech_MS.parquet and forward-filled onto the 5-min grid.

Two ways a static cap can be wrong:
  LOOSE  -- it permits more than the fleet could physically deliver at that date, so it
            is not a constraint at all for that stretch of history
  TIGHT  -- it sits below what the fleet can now deliver, so real dispatch is flagged
            as a violation

Both are measured here against the actual dispatch. The static reference is the
empirical max over THIS window (2025-01..2026-08), rounded outward to 0.1 -- the same
policy the repo's CANON uses, re-derived so nothing outside data/updata is needed.

Scope: power caps only. No SOC / energy reservoir -- that constraint is not established
for this fleet, so counting violations against it would measure the assumption.

    python colab/capacity_experiment/exp1_capacity_bounds.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import load_table  # noqa: E402

# target -> capacity fueltech that bounds it. A BESS is rated for the same power in both
# directions, so both battery targets map to the same fleet total. Curtailment has no
# capacity analogue and is excluded.
TARGET_CAP = {
    "hydro":               "cap_hydro",
    "coal_brown":          "cap_coal_brown",
    "gas_steam":           "cap_gas_steam",
    "gas_ocgt":            "cap_gas_ocgt",
    "battery_charging":    "cap_battery",
    "battery_discharging": "cap_battery",
    "wind":                "cap_wind",
    "solar_utility":       "cap_solar_utility",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--growth", type=float, default=1.25,
                    help="fleet must grow by at least this factor (max C / min C) before "
                         "its bound is made time-varying; below it the static empirical "
                         "max is tighter and is kept (default 1.25)")
    ap.add_argument("--out", default="exp1_capacity_bounds.parquet")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    df = load_table().dropna(subset=list(TARGET_CAP) + list(set(TARGET_CAP.values())))
    idx = df.index
    print(f"rows: {len(df):,}   {idx.min()} .. {idx.max()}")
    print(f"source: data/updata/ (monthly capacity, ffilled to 5-min)\n")

    # static reference = empirical max over this window, rounded outward to 0.1
    static = {t: float(np.ceil(df[t].max() * 10) / 10) for t in TARGET_CAP}

    # ---------------- A. validity ---------------------------------------------------
    print("=" * 92)
    print("A. VALIDITY -- do the actuals ever exceed C(t), the fleet that existed?")
    print("=" * 92)
    print(f"  {'target':20s} {'C(t) min':>10s} {'C(t) max':>10s} {'obs max':>10s} "
          f"{'n_viol':>8s} {'%':>7s} {'max excess':>11s} {'slack':>8s}")
    slack = {}
    for t, cc in TARGET_CAP.items():
        ct, obs = df[cc].values, df[t].values
        v = obs > ct
        n = int(v.sum())
        mx = float((obs[v] - ct[v]).max()) if n else 0.0
        slack[t] = float(np.ceil(mx * 10) / 10)
        print(f"  {t:20s} {ct.min():10.1f} {ct.max():10.1f} {obs.max():10.1f} "
              f"{n:8,d} {100 * n / len(df):7.3f} {mx:11.1f} {slack[t]:8.1f}")
    print("\n  Nameplate is not a strict ceiling -- units run slightly above it. slack is")
    print("  the worst excess rounded outward, so BOUND(t) = C(t) + slack is valid by")
    print("  construction. Everything below uses BOUND(t).")
    bound = pd.DataFrame({t: df[cc].values + slack[t] for t, cc in TARGET_CAP.items()},
                         index=idx)
    print(f"  violations of BOUND(t): "
          f"{ {t: int((df[t].values > bound[t].values).sum()) for t in TARGET_CAP} }")

    # ---------------- B. loose or tight, by quarter ----------------------------------
    print("\n" + "=" * 92)
    print("B. IS THE STATIC CAP LOOSE OR TIGHT?   static / BOUND(t), by quarter")
    print("=" * 92)
    q = idx.to_period("Q")
    qs = sorted(set(q))
    print(f"  {'target':20s} " + " ".join(f"{str(x)[2:]:>7s}" for x in qs))
    for t in TARGET_CAP:
        cells = []
        for x in qs:
            b = bound[t].values[q == x].max()
            cells.append(f"{static[t] / b:7.2f}")
        print(f"  {t:20s} " + " ".join(cells))
    print("\n  >1 means the static cap permits more than the fleet could deliver (LOOSE).")
    print("  <1 means it sits below the fleet's capability (TIGHT -- rejects real dispatch).")

    # ---------------- C. recommended bound -------------------------------------------
    print("\n" + "=" * 92)
    print("C. RECOMMENDED BOUND -- time-varying only where the fleet actually changed")
    print("=" * 92)
    print(f"  materiality threshold: max(C)/min(C) >= {a.growth:.2f}\n")
    print(f"  {'target':20s} {'growth':>8s} {'source':>36s} {'bound (MW)':>20s}")
    rec, policy = {}, {}
    for t, cc in TARGET_CAP.items():
        c = bound[t].values
        growth = c.max() / c.min() if c.min() > 0 else np.inf
        tv = growth >= a.growth
        policy[t] = "time-varying" if tv else "static"
        rec[t] = c if tv else np.full(len(idx), static[t])
        src = "C(t) from monthly capacity" if tv else "static empirical max (fleet stable)"
        rng = f"{rec[t].min():.1f}..{rec[t].max():.1f}" if tv else f"{static[t]:.1f}"
        print(f"  {t:20s} {growth:>7.2f}x {src:>36s} {rng:>20s}")

    nviol = {t: int((df[t].values > rec[t]).sum()) for t in TARGET_CAP}
    tvs = [t for t, p in policy.items() if p == "time-varying"]
    print(f"\n  violations of the recommended bound: {sum(nviol.values())} "
          f"across {len(df):,} rows")

    if not a.no_write:
        out = Path(__file__).with_name(a.out)
        o = pd.DataFrame(rec, index=idx)
        o.index.name = "interval"
        o.to_parquet(out)
        print(f"\nwrote {out.name}  {o.shape}")

    print("\n" + "=" * 92)
    if sum(nviol.values()):
        print(f"VERDICT: recommended bound violated -- {nviol}. Investigate before use.")
    else:
        print(f"VERDICT: valid on all {len(df):,} rows. Time-varying ONLY for: "
              f"{', '.join(tvs) if tvs else '(none)'}.")
        print("         The rest keep the static empirical max, which is tighter than")
        print("         nameplate and still valid because those fleets did not change.")
    print("=" * 92)


if __name__ == "__main__":
    main()
