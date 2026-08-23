"""EXPERIMENT 4 -- the ramp-duration envelope R(k).

A single per-step ramp limit is inert over multi-step horizons: it permits any
trajectory that repeats the 5-minute maximum, and the fleet has never sustained that
rate. R(k) replaces the one number with a curve -- the largest move the fleet has ever
made over each time distance k -- which is the same idea as a running record table
(100 m pace is not marathon pace).

    |p(t+k) - p(t)| <= R_up(k)          k = 1, 2, 3, 6, 9, 12, 18, 24, 30, 36
    |p(t)   - p(t+k)| <= -R_dn(k)

Two envelopes, matching the repo convention in CONSTRAINT.md:
  max     validation -- 0 historical violations by construction, but each channel's
          worst value is often a single unit trip (gas_steam -500 = the whole fueltech
          to zero on 2025-06-13), so the envelope is set by a contingency
  p99.9   modelling  -- what a trajectory can be trained and scored through

Scope: power ramps only. Bounds are FIXED per fueltech, not time-varying -- exp4's
scale test showed the battery fleet tripled (735 -> 2260 MW) while its 5-min ramp grew
1.13x (max) / 1.54x (p99.9), so installed capacity is the wrong scale for a ramp even
though it is the right scale for a cap (exp1). Aggregate ramping is a coordination
property, not a capacity property.

Data: data/updata only, via pipeline.py.

    python colab/capacity_experiment/exp4_ramp_envelope.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import TARGETS, load_table  # noqa: E402

KS = [1, 2, 3, 6, 9, 12, 18, 24, 30, 36]      # 5 min .. 3 h, the gap length
ENFORCE = [1, 3, 6, 12, 24, 36]               # recommended subset for the projection
PCT = 99.9


def envelope(p: np.ndarray, seg: np.ndarray, ks: list[int]) -> dict:
    """R_dn/R_up at both statistics, rounded OUTWARD to 0.1 MW.

    Made monotone in k by a running max/min. A displacement reachable in k steps is
    always reachable in k+1 (move, then hold -- holding is feasible under any envelope
    containing 0), so the true R is non-decreasing. Where the raw estimate dips it is
    a sampling artifact: wind's largest 5-min jump happened to be followed by a
    reversal, so no 10-min window contained a bigger net move. Taking the raw value
    would assert that a big jump MUST reverse.
    """
    out, run = {}, {"max_up": 0.0, "max_dn": 0.0, "p999_up": 0.0, "p999_dn": 0.0}
    for k in ks:
        d = p[k:] - p[:-k]
        d = d[seg[k:] == seg[:-k]]
        pc_up = np.percentile(d, PCT)
        pc_dn = np.percentile(d, 100 - PCT)
        raw = {
            "max_up": float(np.ceil(d.max() * 10) / 10),
            "max_dn": float(np.floor(d.min() * 10) / 10),
            "p999_up": float(np.ceil(pc_up * 10) / 10),
            "p999_dn": float(np.floor(pc_dn * 10) / 10),
        }
        run = {"max_up": max(run["max_up"], raw["max_up"]),
               "max_dn": min(run["max_dn"], raw["max_dn"]),
               "p999_up": max(run["p999_up"], raw["p999_up"]),
               "p999_dn": min(run["p999_dn"], raw["p999_dn"])}
        out[k] = {
            "n_pairs": int(len(d)),
            **run,
            # how many real moves the modelling envelope rejects, at the enforced bound
            "p999_over": int(((d > run["p999_up"]) | (d < run["p999_dn"])).sum()),
            "monotone_lift": float(round(max(run["max_up"] - raw["max_up"],
                                             raw["max_dn"] - run["max_dn"]), 1)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp4_ramp_envelope")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    df = load_table().dropna(subset=TARGETS)
    seg = (df.index.to_series().diff() != pd.Timedelta("5min")).cumsum().values
    print(f"rows {len(df):,}   {df.index.min()} .. {df.index.max()}   "
          f"{seg.max() + 1} contiguous segments\n")

    env = {t: envelope(df[t].values, seg, KS) for t in TARGETS}

    # ---------------- A. the table ----------------------------------------------------
    for stat, lo, hi in (("max", "max_dn", "max_up"), (f"p{PCT}", "p999_dn", "p999_up")):
        print("=" * 108)
        print(f"A. RAMP-DURATION ENVELOPE R(k)   [{stat}]   MW over k steps of 5 min")
        print("=" * 108)
        print(f"  {'target':22s} " + "".join(f"{f'{k * 5}m':>8s}" for k in KS))
        for t in TARGETS:
            print(f"  {t:20s} up " + "".join(f"{env[t][k][hi]:8.0f}" for k in KS))
            print(f"  {'':20s} dn " + "".join(f"{env[t][k][lo]:8.0f}" for k in KS))
        print()

    # ---------------- B. how much does it tighten? -----------------------------------
    print("=" * 108)
    print("B. TIGHTENING vs the per-step box.   R(k) as a fraction of k x R(1)")
    print("   1.00 = the per-step box was already right.  0.10 = it was 10x too loose.")
    print("=" * 108)
    for stat, lo, hi in (("max", "max_dn", "max_up"), (f"p{PCT}", "p999_dn", "p999_up")):
        print(f"  [{stat}]")
        print(f"  {'target':22s} " + "".join(f"{f'{k * 5}m':>8s}" for k in KS[1:]))
        for t in TARGETS:
            r1 = max(env[t][1][hi], -env[t][1][lo])
            if r1 <= 0:
                continue
            cells = []
            for k in KS[1:]:
                rk = max(env[t][k][hi], -env[t][k][lo])
                cells.append(f"{rk / (k * r1):8.2f}")
            print(f"  {t:22s} " + "".join(cells))
        print()

    # ---------------- C. what it costs in the 3h gap ---------------------------------
    print("=" * 108)
    print("C. WHAT THE PER-STEP BOX PERMITS OVER THE 3h GAP, vs what R(36) permits")
    print("=" * 108)
    print(f"  {'target':22s} {'36 x R(1)':>12s} {'R(36)':>10s} {'ratio':>8s} "
          f"{'observed max |move|':>21s}")
    for t in TARGETS:
        r1 = max(env[t][1]["p999_up"], -env[t][1]["p999_dn"])
        r36 = max(env[t][36]["p999_up"], -env[t][36]["p999_dn"])
        if r1 <= 0:
            continue
        print(f"  {t:22s} {36 * r1:12.0f} {r36:10.0f} {36 * r1 / r36:8.1f}x "
              f"{max(env[t][36]['max_up'], -env[t][36]['max_dn']):21.0f}")
    print("\n  ratio = how many times looser the per-step box is than the real envelope")
    print("  over a 3-hour window. This is the constraint that was never firing.")

    # ---------------- D. validity + well-posedness ------------------------------------
    print("\n" + "=" * 108)
    print("D. CHECKS")
    print("=" * 108)
    bad_mono, bad_sub, viol_max = [], [], []
    for t in TARGETS:
        p, e = df[t].values, env[t]
        for k in KS:
            d = p[k:] - p[:-k]
            d = d[seg[k:] == seg[:-k]]
            if (d > e[k]["max_up"]).any() or (d < e[k]["max_dn"]).any():
                viol_max.append((t, k))
            if k > KS[0]:
                kp = KS[KS.index(k) - 1]
                for st in ("max", "p999"):
                    if e[k][f"{st}_up"] < e[kp][f"{st}_up"] - 1e-9:
                        bad_mono.append((t, k, st, "up"))
                    if e[k][f"{st}_dn"] > e[kp][f"{st}_dn"] + 1e-9:
                        bad_mono.append((t, k, st, "dn"))
        # subadditivity: R(a+b) <= R(a) + R(b), automatic for max, checked for p99.9
        for i, ka in enumerate(KS):
            for kb in KS[i:]:
                if ka + kb in e:
                    if e[ka + kb]["p999_up"] > e[ka]["p999_up"] + e[kb]["p999_up"] + 1e-6:
                        bad_sub.append((t, ka, kb, "up"))
    lifted = [(t, k, env[t][k]["monotone_lift"]) for t in TARGETS for k in KS
              if env[t][k]["monotone_lift"] > 0]
    print(f"  max envelope rejects real history : {len(viol_max)} (must be 0)")
    print(f"  monotonicity R(k) non-decreasing  : {len(bad_mono)} failures (must be 0)")
    print(f"  cells lifted by the running max   : {len(lifted)} of "
          f"{len(TARGETS) * len(KS)}, worst +{max([x[2] for x in lifted], default=0):.0f} MW")
    print(f"  p{PCT} subadditive R(a+b)<=R(a)+R(b): {len(bad_sub)} failures "
          f"(not required, but non-subadditive rows are the ones that bite hardest)")
    if bad_sub:
        print(f"    e.g. {bad_sub[:6]}")
    tot = sum(env[t][k]["p999_over"] for t in TARGETS for k in ENFORCE)
    npair = sum(env[t][k]["n_pairs"] for t in TARGETS for k in ENFORCE)
    print(f"  real moves outside the p{PCT} envelope on the enforced subset {ENFORCE}: "
          f"{tot:,} / {npair:,} = {100 * tot / npair:.3f}%")
    print(f"    (a two-sided p{PCT} implies ~0.2% by construction -- same convention as"
          f" CONSTRAINT.md envelope B)")

    if not a.no_write:
        d = Path(__file__).parent
        payload = {
            "source": "data/updata",
            "rows": int(len(df)),
            "window": [str(df.index.min()), str(df.index.max())],
            "ks": KS,
            "enforce": ENFORCE,
            "percentile": PCT,
            "note": "bounds are FIXED per fueltech; see exp4 scale test -- capacity is "
                    "the right scale for a cap (exp1) but not for a ramp",
            "envelope": {t: {str(k): v for k, v in env[t].items()} for t in TARGETS},
        }
        (d / f"{a.out}_results.json").write_text(json.dumps(payload, indent=1))
        rows = [{"target": t, "k": k, "minutes": k * 5, **env[t][k]}
                for t in TARGETS for k in KS]
        pd.DataFrame(rows).to_parquet(d / f"{a.out}.parquet", index=False)
        print(f"\nwrote {a.out}_results.json and {a.out}.parquet")


if __name__ == "__main__":
    main()
