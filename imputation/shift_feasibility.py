"""Does a FEASIBLE whole-day dispatch EXIST for the Option-A counterfactual?

q_max (demand_simulation) is the largest equal shift rebound=reduction=q that keeps
free-window demand under the demand cap -- it says nothing about whether the SIX
dispatch channels can actually follow the shifted net_demand under ramp/box/SOC.
This certifies that separately, for the exact design we're building:

  * whole day (288 steps), FixedPercentageShift(reb=red=q_max), free window 11-14
  * renewables fixed => Δnet_demand = Δdemand at every step (down off-window, up in-window)
  * Option A: NO anchoring -- the day's endpoints float; only interior consecutive
    pairs are ramp-constrained (pL,pR := the trajectory's own current endpoints)

Method = POCS: start from the ACTUAL dispatch and cyclically project onto
{balance to nd_shift} ∩ {interior ramp} ∩ {box} ∩ {SOC/day}. The intersection is
non-empty iff the residual drives to ~0 (a standard feasibility certificate). We
report per-day balance residual, ramp overshoot, negatives, and SOC swing.

    python3 imputation/shift_feasibility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
from gap_data import load_flats, TARGETS, SIGN, ND_COL, DEM_COL                    # noqa: E402
import constraints as C                                                           # noqa: E402
from shift_model import FixedPercentageShift                                      # noqa: E402

T = 288
FREE_HOURS = (11, 14)
DEMAND_CAP_MW = 10783.7            # demand_simulation/hist_constrained_shift.py


def max_equal_shift_pct(demand, index, cap_mw, free_hours=FREE_HOURS):
    """Largest q (rebound=reduction) keeping free-window demand <= cap (+ q<=1)."""
    free = (index.hour >= free_hours[0]) & (index.hour < free_hours[1])
    d = np.asarray(demand, float)
    day = index.normalize()
    removed = np.where(~free, d, 0.0)
    tmp = pd.DataFrame({"day": day, "removed": removed, "free": free.astype(float)}, index=index)
    shifted = tmp.groupby("day")["removed"].transform("sum").to_numpy()
    n_free = tmp.groupby("day")["free"].transform("sum").to_numpy()
    add = np.where(free & (n_free > 0), shifted / np.where(n_free > 0, n_free, 1.0), 0.0)
    denom = d[free] + add[free]
    return float(100.0 * max(0.0, min(1.0, np.min((cap_mw - d[free]) / denom))))


def free_boundary_project(P0, nd, iters=400, soc=True):
    """Option-A projection: interior ramps only (endpoints := own current values),
    balance snapped to nd, box + per-day SOC. P0 (D,T,6) MW, nd (D,T) MW."""
    dev = P0.device; dt = P0.dtype
    sign = C._SIGN_T.to(dev, dt); rup = C._RUP_T.to(dev, dt); rdn = C._RDN_T.to(dev, dt)
    cap = C._CAP_T.to(dev, dt); free_mask = torch.ones(6, device=dev, dtype=dt)
    cap_eff = C.BATT_CAP_MWH - 200.0
    P = P0.clone()
    for _ in range(iters):
        P = C._balance_project(P, nd, sign, free_mask)
        P = C._ramp_project(P, P[:, 0].clone(), P[:, -1].clone(), rup, rdn)   # free endpoints
        P = torch.minimum(P.clamp_min(0.0), cap)
        if soc:
            P = C._soc_project(P, cap_eff)
    return P


def certify(demand, nd_base, P0, NDb, rows, idx, q, iters=400):
    """Feasibility metrics for an equal shift of q% under Option-A constraints."""
    dsh = FixedPercentageShift(q, q, free_hours=FREE_HOURS).transform(
        pd.DataFrame({"demand_mw": demand, "price_aud_per_mwh": 0.0}, index=idx))["demand_mw"].to_numpy()
    nd_shift = nd_base + (dsh - demand)                             # renewables fixed
    NDs = torch.tensor(np.stack([nd_shift[p] for p in rows]), dtype=torch.float64)
    D = len(rows)
    P = free_boundary_project(P0, NDs, iters=iters)
    resid = (NDs - (P * torch.tensor(SIGN)).sum(-1)).abs().max(1).values      # (D,) worst step/day
    full = torch.cat([P[:, :1], P, P[:, -1:]], 1)
    ramp_over = (torch.clamp(torch.diff(full, dim=1) - C._RUP_T, min=0)
                 + torch.clamp(-torch.diff(full, dim=1) - C._RDN_T, min=0)).max()
    chg = P[..., 4].clamp_min(0); dis = P[..., 5].clamp_min(0)
    dE = (dis * C.ETA - chg / C.ETA) * (5 / 60)
    E = torch.cat([torch.zeros(D, 1), torch.cumsum(dE, 1)], 1)
    swing = E.max(1).values - E.min(1).values
    feas = (resid < 1.0) & (swing <= C.BATT_CAP_MWH)
    rise = (NDs - NDb).clamp_min(0).sum(1) * (5 / 60)
    return {"q": q, "D": D, "n_feas": int(feas.sum()), "resid_worst": float(resid.max()),
            "resid_med": float(resid.median()), "ramp_over": float(ramp_over),
            "neg": int((P < -0.1).sum()), "soc_worst": float(swing.max()),
            "soc_over": int((swing > C.BATT_CAP_MWH).sum()),
            "rise_med": float(rise.median()), "rise_max": float(rise.max())}


def bisect_feasible(demand, nd_base, P0, NDb, rows, idx, hi, steps=9):
    """Largest equal shift with ALL days feasible, bisected in (0, hi]."""
    lo, hi0 = 0.0, hi
    for _ in range(steps):
        mid = (lo + hi0) / 2
        if certify(demand, nd_base, P0, NDb, rows, idx, mid)["n_feas"] == len(rows):
            lo = mid
        else:
            hi0 = mid
    return lo


def main():
    f = load_flats()
    lb = f.Xte.shape[0] - len(f.test_index)
    idx = f.test_index
    demand = f.col_mw(f.Xte, DEM_COL)[lb:]
    nd_base = f.col_mw(f.Xte, ND_COL)[lb:]
    disp = f.y_to_mw(f.Yte)[lb:]                                      # actual dispatch (feasible base)

    day = idx.normalize(); hour = idx.hour; step = np.timedelta64(5, "m")
    rows = [pos for d in pd.unique(day)
            for pos in [np.where(np.asarray(day == d))[0]]
            if len(pos) == T and hour[pos[0]] == 0 and (np.diff(idx[pos].values) == step).all()]
    P0 = torch.tensor(np.stack([disp[p] for p in rows]), dtype=torch.float64)
    NDb = torch.tensor(np.stack([nd_base[p] for p in rows]), dtype=torch.float64)

    q_cap = max_equal_shift_pct(demand, idx, DEMAND_CAP_MW)
    at_cap = certify(demand, nd_base, P0, NDb, rows, idx, q_cap)
    q_star = bisect_feasible(demand, nd_base, P0, NDb, rows, idx, hi=q_cap)
    at_star = certify(demand, nd_base, P0, NDb, rows, idx, q_star)

    print(f"Option-A whole-day feasibility ({at_cap['D']} test days, free endpoints, 400 POCS iters)\n")
    for tag, r in (("demand-cap q_max", at_cap), ("dispatch-feasible q*", at_star)):
        print(f"[{tag} = {r['q']:.3f}%]  feasible {r['n_feas']}/{r['D']} days")
        print(f"    balance resid worst {r['resid_worst']:.2e} MW (median {r['resid_med']:.1e}) | "
              f"ramp {r['ramp_over']:.1e} MW | neg {r['neg']} | "
              f"SOC worst {100*r['soc_worst']/C.BATT_CAP_MWH:.1f}% (over-cap {r['soc_over']}) | "
              f"free-window rise med {r['rise_med']:.0f} MWh/day")
    print(f"\n=> use q = {q_star:.3f}% for the counterfactual (q_max {q_cap:.3f}% is demand-cap only "
          f"and leaves {at_cap['D']-at_cap['n_feas']} days dispatch-INFEASIBLE).")


if __name__ == "__main__":
    main()
