"""Counterfactual + placebo evaluation of the trained bi-LSTM gap imputer.

Reconstruction WAPE (train.py) proves the imputer fills real gaps well. This adds
the counterfactual checks (lit_review.md theme 8 validation toolbox). TWO scenarios:

  --scenario increase (legacy)
      placebo    feed the gap UNCHANGED (g=0): response vs base must be ~0.
      scenario   +g% demand INSIDE the gap only; boundaries pinned to the real
                 measured 10:55/14:00 dispatch. Honest caveat: a pure increase
                 with measured boundaries assumes the off-window world is
                 unchanged (right-boundary landing strip, README).

  --scenario shift (the corrected counterfactual)
      FixedPercentageShift(q, q): demand is REDUCED uniformly by q% at every
      non-free interval and that energy (plus a q% rebound) lands in the free
      window 11:00-14:00; free-window price -> 0. Renewables are fixed, so
      net_demand moves exactly as demand does -- DOWN off-window, UP in-window.
      Under this shift NO step's true dispatch is known (the user's point):
      pinning measured boundaries would be inconsistent. So the off-window
      dispatch is replaced by the PROJECTED FEASIBLE REFERENCE -- each whole day
      POCS-projected onto {balance to shifted nd}∩{ramp}∩{box}∩{SOC} with free
      endpoints (Option A, certified in shift_feasibility.py) -- and the gap is
      imputed with boundaries/context taken from that reference. Default
      q = 3.699%, the largest equal shift feasible on ALL 186 test days
      (the demand-cap q_max 4.856% is dispatch-infeasible on 63 days).

    python3 imputation/scenario_eval.py --scenario increase --g 10
    python3 imputation/scenario_eval.py --scenario shift            # q* = 3.699
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
from gap_data import (load_flats, test_gap_windows, TARGETS, SIGN,             # noqa: E402
                      TARGET_FEAT_IDX, ND_COL, DEM_COL, PRICE_COL, GAP_HOURS)
import constraints as C                                                       # noqa: E402
from constraint_layers import deployed_fill                                   # noqa: E402
from model import BiLSTMImputer                                               # noqa: E402
from shift_model import FixedPercentageShift                                  # noqa: E402
from shift_feasibility import free_boundary_project                           # noqa: E402

OUT = HERE / "results"
BATT_CAP_MWH = 4735.75
ETA = np.sqrt(0.834)
DT = 5 / 60
SHIFT_Q = 3.699    # largest ALL-days dispatch-feasible equal shift (shift_feasibility.py)


def impute_project(model, f, gw, context, device, g_pct, mode="posthoc"):
    """Return projected fill (N,6) MW for a +g_pct demand scenario (g=0 => base).
    Scores Π(F(x)): `mode` must match how the checkpoint was TRAINED (rayen_traj
    applies its ray-shot before the shared projection; posthoc/unrolled need only Π)."""
    tfi = np.asarray(TARGET_FEAT_IDX)
    N = len(gw.gap_idx); g0 = gw.gap_idx[0]
    xs = f.Xte[g0 - context: g0 + N + context].copy()
    gsl = slice(context, context + N)
    nd_mw = gw.nd_mw.copy()
    if g_pct:
        # Demand shock lands ENTIRELY on the dispatchables: wind/solar/curtailment
        # are weather-driven and do NOT scale with demand, so
        #   net_demand_scen = demand_scen - wind - solar - curt
        #                   = net_demand_base + g%*demand   (renewables cancel)
        # i.e. add g%*demand, do NOT scale net_demand by (1+g%).
        dem = xs[gsl, DEM_COL] * f.x_scale[DEM_COL] + f.x_mean[DEM_COL]   # base demand MW
        shock = (g_pct / 100.0) * dem                                     # extra load (MW)
        xs[gsl, DEM_COL] = (dem * (1 + g_pct / 100) - f.x_mean[DEM_COL]) / f.x_scale[DEM_COL]
        nd_base = xs[gsl, ND_COL] * f.x_scale[ND_COL] + f.x_mean[ND_COL]
        xs[gsl, ND_COL] = (nd_base + shock - f.x_mean[ND_COL]) / f.x_scale[ND_COL]
        nd_mw = nd_mw + shock                               # projection target = base + shock
    m = np.ones((xs.shape[0], 1), np.float32); m[gsl] = 0.0
    xs[gsl][:, tfi] = 0.0
    with torch.no_grad():
        dev = model(torch.from_numpy(xs[None].astype(np.float32)).to(device),
                    torch.from_numpy(m[None]).to(device))[0, gsl].cpu().numpy()
    tt = (np.arange(1, N + 1) / (N + 1))[:, None]            # residual: interp + deviation
    pL_s, pR_s = f.Yte[g0 - 1], f.Yte[gw.gap_idx[-1] + 1]
    fill = ((pL_s[None, :] + tt * (pR_s - pL_s)[None, :]) + dev) * f.y_scale + f.y_mean
    fill = deployed_fill(fill, gw.pL_mw, gw.pR_mw, nd_mw, mode)   # F(x)
    P, resid = C.project_gap(fill, gw.pL_mw, gw.pR_mw, nd_mw)     # Π(F(x))
    return P, resid, nd_mw


def build_shift_scenario(f, q: float):
    """FixedPercentageShift(q,q) over the WHOLE test flat (incl. the lb_carry
    train-tail prefix, whose timestamps are reconstructed -- it is exactly one
    contiguous day). Returns (dem_s, nd_s, P_ref), all aligned to test-flat rows:

      dem_s / nd_s  shifted demand / net_demand (renewables fixed => Δnd = Δdem)
      P_ref         the counterfactual REFERENCE dispatch: actual dispatch with
                    every full day POCS-projected onto the shifted constraint set
                    with FREE endpoints (Option A). Off-window dispatch under the
                    shift is unknown; P_ref is a feasible representative of it --
                    context and boundaries are read from P_ref, never from the
                    measured (now-inconsistent) values.

    q=0 short-circuits to the real world (dem, nd, actual dispatch)."""
    dem = f.col_mw(f.Xte, DEM_COL).astype(np.float64)
    nd = f.col_mw(f.Xte, ND_COL).astype(np.float64)
    P_act = f.y_to_mw(f.Yte).astype(np.float64)
    if q == 0:
        return dem, nd, P_act
    lb = f.Xte.shape[0] - len(f.test_index)
    step = pd.Timedelta(minutes=5)
    pre = pd.DatetimeIndex([f.test_index[0] - step * k for k in range(lb, 0, -1)])
    idx = pre.append(pd.DatetimeIndex(f.test_index))
    sh = FixedPercentageShift(q, q, free_hours=GAP_HOURS).transform(
        pd.DataFrame({"demand_mw": dem, "price_aud_per_mwh": 0.0}, index=idx))
    dem_s = sh["demand_mw"].to_numpy()
    nd_s = nd + (dem_s - dem)                                    # renewables fixed
    # project every full contiguous day onto the shifted set (batched POCS)
    day = idx.normalize(); hour = idx.hour
    segs = [pos for d in pd.unique(day)
            for pos in [np.where(np.asarray(day == d))[0]]
            if len(pos) == 288 and hour[pos[0]] == 0
            and (np.diff(idx[pos].values) == np.timedelta64(5, "m")).all()]
    P_ref = P_act.copy()
    if segs:
        P0 = torch.tensor(np.stack([P_act[p] for p in segs]), dtype=torch.float64)
        NDs = torch.tensor(np.stack([nd_s[p] for p in segs]), dtype=torch.float64)
        Pp = free_boundary_project(P0, NDs).numpy()
        for k, pos in enumerate(segs):
            P_ref[pos] = Pp[k]
    return dem_s, nd_s, P_ref


def impute_project_shift(model, f, gw, context, device, scen, mode="posthoc",
                         zero_price=True):
    """Impute + project one gap under a shift scenario `scen = (dem_s, nd_s, P_ref)`
    from build_shift_scenario. Differences vs impute_project (the increase path):
    context dispatch and BOTH boundaries come from P_ref (the projected feasible
    reference, NOT the measured values), drivers are the shifted series everywhere
    in the window, and the free-window price is forced to 0 (the scenario spec)."""
    dem_s, nd_s, P_ref = scen
    tfi = np.asarray(TARGET_FEAT_IDX)
    N = len(gw.gap_idx); g0 = gw.gap_idx[0]
    rows = np.arange(g0 - context, g0 + N + context)
    xs = f.Xte[rows].copy()
    gsl = slice(context, context + N)
    xs[:, tfi] = (P_ref[rows] - f.x_mean[tfi]) / f.x_scale[tfi]   # counterfactual context
    xs[:, ND_COL] = (nd_s[rows] - f.x_mean[ND_COL]) / f.x_scale[ND_COL]
    xs[:, DEM_COL] = (dem_s[rows] - f.x_mean[DEM_COL]) / f.x_scale[DEM_COL]
    if zero_price:
        xs[gsl, PRICE_COL] = (0.0 - f.x_mean[PRICE_COL]) / f.x_scale[PRICE_COL]
    m = np.ones((xs.shape[0], 1), np.float32); m[gsl] = 0.0
    xs[gsl][:, tfi] = 0.0
    with torch.no_grad():
        dev = model(torch.from_numpy(xs[None].astype(np.float32)).to(device),
                    torch.from_numpy(m[None]).to(device))[0, gsl].cpu().numpy()
    pL_mw = P_ref[g0 - 1]; pR_mw = P_ref[gw.gap_idx[-1] + 1]     # reference boundaries
    pL_s = (pL_mw - f.y_mean) / f.y_scale; pR_s = (pR_mw - f.y_mean) / f.y_scale
    tt = (np.arange(1, N + 1) / (N + 1))[:, None]
    fill = ((pL_s[None, :] + tt * (pR_s - pL_s)[None, :]) + dev) * f.y_scale + f.y_mean
    nd_gap = nd_s[gw.gap_idx]
    fill = deployed_fill(fill, pL_mw, pR_mw, nd_gap, mode)        # F(x)
    P, resid = C.project_gap(fill, pL_mw, pR_mw, nd_gap)          # Π(F(x))
    return P, resid, nd_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["increase", "shift"], default="increase",
                    help="increase: legacy +g%% in-gap, measured boundaries. shift: the "
                         "corrected FixedPercentageShift counterfactual -- uniform off-window "
                         "reduction shifted into the free window, projected-reference "
                         "boundaries (no measured dispatch pinned anywhere).")
    ap.add_argument("--g", type=float, default=None,
                    help="scenario magnitude %% (default: 10 for increase, 3.699 for shift)")
    ap.add_argument("--context", type=int, default=48)   # match the trained checkpoint
    ap.add_argument("--ckpt", default=str(OUT / "bilstm_imputer.pt"))
    ap.add_argument("--mode", choices=["posthoc", "unrolled", "rayen_traj"], default="posthoc",
                    help="the constraint mode the checkpoint was TRAINED with (rayen_traj "
                         "applies its ray-shot before the shared projection)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.g is None:
        args.g = 10.0 if args.scenario == "increase" else SHIFT_Q
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    f = load_flats()
    gws = test_gap_windows(f, context=args.context)
    model = BiLSTMImputer().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    if args.scenario == "shift":
        # base = the real world (q=0 short-circuit); scenario = shifted drivers +
        # projected-reference context/boundaries. A placebo run is meaningless here
        # (q=0 IS the base), so only the scenario row is produced.
        print(f"building shift scenario q={args.g:g}% (whole-day POCS reference)...", flush=True)
        scen0 = build_shift_scenario(f, 0.0)
        scenS = build_shift_scenario(f, args.g)
        runs = [(f"shift(q={args.g:g})", args.g)]
    else:
        runs = [("placebo(g=0)", 0.0), (f"scenario(g={args.g:g})", args.g)]

    for tag, g in runs:
        cap_num = cap_den = 0.0
        resp = np.zeros(6); base_sum = np.zeros(6)
        track = []; ramp_bad = neg = 0; bal_max = 0.0; soc_bad = 0
        for gw in gws:
            if args.scenario == "shift":
                Pb, _, ndb = impute_project_shift(model, f, gw, args.context, device,
                                                  scen0, mode=args.mode, zero_price=False)
                Ps, resid, nds = impute_project_shift(model, f, gw, args.context, device,
                                                      scenS, mode=args.mode)
            else:
                Pb, _, ndb = impute_project(model, f, gw, args.context, device, 0.0, mode=args.mode)
                Ps, resid, nds = impute_project(model, f, gw, args.context, device, g, mode=args.mode)
            cap_num += (Ps @ SIGN).sum() - (Pb @ SIGN).sum()
            cap_den += (nds - ndb).sum()
            resp += Ps.sum(0); base_sum += Pb.sum(0)
            track.append(np.abs(Ps @ SIGN - nds))
            if args.scenario == "shift":                 # seams vs the REFERENCE boundaries
                bL, bR = scenS[2][gw.gap_idx[0] - 1], scenS[2][gw.gap_idx[-1] + 1]
            else:
                bL, bR = gw.pL_mw, gw.pR_mw
            d = np.diff(np.vstack([bL, Ps, bR]), axis=0)
            ramp_bad += int(((d > C.R_UP + 0.6) | (d < -(C.R_DN + 0.6))).sum())
            neg += int((Ps < -0.1).sum()); bal_max = max(bal_max, float(resid.max()))
            dE = (np.clip(Ps[:, 4], 0, None) * ETA - np.clip(Ps[:, 5], 0, None) / ETA) * DT
            cum = np.concatenate([[0.0], np.cumsum(dE)])
            if (cum.max() - cum.min()) > BATT_CAP_MWH:
                soc_bad += 1
        capture = cap_num / cap_den if abs(cap_den) > 1e-6 else None
        track = np.concatenate(track)
        row = {"tag": tag, "g": g, "n_days": len(gws),
               "capture": None if capture is None else float(capture),
               "resp_pct": {t: float(100 * (resp[i] - base_sum[i]) / base_sum[i])
                            if abs(base_sum[i]) > 1e-6 else None for i, t in enumerate(TARGETS)},
               "track_p50_mw": float(np.percentile(track, 50)),
               "track_p95_mw": float(np.percentile(track, 95)),
               "ramp_violations_incl_seams": ramp_bad, "n_neg": neg,
               "balance_resid_max_mw": bal_max, "soc_infeasible_gapdays": soc_bad}
        OUT.mkdir(exist_ok=True)
        (OUT / f"scenario_{tag.split('(')[0]}.json").write_text(json.dumps(row, indent=2))
        cap = "—" if capture is None else f"{capture:+.3f}"
        print(f"[{tag:16s}] capture={cap} track_p50={row['track_p50_mw']:.0f}MW "
              f"ramp={ramp_bad} neg={neg} bal_max={bal_max:.0f}MW soc_bad={soc_bad}/{len(gws)} "
              f"coal_resp={row['resp_pct']['coal_brown']:+.1f}% batt_dis_resp="
              f"{row['resp_pct']['battery_discharging']:+.1f}%")


if __name__ == "__main__":
    main()
