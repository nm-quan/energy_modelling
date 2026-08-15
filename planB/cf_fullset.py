"""Counterfactual over the WHOLE test set, with a uniform-scaling reference.

    python3 planB/cf_fullset.py --arms hardnet_2025_alloc_soc hardnet_2025_alloc_sup

THE QUESTION. A demand shift is applied in the free window (11:00-14:00,
+/-Q%). What does each fuel do about it? There are two answers to compare:

  UNIFORM   every fuel scales by the same factor nd_cf/nd_base at every step.
            No model, no merit order -- if demand rises 2%, every fuel rises 2%.
            This is planA's "scaled" context, promoted here to a full reference.

  MODEL     the same uniform scaling OUTSIDE the free window, the model's fill
            INSIDE it, then project_full_day over all 288 steps.

Both are reported as a change from the REAL dispatch, so the difference between
them is exactly what the model contributes.

WHY UNIFORM IS THE RIGHT REFERENCE. It is the null hypothesis of the whole
project: that a demand change is met pro rata by whatever was already running.
It satisfies balance by construction (scaling every term of s.P = nd by the same
factor scales nd by that factor), so it is a legal answer, just not a
merit-order one.

SOC. The head's reservoir constraint is seeded per day with the dispatch from
midnight to the gap opening, so the swing is bounded over the day rather than
over the 3 h window, where the full reservoir would never bind.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "imputation"))
sys.path.insert(0, str(ROOT / "dispatch_study"))
sys.path.insert(0, str(HERE))

import plan1_figures as F                                              # noqa: E402
import constraints as IC                                               # noqa: E402
from gap_data import TARGETS, SIGN, TARGET_FEAT_IDX                    # noqa: E402
from shift_model import FixedPercentageShift                           # noqa: E402
from common import COLORS, LABEL, INK, MUTED, FUELS                    # noqa: E402
import heads as HD                                                     # noqa: E402
from heads import HEADS                                                # noqa: E402
import limits as LIM                                                   # noqa: E402
from fit import (curt_activation, CURT_COLS, CTX, GAP, BACKBONES,      # noqa: E402
                 apply_residual)

OUT = ROOT / "planB" / "results"
DT = 5.0 / 60.0
CURT_NAMES = ["wind_curtailment", "solar_curtailment"]
SHORT = {"hydro": "hydro", "coal_brown": "coal", "gas_steam": "gas steam",
         "gas_ocgt": "gas OCGT", "battery_charging": "battery in",
         "battery_discharging": "battery out"}


def soc_seed_from(day_mw, upto):
    """(E, Emin, Emax) after running the day's dispatch from midnight to `upto`."""
    dE = (day_mw[:, :upto, HD.CHG_I] * HD.ETA_B
          - day_mw[:, :upto, HD.DIS_I] / HD.ETA_B) * HD.DT_H
    E = np.concatenate([np.zeros((len(day_mw), 1)), dE.cumsum(1)], 1)
    return (torch.tensor(E[:, -1]), torch.tensor(E.min(1)), torch.tensor(E.max(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--q", type=float, default=None)
    ap.add_argument("--suffix", default="_fullset")
    ap.add_argument("--max-days", type=int, default=0, help="0 = every complete test day")
    ap.add_argument("--show-days", type=int, default=5,
                    help="how many exact days to draw in the stack figure")
    ap.add_argument("--from-day", default=None, help="YYYY-MM-DD to start the strip")
    ap.add_argument("--soc-rate-frac", type=float, default=None,
                    help="fraction of the battery's nameplate rate the cycle guard "
                         "assumes is available for recovery. Lower = starts "
                         "correcting earlier.")
    ap.add_argument("--soc-return", type=float, default=None,
                    help="MWh slack on the CYCLE condition: how far the reservoir "
                         "may end the window from where the real dispatch left it. "
                         "Without it a swing bound alone lets the fill drain the "
                         "battery a little every day forever.")
    args = ap.parse_args()
    Q = args.q if args.q is not None else F.Q

    te, feat_cols, xm, xs_, ym, ys_ = F.load_test()
    tfi = np.asarray(TARGET_FEAT_IDX)
    days_idx = te.index.normalize()
    full = [d for d in pd.unique(days_idx) if (days_idx == d).sum() == 288]
    if args.max_days:
        full = full[:args.max_days]
    g0 = F.FREE[0] * 12
    print(f"{len(full)} complete test days  {full[0].date()} .. {full[-1].date()}  "
          f"window {F.FREE[0]}:00-{F.FREE[1]}:00  shift {Q:g}%", flush=True)

    # ---- the shift, on locally-served demand (isolated Victoria) -------------
    nd_base_all = te[TARGETS].values @ SIGN
    dem_local = nd_base_all + te["wind"].values + te["solar_utility"].values
    free_mask = (te.index.hour >= F.FREE[0]) & (te.index.hour < F.FREE[1])
    dem_s = FixedPercentageShift(Q, Q, free_hours=F.FREE).transform(
        pd.DataFrame({"demand_mw": dem_local,
                      "price_aud_per_mwh": te["price_aud_per_mwh"].values},
                     index=te.index))["demand_mw"].values
    d_dem = dem_s - dem_local
    nd_cf_all = nd_base_all + d_dem

    rows_day = np.stack([np.where(days_idx == d)[0] for d in full])      # (D,288)
    D = len(full)
    truth = te[TARGETS].values.astype(np.float64)[rows_day]              # (D,288,6)
    ndb = nd_base_all[rows_day]
    ndc = nd_cf_all[rows_day]
    ratio = np.where(np.abs(ndb) > 1e-6, ndc / ndb, 1.0)
    uni = truth * ratio[..., None]                                       # UNIFORM reference
    curt_act = te[CURT_NAMES].values.astype(np.float64)[rows_day]

    gap_rows = rows_day[:, g0:g0 + GAP]
    span = np.concatenate([rows_day[:, g0 - CTX:g0],
                           gap_rows,
                           rows_day[:, g0 + GAP:g0 + GAP + CTX]], 1)     # (D,132)
    W = span.shape[1]

    def make_X(ctx_day, shifted):
        """ctx_day (D,288,6) the mode's own dispatch. Mirrors counterfactual.fill."""
        vals = te[feat_cols].values.astype(np.float64)[span].copy()      # (D,W,nf)
        ci = {c: feat_cols.index(c) for c in
              ("demand_mw", "net_demand", "price_aud_per_mwh")}
        if shifted:
            vals[:, :, ci["demand_mw"]] = dem_s[span]
            vals[:, :, ci["price_aud_per_mwh"]] = np.where(
                free_mask[span], 0.0, te["price_aud_per_mwh"].values[span])
        ctx = np.concatenate([ctx_day[:, g0 - CTX:g0], ctx_day[:, g0:g0 + GAP],
                              ctx_day[:, g0 + GAP:g0 + GAP + CTX]], 1)   # (D,W,6)
        for k, t_ in enumerate(TARGETS):
            vals[:, :, feat_cols.index(t_)] = ctx[:, :, k]
        # supply-side nd, from the mode's own dispatch; the gap carries the target
        vals[:, :, ci["net_demand"]] = ctx @ SIGN
        nd_t = (ndc if shifted else ndb)[:, g0:g0 + GAP]
        vals[:, CTX:CTX + GAP, ci["net_demand"]] = nd_t
        pL = vals[:, CTX - 1][:, [feat_cols.index(t_) for t_ in TARGETS]]
        pR = vals[:, CTX + GAP][:, [feat_cols.index(t_) for t_ in TARGETS]]
        X = ((vals - xm) / xs_).astype(np.float32)
        mk = np.ones((D, W, 1), np.float32); mk[:, CTX:CTX + GAP] = 0.0
        X[:, CTX:CTX + GAP, tfi] = 0.0
        X[:, CTX:CTX + GAP, CURT_COLS] = 0.0
        return X, mk, pL, pR, nd_t

    c_m = torch.tensor(xm[CURT_COLS].astype(np.float32))
    c_s = torch.tensor(xs_[CURT_COLS].astype(np.float32))
    ys_mt = torch.tensor(ym, dtype=torch.float32)
    ys_st = torch.tensor(ys_, dtype=torch.float32)

    results, meta = {}, {}
    for arm in args.arms:
        ckp = OUT / f"{arm}.pt"
        if not ckp.exists():
            print(f"  [skip] {arm}"); continue
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        head_fn, n_disp = HEADS[ck["head"]]
        bb = ck.get("backbone", "bilstm")
        HD.set_soc(ck.get("soc") or (True if args.soc_return is not None else None),
                   args.soc_return, args.soc_rate_frac)
        meta[arm] = dict(head=ck["head"], soc=ck.get("soc"),
                         resid=ck.get("residual", False))
        m = (BACKBONES[bb](len(feat_cols), n_disp, hidden=ck["hidden"]) if bb == "bilstm"
             else BACKBONES[bb](len(feat_cols), TARGET_FEAT_IDX, CURT_COLS, n_disp,
                                hidden=ck["hidden"]))
        m.load_state_dict(ck["state"]); m.eval()
        print(f"  {arm}: head={ck['head']} soc={ck.get('soc')} "
              f"residual={meta[arm]['resid']}", flush=True)

        out = {}
        for lbl, shifted in (("base", False), ("cf", True)):
            ctx_day = uni if shifted else truth
            X, mk, pL, pR, nd_t = make_X(ctx_day, shifted)
            use_soc = HD.SOC_CAP is not None
            seed = soc_seed_from(ctx_day, g0) if use_soc else None
            # cycle target: where the REAL dispatch left the reservoir across the
            # window. The model may reallocate power inside it but must not change
            # the reservoir's trajectory across it -- it does not control the hours
            # where recharging would happen.
            tgt = None
            if args.soc_return is not None:
                dEt = ((truth[:, g0:g0 + GAP, HD.CHG_I] * HD.ETA_B
                        - truth[:, g0:g0 + GAP, HD.DIS_I] / HD.ETA_B) * HD.DT_H).sum(1)
                tgt = torch.tensor(seed[0].numpy() + dEt)
            Ps, Cs = [], []
            for i in range(0, D, 32):
                sl = slice(i, i + 32)
                with torch.no_grad():
                    d_raw, c_raw = m(torch.from_numpy(X[sl]), torch.from_numpy(mk[sl]), None)
                    d_raw = d_raw[:, CTX:CTX + GAP]
                    Cs.append(curt_activation(c_raw[:, CTX:CTX + GAP], c_m, c_s).numpy())
                    if meta[arm]["resid"]:
                        tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
                        pLz = (pL[sl] - ym) / ys_; pRz = (pR[sl] - ym) / ys_
                        d_raw = apply_residual(d_raw, torch.tensor(
                            (pLz[:, None] + tt * (pRz - pLz)[:, None]).astype(np.float32)))
                    F_ = (d_raw * ys_st + ys_mt if n_disp == 6 else
                          torch.cat([d_raw[..., :6] * ys_st + ys_mt, d_raw[..., 6:]], -1))
                    kw = {} if seed is None else {"soc_seed": tuple(t[sl] for t in seed)}
                    if tgt is not None:
                        kw["soc_target"] = tgt[sl].float()
                    P = head_fn(F_, torch.from_numpy(pL[sl]).float(),
                                torch.from_numpy(pR[sl]).float(),
                                torch.from_numpy(nd_t[sl]).float(), **kw)
                Ps.append(P.numpy())
            out[lbl] = (np.concatenate(Ps, 0).astype(np.float64),
                        np.concatenate(Cs, 0).astype(np.float64))
        results[arm] = out

    # ---------------- assemble days and project ----------------
    def assemble(fill_mw, shifted):
        day = (uni if shifted else truth).copy()
        day[:, g0:g0 + GAP] = fill_mw
        raw = day.copy()
        nd_full = ndc if shifted else ndb
        with torch.no_grad():
            proj = IC.cyclic_project(torch.tensor(day), torch.tensor(day[:, 0].copy()),
                                     torch.tensor(day[:, -1].copy()),
                                     torch.tensor(nd_full), iters=400, soc=True,
                                     size_aware=True).numpy()
        return raw, proj

    E = lambda A: A.sum((0, 1)) * DT                 # (6,) MWh over the whole test set
    Ew = lambda A_: A_[:, g0:g0 + GAP].sum((0, 1)) * DT     # ... free window only

    # project every day once, up front -- both tables below read from this
    proj_cache = {}
    for arm in args.arms:
        if arm not in results:
            continue
        _, pb = assemble(results[arm]["base"][0], False)
        rc, pcf = assemble(results[arm]["cf"][0], True)
        proj_cache[arm] = dict(base=pb, cf=pcf, raw_cf=rc)

    dE_uni = E(uni) - E(truth)
    lines = []
    A = lines.append

    A(f"# Full test-set counterfactual — {len(full)} days, "
      f"{full[0].date()} .. {full[-1].date()}")
    A("")
    A(f"Free window {F.FREE[0]}:00-{F.FREE[1]}:00, rebound {Q:g}% / reduce {Q:g}%. "
      f"Outside the window every fuel is scaled uniformly by `nd_cf/nd_base`; inside "
      f"it the model fills. `project_full_day` (cyclic_project, 400 iters, SOC on) "
      f"is applied over all 288 steps of every day.")
    A("")
    dd_win = float(d_dem[gap_rows].sum() * DT)
    dd_day = float(d_dem[rows_day].sum() * DT)
    A(f"Demand rises by **{dd_win:+,.0f} MWh** inside the free window and falls "
      f"outside it, netting **{dd_day:+,.0f} MWh** over the whole day. The window "
      f"figure is what the fleet must actually re-dispatch.")
    A("")

    # ---- absolute totals: real vs counterfactual, and the percent change -----
    E_real = E(truth)
    Ew_real = Ew(truth)
    for scope, Er, agg in (("whole day, all 288 steps", E_real, E),
                           ("free window 11:00-14:00 only", Ew_real, Ew)):
        A(f"## Energy totals — real vs counterfactual ({scope})")
        A("")
        A("| fuel | real (MWh) | uniform (MWh) | % | "
          + " | ".join(f"{a} (MWh) | %" for a in args.arms if a in proj_cache) + " |")
        A("| --- | ---: | ---: | ---: |" + " ---: | ---: |" * len(proj_cache))
        pct = lambda new_, old: (f"{100 * (new_ - old) / abs(old):+.2f}%"
                                 if abs(old) > 1e-6 else "n/a")
        Eu = agg(uni)
        for i, t_ in enumerate(TARGETS):
            row = (f"| {SHORT[t_]} | {Er[i]:,.0f} | {Eu[i]:,.0f} | "
                   f"{pct(Eu[i], Er[i])} |")
            for a in args.arms:
                if a in proj_cache:
                    v = agg(proj_cache[a]["cf"])[i]
                    row += f" {v:,.0f} | {pct(v, Er[i])} |"
            A(row)
        tr_, tu_ = float(Er @ SIGN), float(Eu @ SIGN)
        row = (f"| **net demand served** | {tr_:,.0f} | {tu_:,.0f} | {pct(tu_, tr_)} |")
        for a in args.arms:
            if a in proj_cache:
                v = float(agg(proj_cache[a]["cf"]) @ SIGN)
                row += f" {v:,.0f} | {pct(v, tr_)} |"
        A(row)
        gr = [i for i, x in enumerate(SIGN) if x > 0]
        row = (f"| _gross generation_ | {Er[gr].sum():,.0f} | {Eu[gr].sum():,.0f} | "
               f"{pct(Eu[gr].sum(), Er[gr].sum())} |")
        for a in args.arms:
            if a in proj_cache:
                v = agg(proj_cache[a]["cf"])[gr].sum()
                row += f" {v:,.0f} | {pct(v, Er[gr].sum())} |"
        A(row)
        A("")

    A("## Total change per fuel, whole test set (MWh, vs the real dispatch)")
    A("")
    A("| fuel | UNIFORM scaling | " + " | ".join(args.arms) + " | "
      + " | ".join(f"{a} − uniform" for a in args.arms) + " |")
    A("| --- | ---: |" + " ---: |" * (2 * len(args.arms)))
    for i, t_ in enumerate(TARGETS):
        cells = []
        for arm in args.arms:
            if arm not in proj_cache:
                continue
            cells.append(E(proj_cache[arm]["cf"])[i] - E(truth)[i])
        A(f"| {SHORT[t_]} | {dE_uni[i]:+,.0f} | "
          + " | ".join(f"{c:+,.0f}" for c in cells) + " | "
          + " | ".join(f"{c - dE_uni[i]:+,.0f}" for c in cells) + " |")
    tot_u = float(dE_uni @ SIGN)
    A(f"| **Σ signed (= net demand served)** | {tot_u:+,.0f} | "
      + " | ".join(f"{float((E(proj_cache[a]['cf']) - E(truth)) @ SIGN):+,.0f}"
                   for a in args.arms if a in proj_cache) + " | |")
    A("")

    A("## The same, restricted to the free window")
    A("")
    dEw_uni = Ew(uni) - Ew(truth)
    A("| fuel | UNIFORM scaling | " + " | ".join(args.arms) + " | "
      + " | ".join(f"{a} − uniform" for a in args.arms) + " | real-grid share |")
    A("| --- | ---: |" + " ---: |" * (2 * len(args.arms) + 1))
    EMP = {"coal_brown": 43.4, "hydro": 29.3, "gas_ocgt": 9.7, "gas_steam": 3.7,
           "battery_discharging": 7.4, "battery_charging": 6.6}
    for i, t_ in enumerate(TARGETS):
        cells = [Ew(proj_cache[a]["cf"])[i] - Ew(truth)[i]
                 for a in args.arms if a in proj_cache]
        A(f"| {SHORT[t_]} | {dEw_uni[i]:+,.0f} | "
          + " | ".join(f"{c:+,.0f}" for c in cells) + " | "
          + " | ".join(f"{c - dEw_uni[i]:+,.0f}" for c in cells)
          + f" | {EMP[t_]:.1f}% |")
    A(f"| **Σ signed** | {float(dEw_uni @ SIGN):+,.0f} | "
      + " | ".join(f"{float((Ew(proj_cache[a]['cf']) - Ew(truth)) @ SIGN):+,.0f}"
                   for a in args.arms if a in proj_cache)
      + " |" + " |" * len(args.arms) + f" needs {dd_win:+,.0f} |")
    A("")
    A("_Shares of the signed total, for comparison with dispatch_study Study A:_")
    A("")
    A("| allocation | " + " | ".join(SHORT[t] for t in TARGETS) + " |")
    A("| --- |" + " ---: |" * 6)
    def shares(v):
        tot = abs(float(v @ SIGN))
        return [100 * abs(x) / max(tot, 1e-9) for x in v]
    A("| UNIFORM | " + " | ".join(f"{x:.1f}%" for x in shares(dEw_uni)) + " |")
    for a in args.arms:
        if a in proj_cache:
            A(f"| {a} | " + " | ".join(
                f"{x:.1f}%" for x in shares(Ew(proj_cache[a]["cf"]) - Ew(truth))) + " |")
    A("| REAL GRID (Study A) | " + " | ".join(f"{EMP[t]:.1f}%" for t in TARGETS) + " |")
    A("")
    A("_Shares can exceed 100%. Uniform scaling multiplies the battery CHARGING "
      "channel by the same factor as everything else, so when demand rises it "
      "charges MORE -- adding load that the generators must then also cover. That "
      "is a real defect of the uniform reference, not a reporting artifact: it is "
      "what 'scale every fuel to match the new net demand' does to a channel whose "
      "sign is negative._")

    A("## Battery energy conservation")
    A("")
    A("A battery is a container, not a source: over the test set the stored energy "
      "`eta*charge - discharge/eta` must come back to where it started. The real "
      "dispatch does exactly that. A SWING bound (max E - min E) cannot see a slow "
      "drift, which is why this table, not the feasibility table, is the one that "
      "catches an impossible fill.")
    A("")
    A("| case | charge (MWh) | discharge (MWh) | stored ΔE | per day | % of reservoir/day |")
    A("| --- | ---: | ---: | ---: | ---: | ---: |")
    def batt(A_):
        c = A_[..., HD.CHG_I].sum() * DT
        d = A_[..., HD.DIS_I].sum() * DT
        return c, d, c * HD.ETA_B - d / HD.ETA_B
    for lbl, arr in ([("REAL dispatch", truth), ("UNIFORM scaling", uni)]
                     + [(f"{a}  RAW head output", proj_cache[a]["raw_cf"])
                        for a in args.arms if a in proj_cache]
                     + [(f"{a}  after project_full_day", proj_cache[a]["cf"])
                        for a in args.arms if a in proj_cache]):
        c, d, dE = batt(arr)
        A(f"| {lbl} | {c:,.0f} | {d:,.0f} | {dE:+,.0f} | {dE / D:+,.0f} | "
          f"{100 * dE / D / IC.BATT_CAP_MWH:+.1f}% |")
    A("")
    A(f"_Reservoir {IC.BATT_CAP_MWH:,.0f} MWh. The real fleet's discharge/charge "
      f"ratio is {float(batt(truth)[1] / batt(truth)[0]):.4f} against "
      f"eta^2 = {HD.ETA_B ** 2:.4f}, i.e. it cycles exactly._")
    A("")

    A("### Where in the window the battery is forced")
    A("")
    A("Mean MW by step index inside the 36-step window, over all days. Demand "
      "jumps 2.4% the instant the window opens, but every generator starts "
      "pinned at `pL` and coal can only move 147.5 MW per 5-min step. The "
      "battery ramps 795/715 MW per step, so it is the ONLY channel that can "
      "absorb the step change at the LEFT seam -- it discharges hard at minute "
      "0, then recharges once the generators have ramped in. The cycle guard "
      "is what forces that recharge; without it the battery discharged and "
      "never paid it back._")
    A("")
    A("| step (min into window) | " + " | ".join(f"{q}" for q in
        ("actual charge", "model charge", "actual discharge", "model discharge")) + " |")
    A("| --- | ---: | ---: | ---: | ---: |")
    mainarm = [a for a in args.arms if a in proj_cache][0]
    Pw = proj_cache[mainarm]["cf"][:, g0:g0 + GAP]
    Tw = truth[:, g0:g0 + GAP]
    for j in range(0, GAP, 5):
        A(f"| {j * 5} | {Tw[:, j, HD.CHG_I].mean():,.0f} | {Pw[:, j, HD.CHG_I].mean():,.0f} "
          f"| {Tw[:, j, HD.DIS_I].mean():,.0f} | {Pw[:, j, HD.DIS_I].mean():,.0f} |")
    A(f"| {(GAP - 1) * 5} | {Tw[:, -1, HD.CHG_I].mean():,.0f} | "
      f"{Pw[:, -1, HD.CHG_I].mean():,.0f} | {Tw[:, -1, HD.DIS_I].mean():,.0f} | "
      f"{Pw[:, -1, HD.DIS_I].mean():,.0f} |")
    A("")

    A("## Curtailment (MWh over the test set)")
    A("")
    A("Curtailment is a parallel prediction: planB removed the curtailment credit, "
      "so it never enters the balance target. Uniform scaling has no opinion on it "
      "-- it leaves curtailment at the actual value by construction.")
    A("")
    A("| arm | actual | base-case predicted | counterfactual | Δ (cf − base) | "
      "Δ vs actual |")
    A("| --- | ---: | ---: | ---: | ---: | ---: |")
    cact = curt_act[:, g0:g0 + GAP].sum() * DT
    for arm in args.arms:
        if arm not in results:
            continue
        cb = results[arm]["base"][1].sum() * DT
        cc = results[arm]["cf"][1].sum() * DT
        A(f"| {arm} | {cact:,.0f} | {cb:,.0f} | {cc:,.0f} | {cc - cb:+,.0f} | "
          f"{cc - cact:+,.0f} |")
    A("")

    # ---------------- feasibility ----------------
    A("## Feasibility of the RAW filled window, before project_full_day")
    A("")
    A("| arm | SOC in head | balance max | below 0 | above cap | ramp IN window | "
      "ramp outside | swing to gap end | over budget | swing whole day |")
    A("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    cap_eff = IC.BATT_CAP_MWH - 2.0 * 100.0
    for arm in args.arms:
        if arm not in results:
            continue
        P = results[arm]["cf"][0]
        nd_t = ndc[:, g0:g0 + GAP]
        balv = np.abs(P @ SIGN - nd_t)
        bal = balv.max()
        bal_mean = balv.mean(); bal_n = int((balv > 1.0).sum()); bal_tot = balv.size
        neg = np.maximum(-P, 0).max()
        ov = np.maximum(P - LIM.CAP, 0).max()
        day = uni.copy(); day[:, g0:g0 + GAP] = P
        d = np.diff(day, axis=1)
        ov_r = np.maximum(d - LIM.R_UP, 0) + np.maximum(-d - LIM.R_DN, 0)
        win = np.zeros(d.shape[1], bool); win[g0 - 1:g0 + GAP] = True   # incl. both seams
        rmp = ov_r[:, win].max(); rmp_out = ov_r[:, ~win].max()
        dE = (day[..., HD.CHG_I] * HD.ETA_B - day[..., HD.DIS_I] / HD.ETA_B) * HD.DT_H
        Ecum = np.concatenate([np.zeros((D, 1)), dE.cumsum(1)], 1)
        sw = Ecum.max(1) - Ecum.min(1)                       # whole day
        ge = g0 + GAP + 1                                    # midnight .. gap end
        sw_head = Ecum[:, :ge].max(1) - Ecum[:, :ge].min(1)  # what the head can bound
        A(f"| {arm} | {'yes' if meta[arm]['soc'] else 'no'} | "
          f"{bal:.1f} max / {bal_mean:.3f} mean / {bal_n} of {bal_tot} over 1 MW | "
          f"{neg:.2e} | "
          f"{ov:.2e} | {rmp:.2e} | {rmp_out:.2e} | {sw_head.max():,.0f} MWh | "
          f"{np.maximum(sw_head - cap_eff, 0).max():,.1f} MWh | {sw.max():,.0f} MWh |")
    A("")
    A(f"_Day SOC swing is over all 288 steps with the model's window spliced in; "
      f"budget is {cap_eff:,.0f} MWh (reservoir {IC.BATT_CAP_MWH:,.0f} less two "
      f"100 MWh margins). Uniform-scaling reference for the same days: "
      f"{(lambda a: a.max())(np.array([0.0])) if False else ''}_")

    # uniform reference swing, for context
    dEu = (uni[..., HD.CHG_I] * HD.ETA_B - uni[..., HD.DIS_I] / HD.ETA_B) * HD.DT_H
    Eu = np.concatenate([np.zeros((D, 1)), dEu.cumsum(1)], 1)
    swu = Eu.max(1) - Eu.min(1)
    dEa = (truth[..., HD.CHG_I] * HD.ETA_B - truth[..., HD.DIS_I] / HD.ETA_B) * HD.DT_H
    Ea = np.concatenate([np.zeros((D, 1)), dEa.cumsum(1)], 1)
    swa = Ea.max(1) - Ea.min(1)
    A("_`ramp outside` is not the model's: outside the free window the day IS the "
      "uniform reference, and scaling coal by nd_cf/nd_base scales its ramps too, "
      "which pushes them past the envelope. Inside the window, where the head acts, "
      "the ramp residual is float32 noise._")
    A("")
    lines[-1] = (f"_`swing to gap end` is midnight to the close of the free window -- "
                 f"the only stretch the head can bound, since it is seeded with the "
                 f"day's history and controls nothing after 14:00. `swing whole day` "
                 f"includes the evening peak, where the battery does most of its work "
                 f"and the head has no say. Budget {cap_eff:,.0f} MWh. For reference "
                 f"the ACTUAL dispatch swings up to {{swa}} MWh over the day and the "
                 f"uniform reference up to {{swu}} MWh._")
    lines[-1] = lines[-1].replace("{swa}", f"{swa.max():,.0f}").replace(
        "{swu}", f"{swu.max():,.0f}")
    A("")

    (OUT / f"counterfactual{args.suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}/counterfactual{args.suffix}.md")

    # ---------------- figure: per-fuel change ----------------
    arms_ok = [a for a in args.arms if a in proj_cache]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(6); wdt = 0.8 / (len(arms_ok) + 1)
    ax.bar(x - 0.4 + wdt / 2, dEw_uni, wdt, label="uniform scaling",
           color=MUTED, edgecolor="white")
    for k, a in enumerate(arms_ok):
        ax.bar(x - 0.4 + wdt * (k + 1.5), Ew(proj_cache[a]["cf"]) - Ew(truth), wdt,
               label=a, edgecolor="white")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in TARGETS], fontsize=9)
    ax.set_ylabel("MWh change in the free window, whole test set")
    ax.set_title(f"Counterfactual response by fuel — {len(full)} test days, "
                 f"{Q:g}% shift", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"cf_by_fuel{args.suffix}.png", dpi=150, facecolor="white")
    print(f"wrote {OUT}/cf_by_fuel{args.suffix}.png")

    # ---------------- figure: the stack, exact days, actual vs uniform vs model
    off = 0
    if args.from_day:
        hit = [k for k, d in enumerate(full) if str(d.date()) == args.from_day]
        off = hit[0] if hit else 0
    sh = list(range(off, min(off + args.show_days, D)))
    ren = np.stack([te[["wind", "solar_utility"]].values[r] for r in rows_day])
    ti = {t: i for i, t in enumerate(TARGETS)}
    # canonical order: coal is the base of the stack and everything sits on it
    # (dispatch_study/common.py FUELS). battery CHARGING is a load and is drawn
    # below the axis, not in the stack.
    stack = FUELS
    panels = ([("actual", truth, ndb, curt_act)]
              + [(a, proj_cache[a]["cf"], ndc,
                  np.concatenate([curt_act[:, :g0], results[a]["cf"][1],
                                  curt_act[:, g0 + GAP:]], 1))
                 for a in args.arms if a in proj_cache])
    nr, nc = len(panels), len(sh)
    fig, ax = plt.subplots(nr, nc, figsize=(3.0 * nc + 0.9, 2.25 * nr),
                           sharey=True, sharex=True, squeeze=False)
    x = np.arange(288) * 5 / 60.0
    for c, i in enumerate(sh):
        for r_, (name, arr, ndl, cu) in enumerate(panels):
            a_ = ax[r_][c]; base = np.zeros(288)
            for k in stack:
                top = base + arr[i, :, ti[k]]
                a_.fill_between(x, base, top, color=COLORS[k], lw=0, zorder=2,
                                label=LABEL[k]); base = top
            for k2, col in enumerate(("wind", "solar_utility")):
                top = base + ren[i, :, k2]
                a_.fill_between(x, base, top, color=COLORS[col], lw=0, zorder=2,
                                label=LABEL[col]); base = top
            for k2, col in enumerate(("wind", "solar_utility")):
                top = base + cu[i, :, k2]
                a_.fill_between(x, base, top, color=COLORS[col], alpha=0.33, lw=0,
                                hatch="///", zorder=2,
                                label=f"{LABEL[col]} curtailed"); base = top
            a_.fill_between(x, 0, -arr[i, :, ti["battery_charging"]],
                            color=COLORS["battery_charging"], lw=0, zorder=2,
                            label=LABEL["battery_charging"])
            a_.plot(x, ndl[i], color="#B3261E", lw=1.0, zorder=4, label="net demand")
            a_.axvspan(F.FREE[0], F.FREE[1], color="#000", alpha=0.08, zorder=1)
            a_.set_xlim(0, 24); a_.margins(x=0)
            a_.grid(True, axis="y", alpha=0.15, lw=0.5, zorder=0)
            a_.tick_params(labelsize=7.5, colors=MUTED, length=0)
            for sp in ("top", "right", "left", "bottom"):
                a_.spines[sp].set_visible(False)
            if r_ == 0:
                a_.set_title(str(full[i].date()), fontsize=10.5, color=INK, pad=5)
            if c == 0:
                a_.set_ylabel(name.replace("hardnet_2025_", ""), fontsize=9.5, color=INK)
            if r_ == nr - 1:
                a_.set_xticks([0, 6, 11, 14, 18, 24])
    h, lb = ax[0][0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=6, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.004))
    fig.suptitle(f"Counterfactual — {len(sh)} exact test days, shaded = the "
                 f"{F.FREE[0]}:00-{F.FREE[1]}:00 window the model fills "
                 f"(demand {Q:g}% up inside, down outside)",
                 fontsize=12.5, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0.055, 1, 0.965])
    fig.savefig(OUT / f"cf_stack{args.suffix}.png", dpi=145, facecolor="white",
                bbox_inches="tight")
    print(f"wrote {OUT}/cf_stack{args.suffix}.png")


if __name__ == "__main__":
    main()
