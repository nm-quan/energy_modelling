"""Counterfactual + placebo evaluation of the trained bi-LSTM gap imputer.

Reconstruction WAPE (train.py) proves the imputer fills real gaps well. This adds
the two counterfactual checks (lit_review.md theme 8 validation toolbox):

  placebo    feed the gap UNCHANGED (g=0). The response vs the base fill must be
             ~0 -- the pipeline must not invent a demand effect where there is none.
  scenario   feed +g% demand into the gap's known subspace (net_demand, demand),
             re-impute, project against the RAISED net_demand. Measure whether the
             fleet delivers the extra load (capture), how the mix responds, and
             whether it stays feasible incl. BOTH seams -- the whole point.

Boundaries p_L, p_R are the real measured values (energy-neutral assumption; the
honest caveat for a pure increase is the right-boundary landing strip, README).

    python3 imputation/scenario_eval.py --g 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap_data import (load_flats, test_gap_windows, TARGETS, SIGN,             # noqa: E402
                      TARGET_FEAT_IDX, ND_COL, DEM_COL)
import constraints as C                                                       # noqa: E402
from model import BiLSTMImputer                                               # noqa: E402

OUT = HERE / "results"
BATT_CAP_MWH = 4735.75
ETA = np.sqrt(0.834)
DT = 5 / 60


def impute_project(model, f, gw, context, device, g_pct):
    """Return projected fill (N,6) MW for a +g_pct demand scenario (g=0 => base)."""
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
    P, resid = C.project_gap(fill, gw.pL_mw, gw.pR_mw, nd_mw)
    return P, resid, nd_mw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=float, default=10.0)
    ap.add_argument("--context", type=int, default=48)   # match the trained checkpoint
    ap.add_argument("--ckpt", default=str(OUT / "bilstm_imputer.pt"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    f = load_flats()
    gws = test_gap_windows(f, context=args.context)
    model = BiLSTMImputer().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    for tag, g in (("placebo(g=0)", 0.0), (f"scenario(g={args.g:g})", args.g)):
        cap_num = cap_den = 0.0
        resp = np.zeros(6); base_sum = np.zeros(6)
        track = []; ramp_bad = neg = 0; bal_max = 0.0; soc_bad = 0
        for gw in gws:
            Pb, _, ndb = impute_project(model, f, gw, args.context, device, 0.0)
            Ps, resid, nds = impute_project(model, f, gw, args.context, device, g)
            cap_num += (Ps @ SIGN).sum() - (Pb @ SIGN).sum()
            cap_den += (nds - ndb).sum()
            resp += Ps.sum(0); base_sum += Pb.sum(0)
            track.append(np.abs(Ps @ SIGN - nds))
            d = np.diff(np.vstack([gw.pL_mw, Ps, gw.pR_mw]), axis=0)
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
