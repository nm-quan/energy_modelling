"""Constraint check over the ENTIRE test set -- not just each gap + its two seams.

`scenario_eval.py` checks every 11:00-14:00 fill and its two seams, aggregated over
the 186 gap-days. This instead STITCHES the imputed gaps back into the real measured
dispatch and checks constraints CONTINUOUSLY across the whole test span (~53k 5-min
steps), so nothing between/around the gaps is skipped.

    python3 imputation/full_series_check.py                                  # actual (no model)
    python3 imputation/full_series_check.py --ckpt results/bilstm_posthoc.pt --cmode posthoc --g 10

What this adds over the per-gap check, honestly:
  * ramp / balance / n_neg over EVERY consecutive step of the whole series. Outside
    the gaps this is REAL measured data (feasible by construction), and the only
    model-made junctions are the gap seams (already covered by scenario_eval) -- so
    for a well-behaved fill these should come back ~identical. The value is that it
    PROVES it over the whole series instead of asserting it.
  * SOC as a continuous PER-DAY trajectory. THIS is the real add: raising demand in
    the gap changes battery charge/discharge, which shifts the state of charge for
    the REST of that day -- an effect the per-gap SOC reset cannot see. Checked
    per-day (not one 186-day cumsum: that drifts into the ~2000%-of-pack artifact;
    the fleet cycles daily, so daily swing is the physical metric).
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
from gap_data import load_flats, test_gap_windows, TARGETS, SIGN, ND_COL     # noqa: E402
import constraints as C                                                     # noqa: E402
import scenario_eval as SE                                                  # noqa: E402
from scenario_eval import impute_project                                    # noqa: E402
from model import BiLSTMImputer                                             # noqa: E402

OUT = HERE / "results"
TOL = 0.6   # MW ramp tolerance, matched to scenario_eval


def check(P_mw: np.ndarray, nd_mw: np.ndarray, index) -> dict:
    """P_mw (T,6), nd_mw (T,), index (T,) datetimes -- the continuous test series."""
    # continuity guard: spurious ramps if the series has time gaps
    step = np.diff(index.view("i8")) / 1e9 / 60.0            # minutes between rows
    n_time_gaps = int((np.abs(step - 5.0) > 0.1).sum())

    d = np.diff(P_mw, axis=0)                                # every consecutive pair
    over = np.maximum(d - C.R_UP, 0.0) + np.maximum(-d - C.R_DN, 0.0)   # MW past the tube
    bad = over > TOL
    ramp_bad = int(bad.sum())
    ramp_max = float(over.max())
    # attribute violations at midnight day-boundaries separately: the shift scenario's
    # per-day Option-A (free-endpoint) projections are independent across days, so a
    # cross-day seam violation is a known artifact of that independence, not of the fill
    mid = (index.hour == 0) & (index.minute == 0)
    ramp_bad_midnight = int(bad[mid[1:]].sum())

    neg = int((P_mw < -0.1).sum())
    bal = np.abs((P_mw * SIGN).sum(1) - nd_mw)              # |SIGN.P - net_demand| per step
    bal_max = float(bal.max()); bal_p95 = float(np.percentile(bal, 95))

    # SOC per calendar day
    day = index.normalize()
    swings = []
    for d0 in np.unique(day):
        swings.append(C._soc_swing_mwh(P_mw[np.asarray(day == d0)]))
    swings = np.array(swings)
    return {"n_steps": int(len(P_mw)), "n_days": int(len(swings)),
            "n_time_gaps": n_time_gaps,
            "ramp_violations": ramp_bad, "ramp_max_over_mw": ramp_max,
            "ramp_violations_at_midnight_seams": ramp_bad_midnight,
            "n_neg": neg, "balance_resid_max_mw": bal_max, "balance_resid_p95_mw": bal_p95,
            "soc_worst_day_swing_mwh": float(swings.max()),
            "soc_worst_day_pct_of_pack": float(100 * swings.max() / C.BATT_CAP_MWH),
            "soc_infeasible_days": int((swings > C.BATT_CAP_MWH + 1e-6).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["actual", "increase", "shift"], default="actual",
                    help="actual: the real series (feasibility bar, no ckpt needed). "
                         "increase: legacy +g%%-in-gap overlay (needs ckpt). "
                         "shift: the corrected FixedPercentageShift counterfactual -- whole "
                         "period = projected feasible reference + imputed free-window fills; "
                         "without --ckpt it checks the reference series alone.")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cmode", choices=["posthoc", "unrolled", "rayen_traj"], default="posthoc",
                    help="constraint mode the checkpoint was trained with")
    ap.add_argument("--g", type=float, default=None,
                    help="scenario magnitude %% (default: 10 increase, 3.699 shift)")
    ap.add_argument("--context", type=int, default=48)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.g is None:
        args.g = 10.0 if args.scenario == "increase" else SE.SHIFT_Q
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    f = load_flats()
    lb = f.Xte.shape[0] - len(f.test_index)                 # lookback rows prepended (=lb_carry)
    P = f.y_to_mw(f.Yte).astype(float)                      # (T,6) REAL dispatch, MW
    nd = f.col_mw(f.Xte, ND_COL).astype(float)              # (T,)  REAL net_demand, MW

    model = None
    if args.ckpt:
        model = BiLSTMImputer().to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    tag = "actual"
    if args.scenario == "increase":
        if model is None:
            raise SystemExit("--scenario increase needs --ckpt")
        gws = test_gap_windows(f, context=args.context)
        for gw in gws:                                      # overlay the +g% imputed gaps
            Ps, _, nds = impute_project(model, f, gw, args.context, device, args.g, mode=args.cmode)
            P[gw.gap_idx] = Ps
            nd[gw.gap_idx] = nds
        tag = f"counterfactual_g{args.g:g}_{args.cmode}"
    elif args.scenario == "shift":
        # the whole period is counterfactual: off-window = the per-day Option-A
        # projected reference (a feasible dispatch consistent with the shifted nd);
        # each 11:00-14:00 free window = the model's imputed fill (if a ckpt is given)
        print(f"building shift scenario q={args.g:g}% ...", flush=True)
        scen = SE.build_shift_scenario(f, args.g)
        _, nd_s, P_ref = scen
        P, nd = P_ref.copy(), nd_s.copy()
        tag = f"shift_reference_q{args.g:g}"
        if model is not None:
            gws = test_gap_windows(f, context=args.context)
            for gw in gws:
                Ps, _, nds = SE.impute_project_shift(model, f, gw, args.context, device,
                                                     scen, mode=args.cmode)
                P[gw.gap_idx] = Ps
                nd[gw.gap_idx] = nds
            tag = f"shift_q{args.g:g}_{args.cmode}"

    # analyse only the genuine test span (drop the lb_carry lookback carry rows)
    r = check(P[lb:], nd[lb:], f.test_index)
    r["tag"] = tag
    OUT.mkdir(exist_ok=True)
    (OUT / f"full_series_{tag}.json").write_text(json.dumps(r, indent=2))
    warn = "" if r["n_time_gaps"] == 0 else f"  (WARNING: {r['n_time_gaps']} time gaps in series)"
    print(f"[{tag}]  {r['n_steps']} steps over {r['n_days']} days{warn}")
    print(f"  ramp violations (whole series, >{TOL} MW): {r['ramp_violations']}  "
          f"(of which at midnight day-seams: {r['ramp_violations_at_midnight_seams']})   "
          f"max over tube {r['ramp_max_over_mw']:.2e} MW")
    print(f"  n_neg: {r['n_neg']}   balance resid: max {r['balance_resid_max_mw']:.2e} MW  "
          f"p95 {r['balance_resid_p95_mw']:.2e} MW")
    print(f"  SOC per-day: worst {r['soc_worst_day_swing_mwh']:.0f} MWh "
          f"({r['soc_worst_day_pct_of_pack']:.1f}% of pack), infeasible days "
          f"{r['soc_infeasible_days']}/{r['n_days']}")
    print("wrote", OUT / f"full_series_{tag}.json")


if __name__ == "__main__":
    main()
