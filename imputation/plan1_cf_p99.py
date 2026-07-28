"""4-day counterfactual (scaled + masked) for the size_aware model with the
coal_brown ramp limit set to the p99 of the historical 5-min step change instead
of the empirical max.

WHY p99 FOR THE COUNTERFACTUAL. ml/check_caps.py sets the canon ramps to the
full-history empirical MAX, and flags coal_brown's down-ramp itself: -1553.6
MW/5min is a single unit trip (2024-02-13). Measured over 500,831 rows the p99 is
+127.1 / -123.3, so the canon down-ramp overstates plausible movement by ~12x.

The max is the right envelope for FEASIBILITY VALIDATION -- you must never
declare real historical data infeasible. It is the wrong envelope for
COUNTERFACTUAL SIMULATION -- what the fleet plausibly does under a demand shift
should not be governed by a once-in-five-years trip. Two jobs, two envelopes.

Everything else (shift, nd construction, model, projection, figure layout)
is identical to plan1_figures.deliverable_b, so the output is directly
comparable to planA/cf_{scaled,masked}_size_aware.png.

    python3 imputation/plan1_cf_p99.py                     # 4 consecutive days
    python3 imputation/plan1_cf_p99.py --days rise         # 4 largest-rise days
    python3 imputation/plan1_cf_p99.py --pct 95            # p95 instead of p99
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
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "lib"))

import plan1_figures as F                                              # noqa: E402
import constraints as C                                                # noqa: E402
from gap_data import TARGETS, SIGN                                     # noqa: E402
from model import BiLSTMImputer                                        # noqa: E402
from shift_model import FixedPercentageShift                           # noqa: E402

OUT = ROOT / "planA"
DT = 5.0 / 60.0
COAL = TARGETS.index("coal_brown")


def patch_coal_ramp(up: float, dn: float):
    """Override coal_brown's ramp everywhere the constraint code reads it.
    constraints.py caches float32 tensors at import, and both cyclic_project (via
    _RUP_T/_RDN_T) and the violation audit (via R_UP/R_DN) must see the change."""
    C.R_UP[COAL], C.R_DN[COAL] = up, dn
    C._RUP_T = torch.tensor(C.R_UP, dtype=torch.float32)
    C._RDN_T = torch.tensor(C.R_DN, dtype=torch.float32)


def coal_ramp_percentile(pct: float):
    t = pd.read_parquet(F.TABLE)
    d = t["coal_brown"].diff().dropna().values
    return float(np.percentile(d, pct)), float(np.percentile(-d, pct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="consecutive", choices=["consecutive", "rise"],
                    help="4 consecutive highest-median-demand days (matches figA and "
                         "quick_findings), or the 4 largest counterfactual-rise days")
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--arm", default="size_aware")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    te, feat_cols, xm, xs_, ym, ys_ = F.load_test()
    CTX, GAP, FREE, Q = F.CTX, F.GAP, F.FREE, F.Q
    size_aware = args.arm == "size_aware"

    up0, dn0 = float(C.R_UP[COAL]), float(C.R_DN[COAL])
    up, dn = coal_ramp_percentile(args.pct)
    print(f"coal_brown ramp: canon {up0:.1f}/{dn0:.1f}  ->  p{args.pct:g} {up:.1f}/{dn:.1f} MW/5min")

    # ---- day selection ----
    if args.days == "consecutive":
        pick, full = F.pick_days(te, n=4)
    else:
        F.N_RISE = 4
        pick, full = F.pick_large_rise_days(te, n=4)
    print(f"days: {[str(full[i].date()) for i in pick]}")

    # ---- shift / nd construction: identical to plan1_figures.deliverable_b ----
    hour = te.index.hour
    free_all = (hour >= FREE[0]) & (hour < FREE[1])
    sh = FixedPercentageShift(Q, Q, free_hours=FREE).transform(
        pd.DataFrame({"demand_mw": te["demand_mw"].values,
                      "price_aud_per_mwh": te["price_aud_per_mwh"].values}, index=te.index))
    dem_s = sh["demand_mw"].values
    d_dem = dem_s - te["demand_mw"].values
    nd_bal_base = te[TARGETS].values @ SIGN
    nd_bal_cf = nd_bal_base + d_dem
    nd_feat_cf = te["net_demand"].values + d_dem

    def overrides(span_idx):
        rows = te.iloc[span_idx]; fr = free_all[span_idx]
        return {"demand_mw": dem_s[span_idx], "net_demand": nd_feat_cf[span_idx],
                "price_aud_per_mwh": np.where(fr, 0.0, rows["price_aud_per_mwh"].values)}

    rows_all = np.concatenate([np.where(te.index.normalize() == full[di])[0] for di in pick])
    shade = [(o * 288 + FREE[0] * 12, o * 288 + FREE[1] * 12) for o in range(len(pick))]
    act_disp = te.iloc[rows_all][TARGETS].values.astype(np.float64)
    nd_act = te.iloc[rows_all]["net_demand"].values
    nd_cf_line = nd_feat_cf[rows_all]

    model = BiLSTMImputer(n_features=len(feat_cols))
    model.load_state_dict(torch.load(F.WEIGHTS / f"{args.arm}.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    def run(mode):
        """One counterfactual pass over the picked days. Returns (disp, violations,
        free-window energy delta per channel vs the ACTUAL free window)."""
        days_out, viol = [], {"bal>1MW": 0, "ramp": 0, "neg": 0, "SOC": 0}
        dE = np.zeros(6)
        for di in pick:
            day_rows = np.where(te.index.normalize() == full[di])[0]
            truth_day = te.iloc[day_rows][TARGETS].values.astype(np.float64)
            g0 = FREE[0] * 12
            rows = day_rows[g0:g0 + GAP]
            span = np.arange(rows[0] - CTX, rows[-1] + 1 + CTX)
            if mode == "scaled":
                day_disp = truth_day * (nd_bal_cf[day_rows] / nd_bal_base[day_rows])[:, None]
            else:
                day_disp = truth_day.copy()
            ctx_disp = np.zeros((len(span), 6))
            for k, r in enumerate(span):
                pos = np.where(day_rows == r)[0]
                ctx_disp[k] = day_disp[pos[0]] if len(pos) else te.iloc[r][TARGETS].values
            fill, _, _ = F.model_fill(model, te, feat_cols, xm, xs_, ym, ys_,
                                      rows, overrides(span), ctx_disp)
            day_disp[g0:g0 + GAP] = fill
            day_disp = F.project_full_day(day_disp, day_disp[0].copy(),
                                          day_disp[-1].copy(), nd_bal_cf[day_rows], size_aware)
            days_out.append(day_disp)
            dE += (day_disp[g0:g0 + GAP] - truth_day[g0:g0 + GAP]).sum(0) * DT
            bal = np.abs((day_disp * SIGN).sum(-1) - nd_bal_cf[day_rows])
            dd = np.diff(day_disp, axis=0)
            viol["bal>1MW"] += int((bal > 1.0).sum())
            viol["ramp"] += int(((dd > C.R_UP + 0.6) | (dd < -(C.R_DN + 0.6))).sum())
            viol["neg"] += int((day_disp < -0.1).sum())
            viol["SOC"] += int(C._soc_swing_mwh(day_disp) > C.BATT_CAP_MWH + 1e-6)
        return np.concatenate(days_out), viol, dE

    # ---- canon reference (numbers only, for the comparison table) ----
    ref = {}
    for mode in ("scaled", "masked"):
        patch_coal_ramp(up0, dn0)
        ref[mode] = run(mode)[2]

    # ---- p99 run + figures ----
    patch_coal_ramp(up, dn)
    tag = f"p{args.pct:g}"
    for mode in ("scaled", "masked"):
        disp, viol, dE = run(mode)
        title = ("scaled before/after — off-window x nd_after/nd_before, window model-filled"
                 if mode == "scaled" else
                 "actual + masked window — off-window actual, window model-filled")
        fig, ax = plt.subplots(2, 1, figsize=(max(16, 2.6 * len(pick)), 9),
                               sharex=True, sharey=True)
        F.draw_stack(ax[0], te.iloc[rows_all], act_disp,
                     te.iloc[rows_all]["demand_mw"].values, shade,
                     "actual (free window shaded; red = net demand)", nd_line=nd_act)
        F.draw_stack(ax[1], te.iloc[rows_all], disp, dem_s[rows_all], shade,
                     f"counterfactual — {title}  [{args.arm}, coal ramp {tag}]  "
                     f"(rebound {Q:g}%, reduce {Q:g}%)", nd_line=nd_cf_line)
        fig.suptitle(f"plan1 B — {args.arm}, {mode}, coal ramp {tag} "
                     f"({up:.0f}/{dn:.0f} MW/5min): actual vs counterfactual")
        for a in fig.axes:
            a.set_xticks([k * 288 for k in range(len(pick))])
            a.set_xticklabels([str(full[di].date()) for di in pick])
        fig.axes[0].legend(loc="upper left", ncol=6, fontsize=7, framealpha=0.9)
        fig.tight_layout()
        fp = out_dir / f"cf_{mode}_{args.arm}_coal{tag}.png"
        fig.savefig(fp, dpi=140); plt.close(fig)
        print(f"\n[{mode}] wrote {fp}")
        print(f"  violations over {len(pick)} full days: {viol}")
        print(f"  free-window energy change vs actual (MWh), canon -> {tag}:")
        print(f"    {'channel':22s} {'canon':>10s} {'p'+str(args.pct):>10s} {'shift':>10s}")
        for i, t in enumerate(TARGETS):
            print(f"    {t:22s} {ref[mode][i]:+10,.0f} {dE[i]:+10,.0f} {dE[i]-ref[mode][i]:+10,.0f}")
        print(f"    {'SIGN . dE':22s} {float(ref[mode]@SIGN):+10,.0f} {float(dE@SIGN):+10,.0f}")

    patch_coal_ramp(up0, dn0)


if __name__ == "__main__":
    main()
