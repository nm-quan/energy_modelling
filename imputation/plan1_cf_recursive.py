"""4-day counterfactual (scaled + masked) for the RECURSIVE RAYEN head, plus the
attribution that answers "did the model decide the allocation?".

Pipeline, nd construction, day selection and figure layout are identical to
plan1_figures.deliverable_b / plan1_cf_p99.py, so the output is directly
comparable to planA/cf_{scaled,masked}_size_aware.png.

TWO ATTRIBUTION MEASURES, because the recursive head needs both:

  decomposition   P_t = A_t + move_t. Reports |move| against |dA|. NOTE this
                  UNDERSTATES the network: unlike the incumbent head, A_t here is
                  built from P_{t-1}, which is the network's own earlier output,
                  so the anchor already carries model influence.

  null ablation   re-run everything with the direction forced to zero (P = the
                  pure glide-to-pR anchor trajectory) and compare the two
                  counterfactual responses. Model-agnostic and falsifiable: if
                  the trained network and the null network produce the same
                  response, the network decided nothing.

    python3 imputation/plan1_cf_recursive.py
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
from gap_data import TARGETS, SIGN, TARGET_FEAT_IDX                    # noqa: E402
from model import BiLSTMImputer                                        # noqa: E402
from recursive_head import recursive_rayen                             # noqa: E402
from shift_model import FixedPercentageShift                           # noqa: E402

OUT = ROOT / "planA"
DT = 5.0 / 60.0


def set_coal_ramp_percentile(pct: float):
    """coal_brown ramp -> pct-th percentile of the historical 5-min step change.
    Canon is the empirical MAX and check_caps.py flags the down-ramp as a single
    unit trip (-1553.6 vs p99 -123.3). Patches both the numpy arrays (violation
    audits) and the cached f32 tensors (the head)."""
    d = pd.read_parquet(F.TABLE)["coal_brown"].diff().dropna().values
    i = TARGETS.index("coal_brown")
    up, dn = float(np.percentile(d, pct)), float(np.percentile(-d, pct))
    old = (float(C.R_UP[i]), float(C.R_DN[i]))
    C.R_UP[i], C.R_DN[i] = up, dn
    C._RUP_T = torch.tensor(C.R_UP, dtype=torch.float32)
    C._RDN_T = torch.tensor(C.R_DN, dtype=torch.float32)
    print(f"coal_brown ramp: canon {old[0]:.1f}/{old[1]:.1f} -> "
          f"p{pct:g} {up:.1f}/{dn:.1f} MW/5min")
    return up, dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="recursive")
    ap.add_argument("--coal-pct", type=float, default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ramp_tag = "canon ramps"
    if args.coal_pct is not None:
        up, dn = set_coal_ramp_percentile(args.coal_pct)
        ramp_tag = f"coal ramp p{args.coal_pct:g} ({up:.0f}/{dn:.0f} MW/5min)"

    te, feat_cols, xm, xs_, ym, ys_ = F.load_test()
    CTX, GAP, FREE, Q = F.CTX, F.GAP, F.FREE, F.Q
    tfi = np.asarray(TARGET_FEAT_IDX)

    model = BiLSTMImputer(n_features=len(feat_cols), n_targets=7)
    model.load_state_dict(torch.load(F.WEIGHTS / f"{args.ckpt}.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    pick, full = F.pick_days(te, n=4)
    print(f"days: {[str(full[i].date()) for i in pick]}")

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

    def rec_fill(rows, nd_bal, ov=None, ctx=None, null=False):
        """Recursive-head output (GAP,6) MW for gap `rows`. null=True zeroes the
        direction, leaving the pure glide-to-pR anchor trajectory."""
        span = np.arange(rows[0] - CTX, rows[-1] + 1 + CTX)
        vals = te.iloc[span][feat_cols].values.astype(np.float64).copy()
        if ov:
            for col, series in ov.items():
                vals[:, feat_cols.index(col)] = series
        if ctx is not None:
            for k, t_ in enumerate(TARGETS):
                vals[:, feat_cols.index(t_)] = ctx[:, k]
        X = ((vals - xm) / xs_).astype(np.float32)
        mk = np.ones((len(span), 1), np.float32); mk[CTX:CTX + GAP] = 0.0
        X[CTX:CTX + GAP, tfi] = 0.0
        pL = vals[CTX - 1, [feat_cols.index(t_) for t_ in TARGETS]]
        pR = vals[CTX + GAP, [feat_cols.index(t_) for t_ in TARGETS]]
        with torch.no_grad():
            raw = model(torch.from_numpy(X[None]), torch.from_numpy(mk[None]))[:, CTX:CTX + GAP].double()
            if null:
                raw = torch.cat([torch.zeros_like(raw[..., :6]), raw[..., 6:]], -1)
            P, dbg = recursive_rayen(raw, torch.tensor(pL[None]), torch.tensor(pR[None]),
                                     torch.tensor(nd_bal[None]), return_debug=True)
        return P[0].numpy(), dbg

    def run(mode, null=False, shifted=True):
        """One pass over the picked days. Returns concatenated day dispatch, the
        violation audit, free-window energy, and the anchor/move decomposition."""
        days_out, viol = [], {"bal>1MW": 0, "ramp": 0, "neg": 0, "SOC": 0}
        E = np.zeros(6); Anc = np.zeros(6); Mov = np.zeros(6)
        nd_bal_use = nd_bal_cf if shifted else nd_bal_base
        for di in pick:
            day_rows = np.where(te.index.normalize() == full[di])[0]
            truth_day = te.iloc[day_rows][TARGETS].values.astype(np.float64)
            g0 = FREE[0] * 12
            rows = day_rows[g0:g0 + GAP]
            span = np.arange(rows[0] - CTX, rows[-1] + 1 + CTX)
            if shifted and mode == "scaled":
                day_disp = truth_day * (nd_bal_cf[day_rows] / nd_bal_base[day_rows])[:, None]
            else:
                day_disp = truth_day.copy()
            ctx_disp = np.zeros((len(span), 6))
            for k, r in enumerate(span):
                pos = np.where(day_rows == r)[0]
                ctx_disp[k] = day_disp[pos[0]] if len(pos) else te.iloc[r][TARGETS].values
            fill, dbg = rec_fill(rows, nd_bal_use[rows],
                                 overrides(span) if shifted else None, ctx_disp, null=null)
            day_disp[g0:g0 + GAP] = fill
            days_out.append(day_disp)
            E += fill.sum(0) * DT
            Anc += dbg["anchor"][0].numpy().sum(0) * DT
            Mov += dbg["move"][0].numpy().sum(0) * DT
            bal = np.abs((day_disp * SIGN).sum(-1) - nd_bal_use[day_rows])
            dd = np.diff(day_disp, axis=0)
            viol["bal>1MW"] += int((bal[g0:g0 + GAP] > 1.0).sum())
            viol["ramp"] += int(((dd > C.R_UP + 0.6) | (dd < -(C.R_DN + 0.6))).sum())
            viol["neg"] += int((day_disp < -0.1).sum())
            viol["SOC"] += int(C._soc_swing_mwh(day_disp) > C.BATT_CAP_MWH + 1e-6)
        return np.concatenate(days_out), viol, E, Anc, Mov

    rows_all = np.concatenate([np.where(te.index.normalize() == full[di])[0] for di in pick])
    shade = [(o * 288 + FREE[0] * 12, o * 288 + FREE[1] * 12) for o in range(len(pick))]
    act_disp = te.iloc[rows_all][TARGETS].values.astype(np.float64)
    nd_act = te.iloc[rows_all]["net_demand"].values

    for mode in ("scaled", "masked"):
        disp, viol, E_cf, A_cf, M_cf = run(mode, null=False, shifted=True)
        _,    _,    E_b,  A_b,  M_b  = run(mode, null=False, shifted=False)
        _,    _,    E_cfn, _,   _    = run(mode, null=True,  shifted=True)
        _,    _,    E_bn,  _,   _    = run(mode, null=True,  shifted=False)

        title = ("scaled before/after — off-window x nd_after/nd_before, window model-filled"
                 if mode == "scaled" else
                 "actual + masked window — off-window actual, window model-filled")
        fig, ax = plt.subplots(2, 1, figsize=(max(16, 2.6 * len(pick)), 9),
                               sharex=True, sharey=True)
        F.draw_stack(ax[0], te.iloc[rows_all], act_disp,
                     te.iloc[rows_all]["demand_mw"].values, shade,
                     "actual (free window shaded; red = net demand)", nd_line=nd_act)
        F.draw_stack(ax[1], te.iloc[rows_all], disp, dem_s[rows_all], shade,
                     f"counterfactual — {title}  [recursive RAYEN head, 1 epoch, {ramp_tag}]  "
                     f"(rebound {Q:g}%, reduce {Q:g}%)", nd_line=nd_feat_cf[rows_all])
        fig.suptitle(f"plan1 B — recursive head, {mode}, {ramp_tag}: actual vs counterfactual")
        for a in fig.axes:
            a.set_xticks([k * 288 for k in range(len(pick))])
            a.set_xticklabels([str(full[di].date()) for di in pick])
        fig.axes[0].legend(loc="upper left", ncol=6, fontsize=7, framealpha=0.9)
        fig.tight_layout()
        fp = out_dir / f"cf_{mode}_recursive{args.suffix}.png"
        fig.savefig(fp, dpi=140); plt.close(fig)

        resp, resp_null = E_cf - E_b, E_cfn - E_bn
        print(f"\n[{mode}] wrote {fp}")
        print(f"  violations, free window over {len(pick)} days: {viol}")
        print(f"\n  free-window counterfactual response (MWh)")
        print(f"    {'channel':22s} {'trained':>11s} {'null net':>11s} {'difference':>12s}")
        print("    " + "-" * 58)
        for i, t_ in enumerate(TARGETS):
            print(f"    {t_:22s} {resp[i]:+11,.0f} {resp_null[i]:+11,.0f} {resp[i]-resp_null[i]:+12,.0f}")
        print("    " + "-" * 58)
        print(f"    {'SIGN . response':22s} {float(resp@SIGN):+11,.0f} {float(resp_null@SIGN):+11,.0f}")
        d = np.abs(resp - resp_null).sum(); tot = np.abs(resp).sum()
        print(f"\n  NULL ABLATION: trained and null responses differ by {d:,.0f} of "
              f"{tot:,.0f} MWh = {100*d/tot:.1f}%")
        dA, dM = np.abs(A_cf - A_b).sum(), np.abs(M_cf - M_b).sum()
        print(f"  decomposition (lower bound): |move| share = {100*dM/(dA+dM):.1f}% "
              f"(anchor carries earlier model influence)")


if __name__ == "__main__":
    main()
