"""planA-protocol counterfactual over 10 days, for every planB arm plus the
no-model NEM baseline and plain interpolation.

PROTOCOL is planA's (plan1_figures.deliverable_b), unchanged:
  free window   11:00-14:00, rebound 2.4% / reduce 2.4%
  nd target     nd_cf = SIGN.truth + delta(demand); renewables fixed
  two modes     scaled  off-window dispatch x nd_after/nd_before
                masked  off-window stays actual
  context       the model is fed the mode's OWN off-window dispatch
  post          project_full_day over all 288 steps

WHAT IS COMPARED, and the one confound. All planB arms share a backbone size, a
loss, an activation and the p99.9 envelope, so they are a clean A/B. The two
planA arms (curt_plain, curt_recursive) are included because they were asked for,
but they were trained through the MAX envelope with a different loss, so their
row is not a like-for-like architecture comparison -- the envelope is named in the
table. The no-model baseline and interpolation use no network at all.

The benchmark column is the real grid's measured 3-hour marginal response
(dispatch_study/response.md Study A), which is what any of these should be
reproducing.

    python3 planB/counterfactual.py --days 10
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
sys.path.insert(0, str(ROOT / "imputation")); sys.path.insert(0, str(ROOT / "dispatch_study"))
sys.path.insert(0, str(HERE))

import plan1_figures as F                                              # noqa: E402
import constraints as C                                                # noqa: E402
from gap_data import TARGETS, SIGN, TARGET_FEAT_IDX                    # noqa: E402
from shift_model import FixedPercentageShift                           # noqa: E402
from common import COLORS, LABEL, INK, MUTED, FUELS                    # noqa: E402
from nets import build_delta                                           # noqa: E402
from heads import HEADS                                                # noqa: E402
import limits as LIM                                                   # noqa: E402
from fit import (curt_activation, CURT_COLS, CTX, GAP, apply_residual,                 # noqa: E402
                 ARMS as ARM_SPEC, BACKBONES)

CURT_NAMES = ["wind_curtailment", "solar_curtailment"]

OUT = ROOT / "planB" / "results"
DT = 5.0 / 60.0
# real grid, 3-hour marginal response, normalised (dispatch_study Study A)
EMP = {"coal_brown": 43.4, "hydro": 29.3, "gas_ocgt": 9.7, "gas_steam": 3.7,
       "battery_discharging": 7.4, "battery_charging": 6.6}
PLANB = [a for a in ARM_SPEC if a != "blackout"]
META = {a: (ARM_SPEC[a][0], ARM_SPEC[a][1], "p99.9") for a in ARM_SPEC}


def draw(ax, rows_df, disp, dem, nd, shade, title, curt=None):
    """curt (n,2) is the wind/solar curtailment to draw hatched above the stack.

    In planB curtailment is a PARALLEL prediction: it is not credited into the
    balance target (planA's nd_eff coupling was removed), so it never moves energy
    into or out of the dispatch. The band is what the model thinks was available
    and spilled, nothing more."""
    x = np.arange(len(disp)); base = np.zeros(len(x))
    ti = {t: i for i, t in enumerate(TARGETS)}
    for f_ in FUELS:
        top = base + disp[:, ti[f_]]
        ax.fill_between(x, base, top, color=COLORS[f_], lw=0.3, edgecolor="white",
                        zorder=2, label=LABEL[f_])
        base = top
    for col in ("wind", "solar_utility"):
        top = base + rows_df[col].values
        ax.fill_between(x, base, top, color=COLORS[col], lw=0.3, edgecolor="white",
                        zorder=2, label=LABEL[col])
        base = top
    if curt is not None:
        for k2, key in enumerate(("wind", "solar_utility")):
            top = base + curt[:, k2]
            ax.fill_between(x, base, top, color=COLORS[key], alpha=0.35, lw=0,
                            hatch="///", zorder=2, label=f"{LABEL[key]} curtailed")
            base = top
    ax.fill_between(x, 0, -disp[:, ti["battery_charging"]],
                    color=COLORS["battery_charging"], lw=0.3, edgecolor="white",
                    zorder=2, label=LABEL["battery_charging"])
    ax.plot(x, dem, color=INK, lw=1.1, ls="--", zorder=4, label="demand")
    ax.plot(x, nd, color="#B3261E", lw=1.1, zorder=4, label="net demand")
    for a, b in shade:
        ax.axvspan(a, b, color="#000000", alpha=0.07, zorder=1)
    ax.axhline(0, color=MUTED, lw=0.5); ax.margins(x=0)
    ax.grid(True, axis="y", alpha=0.15, lw=0.5, zorder=0)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=9, color=INK, pad=2)
    ax.set_ylabel("MW", fontsize=7.5, color=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--q", type=float, default=None,
                    help="rebound = reduction, in %%. Default 2.4 came from the MAX "
                         "ramp envelope. Under p99.9 the fleet has 1,241.6 MW of "
                         "one-step up-ramp, and a 2.4%% rebound needs a 1,402 MW jump "
                         "at the window edge -- 113%% of capacity, so infeasible. "
                         "1.0%% needs 584 MW (47%%).")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="checkpoint stems to run; default is all found")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    Q = args.q if args.q is not None else F.Q

    te, feat_cols, xm, xs_, ym, ys_ = F.load_test()
    tfi = np.asarray(TARGET_FEAT_IDX)
    c_m = torch.tensor(xm[CURT_COLS].astype(np.float32))
    c_s = torch.tensor(xs_[CURT_COLS].astype(np.float32))
    ys_mt = torch.tensor(ym, dtype=torch.float32)
    ys_st = torch.tensor(ys_, dtype=torch.float32)
    pick, full = F.pick_days(te, n=args.days)
    g0 = F.FREE[0] * 12
    print(f"days {full[pick[0]].date()} .. {full[pick[-1]].date()}  "
          f"window {F.FREE[0]}:00-{F.FREE[1]}:00")

    # ISOLATED VICTORIA: no interconnector, so local supply must equal local
    # demand and the demand line is DEFINED as what the local fleet served.
    # AEMO's demand_mw exceeds that by ~197 MW on average (real export), which
    # would otherwise sit in every panel as an unexplained gap.
    nd_base = te[TARGETS].values @ SIGN                 # what the six fuels supply
    dem_local = nd_base + te["wind"].values + te["solar_utility"].values
    free = (te.index.hour >= F.FREE[0]) & (te.index.hour < F.FREE[1])
    sh = FixedPercentageShift(Q, Q, free_hours=F.FREE).transform(
        pd.DataFrame({"demand_mw": dem_local,
                      "price_aud_per_mwh": te["price_aud_per_mwh"].values}, index=te.index))
    dem_s = sh["demand_mw"].values
    d_dem = dem_s - dem_local
    nd_cf = nd_base + d_dem
    # the nd FEATURE the model reads is now the same supply-side quantity the head
    # balances to -- previously it was the demand-side one, off by up to 4,974 MW
    ndf_cf = nd_cf

    models = {}
    want = args.arms or PLANB
    for arm in want:
        p = OUT / f"{arm}.pt"
        if not p.exists():
            print(f"  [skip] {arm}"); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        head_fn, n_disp = HEADS[ck["head"]]
        bb = ck.get("backbone", "brits" if arm == "brits" else "bilstm")
        META.setdefault(arm, (bb, ck["head"], "p99.9"))
        m = (BACKBONES[bb](len(feat_cols), n_disp, hidden=ck["hidden"]) if bb == "bilstm"
             else BACKBONES[bb](len(feat_cols), TARGET_FEAT_IDX, CURT_COLS, n_disp,
                                hidden=ck["hidden"]))
        m.load_state_dict(ck["state"]); m.eval()
        models[arm] = (m, head_fn, n_disp, bb, ck.get("residual", False))

    def fill(arm, rows, ctx, shifted):
        span = np.arange(rows[0] - CTX, rows[-1] + 1 + CTX)
        vals = te.iloc[span][feat_cols].values.astype(np.float64).copy()
        if shifted:
            vals[:, feat_cols.index("demand_mw")] = dem_s[span]
            vals[:, feat_cols.index("net_demand")] = ndf_cf[span]
            vals[:, feat_cols.index("price_aud_per_mwh")] = np.where(
                free[span], 0.0, te.iloc[span]["price_aud_per_mwh"].values)
        for k, t_ in enumerate(TARGETS):
            vals[:, feat_cols.index(t_)] = ctx[:, k]
        nd_t = (nd_cf if shifted else nd_base)[rows]
        # supply-side nd feature, from the mode's own dispatch world, matching the
        # balance target. Inside the gap the true dispatch is not available, so the
        # window carries the scenario's nd directly.
        vals[:, feat_cols.index("net_demand")] = ctx @ SIGN
        vals[CTX:CTX + GAP, feat_cols.index("net_demand")] = nd_t
        pL = vals[CTX - 1, [feat_cols.index(t_) for t_ in TARGETS]]
        pR = vals[CTX + GAP, [feat_cols.index(t_) for t_ in TARGETS]]
        X = ((vals - xm) / xs_).astype(np.float32)
        mk = np.ones((len(span), 1), np.float32); mk[CTX:CTX + GAP] = 0.0
        X[CTX:CTX + GAP, tfi] = 0.0
        X[CTX:CTX + GAP, CURT_COLS] = 0.0
        m, head_fn, n_disp, bb, resid = models[arm]
        with torch.no_grad():
            xt = torch.from_numpy(X[None]); mkt = torch.from_numpy(mk[None])
            dl = build_delta(mkt, len(feat_cols)) if bb == "brits" else None
            d_raw, c_raw = m(xt, mkt, dl)
            d_raw = d_raw[:, CTX:CTX + GAP]
            curt = curt_activation(c_raw[:, CTX:CTX + GAP], c_m, c_s)[0].numpy()
            if resid:
                tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
                pLz = (pL - ym) / ys_; pRz = (pR - ym) / ys_
                d_raw = apply_residual(d_raw, torch.tensor(
                    (pLz[None, None] + tt * (pRz - pLz)[None, None]).astype(np.float32)))
            if head_fn is None:
                P = (d_raw[..., :6] * ys_st + ys_mt)[0].numpy()
            else:
                F_ = (d_raw * ys_st + ys_mt if n_disp == 6 else
                      torch.cat([d_raw[..., :6] * ys_st + ys_mt, d_raw[..., 6:]], -1))
                P = head_fn(F_, torch.tensor(pL[None]).float(),
                            torch.tensor(pR[None]).float(),
                            torch.tensor(nd_t[None]).float())[0].numpy()
        return P.astype(np.float64), curt.astype(np.float64)

    def run(arm, mode):
        days_d, days_c = [], []
        raw_v = {"bal>1MW": 0, "ramp": 0, "neg": 0}
        E_b = np.zeros(6); E_c = np.zeros(6); dd = 0.0
        Cb = np.zeros(2); Cc = np.zeros(2); Cact = np.zeros(2)
        for d in pick:
            dr = np.where(te.index.normalize() == full[d])[0]
            rows = dr[g0:g0 + GAP]
            span = np.arange(rows[0] - CTX, rows[-1] + 1 + CTX)
            truth = te.iloc[dr][TARGETS].values.astype(np.float64)
            day = (truth * (nd_cf[dr] / nd_base[dr])[:, None] if mode == "scaled"
                   else truth.copy())
            ctx = np.zeros((len(span), 6))
            for k, r in enumerate(span):
                pos = np.where(dr == r)[0]
                ctx[k] = day[pos[0]] if len(pos) else te.iloc[r][TARGETS].values
            Pb, cb = fill(arm, rows, ctx, shifted=False)
            Pc, cc = fill(arm, rows, ctx, shifted=True)
            E_b += Pb.sum(0) * DT; E_c += Pc.sum(0) * DT
            Cb += cb.sum(0) * DT; Cc += cc.sum(0) * DT
            Cact += te.iloc[rows][CURT_NAMES].values.sum(0) * DT
            dd += d_dem[rows].sum() * DT
            day[g0:g0 + GAP] = Pc
            b = np.abs((day * SIGN).sum(-1) - nd_cf[dr])
            df = np.diff(day, axis=0)
            raw_v["bal>1MW"] += int((b[g0:g0 + GAP] > 1.0).sum())
            raw_v["ramp"] += int(((df > LIM.R_UP + 0.6) | (df < -(LIM.R_DN + 0.6))).sum())
            raw_v["neg"] += int((day < -0.1).sum())
            day = F.project_full_day(day, day[0].copy(), day[-1].copy(), nd_cf[dr], True)
            dayc = te.iloc[dr][CURT_NAMES].values.astype(np.float64).copy()
            dayc[g0:g0 + GAP] = cc                 # predicted inside, actual outside
            days_d.append(day); days_c.append(dayc)
        return dict(disp=np.concatenate(days_d), curt=np.concatenate(days_c),
                    raw_v=raw_v, resp=E_c - E_b, dd=dd,
                    Cb=Cb, Cc=Cc, Cact=Cact)

    rows_all = np.concatenate([np.where(te.index.normalize() == full[d])[0] for d in pick])
    shade = [(o * 288 + g0, o * 288 + g0 + GAP) for o in range(len(pick))]
    act = te.iloc[rows_all][TARGETS].values.astype(np.float64)

    res = {}
    for mode in ("scaled", "masked"):
        for arm in models:
            res[(mode, arm)] = run(arm, mode)
            print(f"  {mode:7s} {arm}")

    # ---------------- figure: one row per arm, masked mode ----------------
    order = [a for a in want if a in models]
    fig, ax = plt.subplots(len(order) + 1, 1, figsize=(2.0 * len(pick) + 3,
                                                       2.6 * (len(order) + 1)),
                           sharex=True, sharey=True)
    draw(ax[0], te.iloc[rows_all], act, dem_local[rows_all], nd_base[rows_all], shade,
         f"actual (shaded = the {F.FREE[0]}:00-{F.FREE[1]}:00 window each model fills)",
         curt=te.iloc[rows_all][CURT_NAMES].values)
    for i, arm in enumerate(order):
        draw(ax[i + 1], te.iloc[rows_all], res[("masked", arm)]["disp"],
             dem_s[rows_all], nd_cf[rows_all], shade,
             f"{arm}  ({META[arm][0]} + {META[arm][1]} head)",
             curt=res[("masked", arm)]["curt"])
    ax[-1].set_xticks([k * 288 + 144 for k in range(len(pick))])
    ax[-1].set_xticklabels([str(full[d].date())[5:] for d in pick], fontsize=8)
    ax[0].legend(loc="upper left", ncol=6, fontsize=7, frameon=False)
    fig.suptitle(f"planB counterfactual, masked mode - {len(pick)} days, planA protocol "
                 f"(rebound {Q:g}%, reduce {Q:g}%)", fontsize=12.5, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUT / f"counterfactual_10day{args.suffix}.png", dpi=120, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}/counterfactual_10day{args.suffix}.png")

    # ---------------- report ----------------
    L = [f"# planB counterfactual - {len(pick)} days, planA protocol", "",
         f"Days {full[pick[0]].date()} .. {full[pick[-1]].date()}. Free window "
         f"{F.FREE[0]}:00-{F.FREE[1]}:00, rebound {Q:g}% / reduce {Q:g}%, both modes, "
         f"context = the mode's own off-window dispatch, `project_full_day` over all "
         f"288 steps. All arms 1 epoch, hidden 192, one MSE, p99.9 envelope.", "",
         "| arm | backbone | head | envelope |", "| --- | --- | --- | --- |"]
    for a in order:
        L.append(f"| {a} | {META[a][0]} | {META[a][1]} | {META[a][2]} |")

    L += ["", "## Feasibility of the raw filled window (before `project_full_day`)", "",
          "| arm | balance > 1 MW | ramp | negative |", "| --- | ---: | ---: | ---: |"]
    for a in order:
        v = res[("masked", a)]["raw_v"]
        L.append(f"| {a} | {v['bal>1MW']} | {v['ramp']} | {v['neg']} |")
    L += ["", f"_{len(pick)*GAP} window steps x 6 channels = {len(pick)*GAP*6} cells; "
          "ramp counted over the full day._", ""]

    for mode in ("scaled", "masked"):
        L += [f"## Response, {mode} (MWh in the window, share of the total)", "",
              "| channel | " + " | ".join(order) + " | REAL GRID |",
              "| --- |" + " ---: |" * (len(order) + 1)]
        tot = {a: abs(float(res[(mode, a)]["resp"] @ SIGN)) for a in order}
        for i, t_ in enumerate(TARGETS):
            cells = [f"{res[(mode,a)]['resp'][i]:+,.0f} ({100*abs(res[(mode,a)]['resp'][i])/max(tot[a],1e-9):.0f}%)"
                     for a in order]
            L.append(f"| {LABEL[t_]} | " + " | ".join(cells) + f" | {EMP[t_]:.1f}% |")
        L.append("| **Σ signed** | " + " | ".join(
            f"{float(res[(mode,a)]['resp']@SIGN):+,.0f}" for a in order)
            + f" | needs {res[(mode, order[0])]['dd']:+,.0f} |")
        L.append("")

    L += ["## Curtailment in the window (MWh) - a PARALLEL prediction", "",
          "planB removed planA's curtailment credit, so curtailment does not enter "
          "the balance target and cannot move energy into or out of the dispatch. "
          "These are the models' predictions of what was available and spilled.", "",
          "| arm | actual | base-case predicted | counterfactual | delta |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for a in order:
        r = res[("masked", a)]
        L.append(f"| {a} | {r['Cact'].sum():,.0f} | {r['Cb'].sum():,.0f} | "
                 f"{r['Cc'].sum():,.0f} | {r['Cc'].sum()-r['Cb'].sum():+,.0f} |")
    L += ["", "_A negative delta means the model spills less under the higher-demand "
          "counterfactual, which is the physically expected direction even though "
          "nothing forces it here._", ""]

    (OUT / f"counterfactual{args.suffix}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
