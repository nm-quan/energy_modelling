"""10-day counterfactual: plain BiLSTM vs BiLSTM + recursive RAYEN head, both
also predicting wind and solar curtailment.

PROTOCOL IS planA's (plan1_figures.deliverable_b), not a variant of it:
  free window   11:00-14:00, rebound 2.4% / reduce 2.4%
  nd target     nd_bal_cf = SIGN.truth + delta(total demand); renewables fixed
  two modes     scaled  off-window dispatch x nd_after/nd_before, window model-filled
                masked  off-window stays ACTUAL, window model-filled
  context       the model is fed the mode's OWN off-window dispatch (ctx_disp),
                not the actual, so it sees the world the mode implies
  post          project_full_day over all 288 steps (soc=True) so SOC, balance,
                ramp and box hold across the whole day, identically for both arms

The only departure from planA is forced by the new output: the recursive arm
couples curtailment into its balance target inside the window,
nd_eff = nd - (curt_actual - curt_pred), so spilling less lowers what the fuels
must cover. The day-level target passed to project_full_day carries that same
credit inside the window and nd_bal_cf everywhere else.

Also written: the same protocol with the window moved to the evening peak
(17:00-20:00), because dispatch_study/ shows the evening turn-up is 1,997 MW
against 314 MW in the morning and midday has no battery-discharge signal at all.

    python3 imputation/curt_cf.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "dispatch_study"))

import plan1_figures as F                                              # noqa: E402
import constraints as C                                                # noqa: E402
from gap_data import TARGETS, SIGN, TARGET_FEAT_IDX                    # noqa: E402
from model import BiLSTMImputer                                        # noqa: E402
from recursive_head import recursive_rayen                             # noqa: E402
from curt_train import curt_activation                                 # noqa: E402
from shift_model import FixedPercentageShift                           # noqa: E402
from common import COLORS, LABEL, INK, MUTED, FUELS                    # noqa: E402

OUT = ROOT / "planA"
CTX, GAP = F.CTX, F.GAP
Q = F.Q
CURT_COLS = [19, 20]
DT = 5.0 / 60.0
ARMS = ["plain", "recursive"]
EMP = {"coal_brown": 40.4, "hydro": 30.2, "gas_ocgt": 11.1, "gas_steam": 4.9,
       "battery_discharging": 6.0, "battery_charging": 7.4}      # dispatch_study


def draw(ax, rows_df, disp, curt, dem, nd, shade, title, renew=None):
    """Stacked dispatch + DELIVERED renewables + curtailment (hatched on top).

    renew (n,2) is delivered wind / solar in MW. It must be supplied whenever the
    scenario RELEASES curtailment: released energy moves out of the hatched band
    and into the solid band, so total available renewables stay conserved. Passing
    None uses the actual columns, which is only correct when curtailment is
    unchanged -- otherwise the released energy silently disappears from the stack
    while the fuels also drop by it, losing it twice."""
    if renew is None:
        renew = rows_df[["wind", "solar_utility"]].values
    x = np.arange(len(disp)); base = np.zeros(len(x))
    ti = {t: i for i, t in enumerate(TARGETS)}
    for f in FUELS:
        top = base + disp[:, ti[f]]
        ax.fill_between(x, base, top, color=COLORS[f], lw=0.4, edgecolor="white",
                        zorder=2, label=LABEL[f])
        base = top
    for k2, key in enumerate(("wind", "solar_utility")):
        top = base + renew[:, k2]
        ax.fill_between(x, base, top, color=COLORS[key], lw=0.4, edgecolor="white",
                        zorder=2, label=LABEL[key])
        base = top
    for k, key in enumerate(("wind", "solar_utility")):
        top = base + curt[:, k]
        ax.fill_between(x, base, top, color=COLORS[key], alpha=0.35, lw=0,
                        hatch="///", zorder=2, label=f"{LABEL[key]} curtailed")
        base = top
    ax.fill_between(x, 0, -disp[:, ti["battery_charging"]],
                    color=COLORS["battery_charging"], lw=0.4, edgecolor="white",
                    zorder=2, label=LABEL["battery_charging"])
    ax.plot(x, dem, color=INK, lw=1.3, ls="--", zorder=4, label="demand")
    ax.plot(x, nd, color="#B3261E", lw=1.3, zorder=4, label="net demand")
    for a, b in shade:
        ax.axvspan(a, b, color="#000000", alpha=0.07, zorder=1)
    ax.axhline(0, color=MUTED, lw=0.6); ax.margins(x=0)
    ax.grid(True, axis="y", alpha=0.18, lw=0.6, zorder=0)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=3)
    ax.set_ylabel("MW", fontsize=8, color=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    te, feat_cols, xm, xs_, ym, ys_ = F.load_test()
    tfi = np.asarray(TARGET_FEAT_IDX)
    c_m, c_s = xm[CURT_COLS], xs_[CURT_COLS]
    pick, full = F.pick_days(te, n=args.days)
    print(f"days: {full[pick[0]].date()} .. {full[pick[-1]].date()}  ({len(pick)})")

    models = {}
    for arm in ARMS:
        m = BiLSTMImputer(n_features=len(feat_cols), n_targets=8 if arm == "plain" else 9)
        m.load_state_dict(torch.load(F.WEIGHTS / f"curt_{arm}.pt", map_location="cpu",
                                     weights_only=True))
        m.eval(); models[arm] = m

    def shift_world(free):
        """planA's counterfactual drivers for a given free window."""
        fr = (te.index.hour >= free[0]) & (te.index.hour < free[1])
        sh = FixedPercentageShift(Q, Q, free_hours=free).transform(
            pd.DataFrame({"demand_mw": te["demand_mw"].values,
                          "price_aud_per_mwh": te["price_aud_per_mwh"].values},
                         index=te.index))
        dem_s = sh["demand_mw"].values
        d_dem = dem_s - te["demand_mw"].values
        nd_base = te[TARGETS].values @ SIGN
        return dict(free=fr, dem_s=dem_s, d_dem=d_dem, nd_base=nd_base,
                    nd_cf=nd_base + d_dem, ndf_cf=te["net_demand"].values + d_dem)

    def fill(arm, rows, W, ctx_disp, shifted=True):
        """planA's model_fill, extended for 8/9 outputs and curtailment masking."""
        span = np.arange(rows[0] - CTX, rows[-1] + 1 + CTX)
        vals = te.iloc[span][feat_cols].values.astype(np.float64).copy()
        if shifted:
            vals[:, feat_cols.index("demand_mw")] = W["dem_s"][span]
            vals[:, feat_cols.index("net_demand")] = W["ndf_cf"][span]
            vals[:, feat_cols.index("price_aud_per_mwh")] = np.where(
                W["free"][span], 0.0, te.iloc[span]["price_aud_per_mwh"].values)
        if ctx_disp is not None:                       # the mode's own off-window world
            for k, t_ in enumerate(TARGETS):
                vals[:, feat_cols.index(t_)] = ctx_disp[:, k]
        c_act = vals[CTX:CTX + GAP][:, CURT_COLS].copy()
        X = ((vals - xm) / xs_).astype(np.float32)
        mk = np.ones((len(span), 1), np.float32); mk[CTX:CTX + GAP] = 0.0
        X[CTX:CTX + GAP, tfi] = 0.0
        X[CTX:CTX + GAP, CURT_COLS] = 0.0
        pL = vals[CTX - 1, [feat_cols.index(t_) for t_ in TARGETS]]
        pR = vals[CTX + GAP, [feat_cols.index(t_) for t_ in TARGETS]]
        nd_t = (W["nd_cf"] if shifted else W["nd_base"])[rows]
        with torch.no_grad():
            raw = models[arm](torch.from_numpy(X[None]),
                              torch.from_numpy(mk[None]))[:, CTX:CTX + GAP].double()
            curt = curt_activation(raw[..., -2:], torch.tensor(c_m), torch.tensor(c_s))
            if arm == "plain":
                return (raw[0, :, :6].numpy() * ys_ + ym), curt[0].numpy(), nd_t
            nd_eff = torch.tensor(nd_t[None]) - (torch.tensor(c_act[None]) - curt).sum(-1)
            P = recursive_rayen(raw[..., :7], torch.tensor(pL[None]),
                                torch.tensor(pR[None]), nd_eff)
            return P[0].numpy(), curt[0].numpy(), nd_eff[0].numpy()

    def run(arm, mode, free):
        """One planA pass: build the mode's day, fill the window, project the day."""
        W = shift_world(free)
        g0 = free[0] * 12
        days_d, days_c = [], []
        raw_v = {"bal>1MW": 0, "ramp": 0, "neg": 0}
        prj_v = {"bal>1MW": 0, "ramp": 0, "neg": 0, "SOC": 0}
        E_b = np.zeros(6); E_c = np.zeros(6); Cb = np.zeros(2); Cc = np.zeros(2)
        credit = 0.0; dd = 0.0
        for d in pick:
            dr = np.where(te.index.normalize() == full[d])[0]
            rows = dr[g0:g0 + GAP]
            span = np.arange(rows[0] - CTX, rows[-1] + 1 + CTX)
            truth = te.iloc[dr][TARGETS].values.astype(np.float64)
            if mode == "scaled":
                day = truth * (W["nd_cf"][dr] / W["nd_base"][dr])[:, None]
            else:
                day = truth.copy()
            ctx = np.zeros((len(span), 6))
            for k, r in enumerate(span):
                pos = np.where(dr == r)[0]
                ctx[k] = day[pos[0]] if len(pos) else te.iloc[r][TARGETS].values
            Pb, cb, _ = fill(arm, rows, W, ctx, shifted=False)      # same mode, no shift
            Pc, cc, nd_eff = fill(arm, rows, W, ctx, shifted=True)
            credit = max(credit, float(np.abs(nd_eff - W["nd_cf"][rows]).max()))
            dd += W["d_dem"][rows].sum() * DT
            E_b += Pb.sum(0) * DT; E_c += Pc.sum(0) * DT
            Cb += cb.sum(0) * DT; Cc += cc.sum(0) * DT

            day[g0:g0 + GAP] = Pc
            nd_day = W["nd_cf"][dr].copy(); nd_day[g0:g0 + GAP] = nd_eff
            b = np.abs((day * SIGN).sum(-1) - nd_day)
            df = np.diff(day, axis=0)
            raw_v["bal>1MW"] += int((b[g0:g0 + GAP] > 1.0).sum())
            raw_v["ramp"] += int(((df > C.R_UP + 0.6) | (df < -(C.R_DN + 0.6))).sum())
            raw_v["neg"] += int((day < -0.1).sum())
            day = F.project_full_day(day, day[0].copy(), day[-1].copy(), nd_day, True)
            b = np.abs((day * SIGN).sum(-1) - nd_day)
            df = np.diff(day, axis=0)
            prj_v["bal>1MW"] += int((b > 1.0).sum())
            prj_v["ramp"] += int(((df > C.R_UP + 0.6) | (df < -(C.R_DN + 0.6))).sum())
            prj_v["neg"] += int((day < -0.1).sum())
            prj_v["SOC"] += int(C._soc_swing_mwh(day) > C.BATT_CAP_MWH + 1e-6)
            dayc = te.iloc[dr][["wind_curtailment", "solar_curtailment"]].values.copy()
            dayc[g0:g0 + GAP] = cc
            days_d.append(day); days_c.append(dayc)
        return dict(disp=np.concatenate(days_d), curt=np.concatenate(days_c),
                    raw_v=raw_v, prj_v=prj_v, resp=E_c - E_b, cresp=Cc - Cb,
                    Cb=Cb, credit=credit, dd=dd, W=W)

    rows_all = np.concatenate([np.where(te.index.normalize() == full[d])[0] for d in pick])
    act_disp = te.iloc[rows_all][TARGETS].values.astype(np.float64)
    act_curt = te.iloc[rows_all][["wind_curtailment", "solar_curtailment"]].values

    L = [f"# Counterfactual over {len(pick)} days — planA protocol", "",
         f"Days {full[pick[0]].date()} .. {full[pick[-1]].date()}. Free window "
         f"{F.FREE[0]}:00–{F.FREE[1]}:00, rebound {Q:g}% / reduce {Q:g}%, both modes, "
         f"context = the mode's own off-window dispatch, `project_full_day` over all "
         f"288 steps. Both models trained **1 epoch** and both predict wind and solar "
         f"curtailment.", ""]

    res = {}
    for mode in ("scaled", "masked"):
        g0 = F.FREE[0] * 12
        shade = [(o * 288 + g0, o * 288 + g0 + GAP) for o in range(len(pick))]
        for arm in ARMS:
            res[(mode, arm)] = run(arm, mode, F.FREE)
        W = res[(mode, "plain")]["W"]
        fig, ax = plt.subplots(3, 1, figsize=(2.0 * len(pick) + 3, 11),
                               sharex=True, sharey=True)
        draw(ax[0], te.iloc[rows_all], act_disp, act_curt,
             te.iloc[rows_all]["demand_mw"].values,
             te.iloc[rows_all]["net_demand"].values, shade,
             f"actual (shaded = the {F.FREE[0]}:00–{F.FREE[1]}:00 window the models fill)")
        for i, arm in enumerate(ARMS):
            r = res[(mode, arm)]
            draw(ax[i + 1], te.iloc[rows_all], r["disp"], r["curt"],
                 W["dem_s"][rows_all], W["ndf_cf"][rows_all], shade,
                 f"counterfactual, {mode} — {arm}"
                 + ("  (no constraints)" if arm == "plain" else "  (recursive RAYEN head)"))
        ax[2].set_xticks([k * 288 + 144 for k in range(len(pick))])
        ax[2].set_xticklabels([str(full[d].date())[5:] for d in pick], fontsize=8)
        ax[0].legend(loc="upper left", ncol=6, fontsize=7, frameon=False)
        fig.suptitle(f"planA counterfactual, {mode} — {len(pick)} days, "
                     f"1-epoch models with curtailment prediction",
                     fontsize=12.5, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        fig.savefig(OUT / f"curt_cf_{mode}.png", dpi=130, facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT/f'curt_cf_{mode}.png'}")

    L += ["## Feasibility", "",
          "Raw = the filled day straight out of the model. Projected = after planA's "
          "`project_full_day`. Balance is audited against the target each arm used.", "",
          "| mode | arm | raw bal>1MW | raw ramp | raw neg | projected bal/ramp/neg/SOC |",
          "| --- | --- | ---: | ---: | ---: | ---: |"]
    for mode in ("scaled", "masked"):
        for arm in ARMS:
            r = res[(mode, arm)]; a, b = r["raw_v"], r["prj_v"]
            L.append(f"| {mode} | {arm} | {a['bal>1MW']} | {a['ramp']} | {a['neg']} | "
                     f"{b['bal>1MW']} / {b['ramp']} / {b['neg']} / {b['SOC']} |")
    L += ["", f"_Window cells: {len(pick)*GAP} steps x 6 channels; full day "
          f"{len(pick)*288} steps._", ""]

    for mode in ("scaled", "masked"):
        L += [f"## Counterfactual response, {mode} (MWh in the window)", "",
              "| channel | plain | recursive | actual-grid evening turn-up |",
              "| --- | ---: | ---: | ---: |"]
        tot = {a: abs(float(res[(mode, a)]["resp"] @ SIGN)) for a in ARMS}
        for i, t_ in enumerate(TARGETS):
            cells = [f"{res[(mode,a)]['resp'][i]:+,.0f} "
                     f"({100*abs(res[(mode,a)]['resp'][i])/tot[a]:.0f}%)" for a in ARMS]
            L.append(f"| {LABEL[t_]} | {cells[0]} | {cells[1]} | {EMP[t_]:.1f}% |")
        L += [f"| **Σ signed** | {float(res[(mode,'plain')]['resp']@SIGN):+,.0f} | "
              f"{float(res[(mode,'recursive')]['resp']@SIGN):+,.0f} | — |",
              f"| _demand rise to be met_ | _{res[(mode,'plain')]['dd']:+,.0f}_ | "
              f"_{res[(mode,'recursive')]['dd']:+,.0f}_ | — |", ""]

    sel = rows_all[((te.index.hour >= F.FREE[0]) & (te.index.hour < F.FREE[1]))[rows_all]]
    act_c = te.iloc[sel][["wind_curtailment", "solar_curtailment"]].sum().values * DT
    L += ["## Curtailment (MWh in the window, masked mode)", "",
          "| channel | ACTUAL | plain base → cf | Δ | recursive base → cf | Δ |",
          "| --- | ---: | --- | ---: | --- | ---: |"]
    p_, r_ = res[("masked", "plain")], res[("masked", "recursive")]
    note_coupling = (
        "**The coupling suppressed the curtailment prediction.** The plain arm "
        f"predicts {p_['Cb'].sum():,.0f} MWh against an actual {act_c.sum():,.0f}; "
        f"the recursive arm only {r_['Cb'].sum():,.0f}. In the recursive arm "
        "curtailment feeds the balance target, so a large curtailment prediction "
        "immediately moves the dispatch it is scored on -- after one epoch the "
        "easiest way to avoid that penalty is to keep the prediction near zero. "
        "Making curtailment *mean* something also made it *harder to learn*, "
        "which is a real cost of the coupling.")
    for k, nm in enumerate(["wind curtailment", "solar curtailment"]):
        L.append(f"| {nm} | **{act_c[k]:,.0f}** | {p_['Cb'][k]:,.0f} → "
                 f"{p_['Cb'][k]+p_['cresp'][k]:,.0f} | {p_['cresp'][k]:+,.0f} | "
                 f"{r_['Cb'][k]:,.0f} → {r_['Cb'][k]+r_['cresp'][k]:,.0f} | "
                 f"{r_['cresp'][k]:+,.0f} |")
    L += ["", f"_Worst curtailment credit applied to the recursive balance target: "
          f"{r_['credit']:,.0f} MW._", "", note_coupling, ""]

    # ---------------- peak-fill extra ----------------
    PEAK = (17, 20)
    pf = {arm: run(arm, "masked", PEAK) for arm in ARMS}
    d = pick[int(np.argmax([te.iloc[np.where(te.index.normalize() == full[x])[0]]
                            ["demand_mw"].max() for x in pick]))]
    dr = np.where(te.index.normalize() == full[d])[0]
    g0p = PEAK[0] * 12
    win = np.arange(g0p - 24, g0p + GAP + 24)
    sub = te.iloc[dr[win]]; o = pick.index(d) * 288
    Wp = pf["plain"]["W"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    draw(ax[0], sub, sub[TARGETS].values.astype(np.float64),
         sub[["wind_curtailment", "solar_curtailment"]].values,
         sub["demand_mw"].values, sub["net_demand"].values, [(24, 24 + GAP)], "actual")
    for i, arm in enumerate(ARMS):
        draw(ax[i + 1], sub, pf[arm]["disp"][o + win], pf[arm]["curt"][o + win],
             Wp["dem_s"][dr[win]], Wp["ndf_cf"][dr[win]], [(24, 24 + GAP)],
             f"counterfactual — {arm}")
        ax[i + 1].set_ylabel("")
    for a in ax:
        a.set_xticks([0, 24, 24 + GAP, len(win) - 1])
        a.set_xticklabels(["15:00", "17:00", "20:00", "22:00"], fontsize=8)
    ax[0].legend(loc="upper left", ncol=3, fontsize=7, frameon=False)
    fig.suptitle(f"Peak-fill example — {full[d].date()}, planA masked protocol with the "
                 f"free window moved to the evening peak 17:00–20:00",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "curt_cf_peakfill.png", dpi=140, facecolor="white"); plt.close(fig)
    print(f"wrote {OUT/'curt_cf_peakfill.png'}")

    L += ["## Peak-fill extra — same protocol, window on the evening peak", "",
          "`dispatch_study/` shows the evening turn-up is 1,997 MW against 314 MW in "
          "the morning, and midday has no battery-discharge signal (corr +0.035 vs "
          "+0.287). Same planA masked protocol, window moved to 17:00–20:00.", "",
          "| channel | plain | recursive | actual-grid evening turn-up |",
          "| --- | ---: | ---: | ---: |"]
    tot = {a: abs(float(pf[a]["resp"] @ SIGN)) for a in ARMS}
    for i, t_ in enumerate(TARGETS):
        cells = [f"{pf[a]['resp'][i]:+,.0f} ({100*abs(pf[a]['resp'][i])/tot[a]:.0f}%)"
                 for a in ARMS]
        L.append(f"| {LABEL[t_]} | {cells[0]} | {cells[1]} | {EMP[t_]:.1f}% |")
    L += [f"| **Σ signed** | {float(pf['plain']['resp']@SIGN):+,.0f} | "
          f"{float(pf['recursive']['resp']@SIGN):+,.0f} | — |",
          f"| _demand rise to be met_ | _{pf['plain']['dd']:+,.0f}_ | "
          f"_{pf['recursive']['dd']:+,.0f}_ | — |", ""]

    (OUT / "curt_cf.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
