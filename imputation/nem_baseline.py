"""A NO-MODEL counterfactual filler built from measured NEM behaviour.

No neural network anywhere. Every number in the rule comes from dispatch_study/.
The point is to answer: if a dispatcher filled the planA free window by hand,
what would come out, and what would curtailment do?

THE CURTAILMENT QUESTION FIRST, because it drives the rest.
Measured over 2022-2026 (7,681,136 MWh of curtailment):

    price < 0        84.7% of all curtailment MWh   <- economic. Generators stop
                                                       because running costs them
                                                       money. RELEASABLE if demand
                                                       rises and price recovers.
    price >= 0       15.3%                          <- NOT explainable by price.
                                                       Network congestion, system
                                                       strength, local limits. This
                                                       does NOT go away when demand
                                                       rises, because more load in
                                                       Melbourne does not add
                                                       transmission capacity out of
                                                       western Victoria.
    nd quintile 1    85.4% of curtailment           <- it is a low-net-demand event
    nd quintile 5     1.0%  (mean 9.6 MW)           <- already ~zero when nd is high

So curtailment does NOT zero out. It falls toward a floor set by the network, and
that floor is roughly 15% of the historical level plus whatever the free window
was already carrying at non-negative prices.

THE FILL RULE
  1. Release curtailment: at steps priced below zero the curtailment is treated as
     economic and released, capped by the demand increase at that step (you cannot
     deliver more free energy than there is new load to absorb). At steps priced at
     or above zero nothing is released -- that curtailment is network-bound.
  2. The straight line between the pinned boundaries is the starting point for
     every channel -- the "no news" continuation. Nothing is pinned: an earlier
     version held coal fixed, which pushed the residual onto hydro (46% against a
     measured 29%). Study A says coal supplies 43% of a 3h change; that governs.
  3. The remaining residual is allocated across the fleet by the MEASURED 3-hour
     marginal-response slopes (dispatch_study Study A), then clipped to each
     channel's per-step ramp and headroom.
  4. Anything left unallocated because a channel hit a limit is redistributed to
     whatever still has room, so balance closes exactly.

    python3 imputation/nem_baseline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "dispatch_study"))

import plan1_figures as F                                              # noqa: E402
import constraints as C                                                # noqa: E402
from gap_data import TARGETS, SIGN                                     # noqa: E402
from shift_model import FixedPercentageShift                           # noqa: E402
from common import COLORS, LABEL, INK, MUTED, FUELS                    # noqa: E402
from curt_cf import draw                                               # noqa: E402

OUT = ROOT / "planA"
CTX, GAP, Q = F.CTX, F.GAP, F.Q
DT = 5.0 / 60.0
CURT = ["wind_curtailment", "solar_curtailment"]

# Measured 3-hour marginal response, dispatch_study/response.md Study A.
# MW of each fuel per MW of net-demand change over a 3h horizon.
SLOPE = {"coal_brown": 0.336, "hydro": 0.227, "gas_ocgt": 0.075,
         "gas_steam": 0.029, "battery_discharging": 0.057,
         "battery_charging": -0.051}
# Nothing is held. An earlier version pinned coal on the argument that a 3h price
# signal does not re-commit a brown-coal unit -- true for COMMITMENT, but the
# measurement disagrees for DISPATCH: Study A puts coal at 0.336 MW per MW of
# 3-hour net-demand change, 43% of the local fleet's response, and the units are
# already synchronised with ~1.8 GW of headroom at midday. Holding it pushed the
# residual onto hydro (46% against a measured 29%). The slopes are the evidence;
# the narrative was not.
HELD: set[str] = set()


def fill_window(pL, pR, cL, cR, nd_cf, d_dem, price):
    """No-model counterfactual fill for one window. Fills ALL EIGHT channels.

    pL/pR (6,) pinned dispatch boundaries, cL/cR (2,) pinned CURTAILMENT
    boundaries, nd_cf (G,) counterfactual balance target, d_dem (G,) demand
    increase, price (G,) price. Nothing inside the gap is read from the actual --
    curtailment is interpolated between its own boundaries exactly like dispatch,
    so this is a genuine gap fill and is comparable to the learned models.
    Returns (P (G,6), curt_base (G,2), curt_cf (G,2), unserved (G,)).
    """
    G = len(nd_cf)
    tt0 = (np.arange(1, G + 1) / (G + 1))[:, None]
    curt_base = cL[None] + tt0 * (cR - cL)[None]   # the FILLED base-case curtailment
    # ---- 1. curtailment release -------------------------------------------
    econ = (price < 0)[:, None]                    # economic <=> priced below zero
    releasable = curt_base * econ
    # cannot deliver more free energy than there is new load for it
    room = np.maximum(d_dem, 0.0)
    scale = np.where(releasable.sum(1) > 1e-9,
                     np.minimum(1.0, room / np.maximum(releasable.sum(1), 1e-9)), 0.0)
    released = releasable * scale[:, None]
    curt = curt_base - released

    # ---- 2. coal holds its committed line ----------------------------------
    tt = (np.arange(1, G + 1) / (G + 1))[:, None]
    base = pL[None] + tt * (pR - pL)[None]         # straight line for every channel

    # ---- 3. allocate the residual by measured slopes ----------------------
    # what the fuels must still supply, after the released renewables
    resid = nd_cf - released.sum(1) - base @ SIGN
    P = base.copy()
    w0 = np.array([SLOPE[t] for t in TARGETS])
    prev = pL.copy()
    unserved = np.zeros(G)
    for t in range(G):
        r = resid[t]
        for _ in range(4):                          # redistribute after clipping
            if abs(r) < 1e-6:
                break
            free = np.array([0.0 if x in HELD else 1.0 for x in TARGETS])
            # headroom in the direction each channel must move for this residual
            lo = np.maximum(0.0, prev - C.R_DN)
            hi = np.minimum(C.CAP, prev + C.R_UP)
            if t == G - 1:                          # must still reach pR next step
                lo = np.maximum(lo, pR - C.R_UP)
                hi = np.minimum(hi, pR + C.R_DN)
            hi = np.maximum(hi, lo)
            step = np.sign(r) * SIGN                # +1 where the channel must rise
            room_i = np.where(step > 0, hi - P[t], P[t] - lo) * free
            w = np.abs(w0) * (room_i > 1e-9)
            if w.sum() < 1e-9:
                break
            move = np.abs(r) * w / w.sum()
            move = np.minimum(move, room_i)
            P[t] = P[t] + step * move
            r = nd_cf[t] - released[t].sum() - P[t] @ SIGN
        unserved[t] = r
        P[t] = np.clip(P[t], np.maximum(0.0, prev - C.R_DN),
                       np.minimum(C.CAP, prev + C.R_UP))
        prev = P[t]
    return P, curt_base, curt, unserved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    args = ap.parse_args()
    te, feat_cols, xm, xs_, ym, ys_ = F.load_test()

    # ---- ISOLATED VICTORIA -------------------------------------------------
    # No interconnector, no import or export. Local supply must equal local
    # demand, so the demand line is DEFINED as what the local fleet served:
    #     demand_local = SIGN.dispatch + wind_delivered + solar_delivered
    # AEMO's operational demand_mw cannot be used: measured over 2022-2026 the
    # fleet produces 196.7 MW more than it on average (4.2% of demand, up to
    # 2,508 MW), which is real export. Under an isolated assumption that surplus
    # has nowhere to go, so it would sit in every figure as an unexplained gap.
    # Defining demand locally removes it and costs nothing elsewhere: the
    # constraint machinery already balances to SIGN.dispatch.
    te = te.copy()
    te["demand_local"] = (te[TARGETS].values @ SIGN) + te["wind"].values \
        + te["solar_utility"].values
    DEM = "demand_local"
    pick, full = F.pick_days(te, n=args.days)
    g0 = F.FREE[0] * 12
    print(f"days {full[pick[0]].date()} .. {full[pick[-1]].date()}, "
          f"window {F.FREE[0]}:00-{F.FREE[1]}:00")

    sh = FixedPercentageShift(Q, Q, free_hours=F.FREE).transform(
        pd.DataFrame({"demand_mw": te[DEM].values,
                      "price_aud_per_mwh": te["price_aud_per_mwh"].values},
                     index=te.index))
    dem_s = sh["demand_mw"].values                     # shifted LOCAL demand
    d_dem = dem_s - te[DEM].values
    nd_base = te[TARGETS].values @ SIGN
    nd_cf = nd_base + d_dem
    ndf_cf = te["net_demand"].values + d_dem

    days_d, days_c, days_r = [], [], []
    E_b = np.zeros(6); E_c = np.zeros(6); Cb = np.zeros(2); Cc = np.zeros(2)
    viol = {"bal>1MW": 0, "ramp": 0, "neg": 0}; uns = 0.0; dd = 0.0
    cerr = np.zeros(2); Cact = np.zeros(2); ncell = 0; closure = 0.0
    for d in pick:
        dr = np.where(te.index.normalize() == full[d])[0]
        rows = dr[g0:g0 + GAP]
        truth = te.iloc[dr][TARGETS].values.astype(np.float64)
        # planA "scaled" mode: the 2.4% reduction is applied on BOTH sides of the
        # window by scaling the off-window dispatch to the shifted net demand.
        # Scaling preserves balance exactly (SIGN.(k*P) = k*nd), so the off-window
        # needs no repair.
        scaled = truth * (nd_cf[dr] / nd_base[dr])[:, None]
        pL, pR = scaled[g0 - 1].copy(), scaled[g0 + GAP].copy()
        cw = te.iloc[rows][CURT].values.astype(np.float64)       # actual, for scoring only
        cL = te.iloc[dr[g0 - 1]][CURT].values.astype(np.float64)
        cR = te.iloc[dr[g0 + GAP]][CURT].values.astype(np.float64)
        pr = te.iloc[rows]["price_aud_per_mwh"].values
        P, curt_b, curt, u = fill_window(pL, pR, cL, cR, nd_cf[rows], d_dem[rows], pr)
        released = curt_b - curt                       # moves into DELIVERED renewables
        cerr += np.abs(curt_b - cw).sum(0)                        # fill error vs truth
        day = scaled.copy(); day[g0:g0 + GAP] = P
        dayc = te.iloc[dr][CURT].values.copy(); dayc[g0:g0 + GAP] = curt
        dayr = te.iloc[dr][["wind", "solar_utility"]].values.astype(np.float64).copy()
        dayr[g0:g0 + GAP] += released                  # conserve total available energy
        days_d.append(day); days_c.append(dayc); days_r.append(dayr)
        E_b += truth[g0:g0 + GAP].sum(0) * DT; E_c += P.sum(0) * DT
        Cb += curt_b.sum(0) * DT; Cc += curt.sum(0) * DT
        Cact += cw.sum(0) * DT; ncell += GAP
        dd += d_dem[rows].sum() * DT; uns += np.abs(u).sum() * DT
        b = np.abs(P @ SIGN + (curt_b - curt).sum(1) - nd_cf[rows])
        # isolated-VIC closure: local supply must equal local demand exactly
        wsol = te.iloc[rows][["wind", "solar_utility"]].values.astype(np.float64)
        close = np.abs(P @ SIGN + (wsol + released).sum(1) - dem_s[rows])
        closure = max(closure, float(close.max()))
        fp = np.vstack([pL[None], P, pR[None]]); df = np.diff(fp, axis=0)
        viol["bal>1MW"] += int((b > 1.0).sum())
        viol["ramp"] += int(((df > C.R_UP + 0.6) | (df < -(C.R_DN + 0.6))).sum())
        viol["neg"] += int((P < -0.1).sum())

    rows_all = np.concatenate([np.where(te.index.normalize() == full[d])[0] for d in pick])
    shade = [(o * 288 + g0, o * 288 + g0 + GAP) for o in range(len(pick))]
    fig, ax = plt.subplots(2, 1, figsize=(2.0 * len(pick) + 3, 8), sharex=True, sharey=True)
    draw(ax[0], te.iloc[rows_all], te.iloc[rows_all][TARGETS].values.astype(np.float64),
         te.iloc[rows_all][CURT].values, te.iloc[rows_all][DEM].values,
         te.iloc[rows_all][TARGETS].values @ SIGN, shade,
         f"actual (shaded = the {F.FREE[0]}:00–{F.FREE[1]}:00 free window)")
    draw(ax[1], te.iloc[rows_all], np.concatenate(days_d), np.concatenate(days_c),
         dem_s[rows_all], nd_cf[rows_all], shade, renew=np.concatenate(days_r),
         title=
         f"NO-MODEL counterfactual — off-window scaled to the reduced demand; in the "
         f"window, curtailment released only where price < 0 (released energy moves "
         f"into the solid wind/solar band) and the residual allocated by measured 3h "
         f"slopes  (rebound {Q:g}%, reduce {Q:g}%)")
    ax[1].set_xticks([k * 288 + 144 for k in range(len(pick))])
    ax[1].set_xticklabels([str(full[d].date())[5:] for d in pick], fontsize=8)
    ax[0].legend(loc="upper left", ncol=6, fontsize=7, frameon=False)
    fig.suptitle("No-model NEM-style counterfactual fill — every rule measured, "
                 "no neural network", fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / "nem_baseline.png", dpi=130, facecolor="white"); plt.close(fig)
    print(f"wrote {OUT/'nem_baseline.png'}")

    # ---- zoom: 3 days at a scale where the window and its curtailment are visible
    z = pick[:3]
    zi = np.concatenate([np.where(te.index.normalize() == full[d])[0] for d in z])
    zsh = [(o * 288 + g0, o * 288 + g0 + GAP) for o in range(len(z))]
    off = [pick.index(d) * 288 for d in z]
    zd = np.concatenate([days_d[pick.index(d)] for d in z])
    zc = np.concatenate([days_c[pick.index(d)] for d in z])
    zr = np.concatenate([days_r[pick.index(d)] for d in z])
    fig, ax = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True, sharey=True)
    draw(ax[0], te.iloc[zi], te.iloc[zi][TARGETS].values.astype(np.float64),
         te.iloc[zi][CURT].values, te.iloc[zi][DEM].values,
         te.iloc[zi][TARGETS].values @ SIGN, zsh, "actual")
    draw(ax[1], te.iloc[zi], zd, zc, dem_s[zi], nd_cf[zi], zsh,
         "no-model counterfactual — inside the shaded window BOTH dispatch and "
         "curtailment are filled, and released curtailment moves into the solid "
         "wind/solar band; outside it the day is scaled to the reduced demand",
         renew=zr)
    ax[1].set_xticks([k * 288 + h * 12 for k in range(len(z)) for h in (0, 11, 14)])
    ax[1].set_xticklabels([lbl for k in range(len(z))
                           for lbl in (str(full[z[k]].date())[5:], "11", "14")], fontsize=7.5)
    ax[0].legend(loc="upper left", ncol=6, fontsize=7.5, frameon=False)
    fig.suptitle("No-model fill, zoomed to 3 days — the 11:00-14:00 window and its "
                 "released curtailment", fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / "nem_baseline_zoom.png", dpi=140, facecolor="white"); plt.close(fig)
    print(f"wrote {OUT/'nem_baseline_zoom.png'}")

    resp = E_c - E_b
    EMP = {"coal_brown": 43.4, "hydro": 29.3, "gas_ocgt": 9.7, "gas_steam": 3.7,
           "battery_discharging": 7.4, "battery_charging": 6.6}
    L = ["# No-model counterfactual fill (NEM-style)", "",
         f"{len(pick)} days {full[pick[0]].date()} .. {full[pick[-1]].date()}, window "
         f"{F.FREE[0]}:00–{F.FREE[1]}:00, rebound {Q:g}% / reduce {Q:g}%. "
         "No neural network. Every rule is measured in `dispatch_study/`.", "",
         "## Curtailment — filled, then partially released", "",
         "Curtailment inside the gap is UNKNOWN, so it is filled by interpolating "
         "between its own pinned boundaries, exactly like dispatch. Nothing inside "
         "the window is read from the actual. The base-case fill is then scored "
         "against the truth, and the counterfactual releases only the part priced "
         "below zero.", "",
         "| | wind | solar | total |", "| --- | ---: | ---: | ---: |",
         f"| ACTUAL in window (MWh) | {Cact[0]:,.0f} | {Cact[1]:,.0f} | {Cact.sum():,.0f} |",
         f"| base-case FILL (MWh) | {Cb[0]:,.0f} | {Cb[1]:,.0f} | {Cb.sum():,.0f} |",
         f"| fill MAE vs actual (MW) | {cerr[0]/ncell:,.1f} | {cerr[1]/ncell:,.1f} | "
         f"{cerr.sum()/(2*ncell):,.1f} |",
         f"| counterfactual (MWh) | {Cc[0]:,.0f} | {Cc[1]:,.0f} | {Cc.sum():,.0f} |",
         f"| released by the rebound | {Cb[0]-Cc[0]:,.0f} | {Cb[1]-Cc[1]:,.0f} | "
         f"{Cb.sum()-Cc.sum():,.0f} |",
         f"| **retained** | **{Cc[0]:,.0f}** | **{Cc[1]:,.0f}** | "
         f"**{100*Cc.sum()/max(Cb.sum(),1e-9):.0f}% of the fill** |", "",
         "The retained part is curtailment at non-negative prices — network "
         "congestion and system-strength limits, which extra Melbourne load does not "
         "relieve. Measured over 2022–2026 that is 15.3% of all curtailment MWh.", "",
         "## Dispatch response (MWh in the window)", "",
         "| channel | no-model fill | share | measured 3h grid response |",
         "| --- | ---: | ---: | ---: |"]
    tot = abs(float(resp @ SIGN))
    for i, t_ in enumerate(TARGETS):
        L.append(f"| {LABEL[t_]} | {resp[i]:+,.0f} | {100*abs(resp[i])/tot:.1f}% | "
                 f"{EMP[t_]:.1f}% |")
    L += [f"| **Σ signed** | {float(resp@SIGN):+,.0f} | | |",
          f"| _demand rise_ | _{dd:+,.0f}_ | | |",
          f"| _met by released curtailment_ | _{Cb.sum()-Cc.sum():+,.0f}_ | | |", "",
          "## Feasibility", "",
          f"balance >1 MW: {viol['bal>1MW']} | ramp: {viol['ramp']} | "
          f"negative: {viol['neg']} | unserved energy: {uns:,.0f} MWh", "",
          "**Isolated-Victoria closure.** No interconnector, so local supply must "
          "equal local demand at every step:", "",
          "```",
          "SIGN.dispatch + wind_delivered + solar_delivered  =  demand_local",
          f"worst residual over all {len(pick)*GAP} window steps: {closure:.3e} MW",
          "```", "",
          "`demand_local` is defined as the local fleet's output rather than taken "
          "from AEMO's `demand_mw`, which the fleet exceeds by 196.7 MW on average "
          "(4.2% of demand, up to 2,508 MW). That surplus is export; with export "
          "forbidden it has nowhere to go, so it would otherwise sit in every figure "
          "as an unexplained gap. Curtailment is still drawn above the demand line — "
          "it was available and was not delivered.", ""]
    (OUT / "nem_baseline.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
