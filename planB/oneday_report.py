"""Deliverables for one-day (288-step) imputation with the trained `blackout` arm.

  1. EVALUATION, teacher-forced. Per-channel error rates and a stacked chart,
     actual against imputed. Both sides of the gap are observed, which is what
     makes this imputation rather than forecasting, so using them is the task
     definition and not an advantage handed to the model.

  2. COUNTERFACTUAL, recursive. Demand is raised and the same change is passed
     to net demand: nd_cf = nd_base + (demand_new - demand_old). There is no
     longer any true dispatch to feed back, so the model runs on its own output:
     day 1 with no dispatch context at all, each later day taking its left
     context from its own previous day. Stacked chart is baseline against
     counterfactual, since no actual exists for a scenario that did not happen.

  3. Constraint violations for both, with the actual dispatch as the reference row.

    python3 planB/oneday_report.py --days 60 --chart-days 7 --shift 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "imputation")); sys.path.insert(0, str(HERE))

from gap_data import TARGETS, SIGN                                     # noqa: E402
import constraints as C                                                # noqa: E402
from oneday_impute import impute, SHORT, DAY, FREE_HOURS                           # noqa: E402

OUT = HERE / "results"
STACK = [("coal_brown", "coal", "#A0661F"), ("hydro", "hydro", "#2277BB"),
         ("gas_ocgt", "gas OCGT", "#E9A020"), ("gas_steam", "gas steam", "#C0392B"),
         ("battery_discharging", "battery out", "#8E44AD")]
LOAD = ("battery_charging", "battery in (load)", "#6E6E6E")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1", "#fcfcfb"


def err_table(P, Y, label):
    e = np.abs(P - Y).reshape(-1, 6)
    den = np.abs(Y).reshape(-1, 6).sum(0)
    return (f"| {label} | " + " | ".join(f"{v:,.1f}" for v in e.mean(0)) +
            f" | **{e.mean():,.1f}** |",
            f"| {label} | " + " | ".join(f"{v:,.1f}" for v in
                                         100 * e.sum(0) / np.maximum(den, 1e-9)) + " | |")


def chain_swing(P, window):
    CHG, DIS = TARGETS.index("battery_charging"), TARGETS.index("battery_discharging")
    out = []
    for i in range(0, len(P), window):
        blk = P[i:i + window].reshape(-1, 6)
        dE = (blk[:, CHG] * C.ETA - blk[:, DIS] / C.ETA) * C.DT
        c = np.concatenate([[0.0], np.cumsum(dE)])
        out.append(c.max() - c.min())
    return max(out)


def violations(P, nd, label, window):
    flat = P.reshape(-1, 6)
    bal = np.abs(flat @ SIGN - nd.reshape(-1))
    d = np.diff(P, axis=1).reshape(-1, 6)
    ro = max((d - C.R_UP).max(), (-d - C.R_DN).max(), 0.0)
    sw = chain_swing(P, window)
    return (f"| {label} | {bal.max():,.2f} | {100 * (bal > 1).mean():.2f}% | "
            f"{ro:,.2f} | {max(0.0, -flat.min()):,.2f} | "
            f"{max(0.0, (flat - C.CAP).max()):,.2f} | {sw:,.0f} | "
            f"{'yes' if sw <= C.BATT_CAP_MWH + 1e-6 else 'NO'} |")


def stacked(ax, P, title, hours, shade_free=True):
    x = np.arange(P.shape[0]) / 12.0
    if shade_free:
        for d0 in range(0, int(hours), 24):
            ax.axvspan(d0 + FREE_HOURS[0], d0 + FREE_HOURS[1], color="#000000",
                       alpha=0.10, lw=0, zorder=1)
    base = np.zeros(len(x))
    for key, lbl, col in STACK:
        v = P[:, TARGETS.index(key)]
        ax.fill_between(x, base, base + v, color=col, lw=0, label=lbl, zorder=3)
        base = base + v
    ax.fill_between(x, 0, -P[:, TARGETS.index(LOAD[0])], color=LOAD[2], lw=0,
                    label=LOAD[1], zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=4)
    for dd in range(1, int(hours // 24) + 1):
        ax.axvline(dd * 24, color=INK2, lw=0.6, ls=(0, (3, 3)), zorder=5)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlim(0, hours); ax.set_ylabel("MW", fontsize=9, color=INK2)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def figure(top, bot, top_t, bot_t, stamps, suptitle, path, shade_free=True,
           tick=24):
    hours = top.shape[0] * DAY / 12.0
    A, B = top.reshape(-1, 6), bot.reshape(-1, 6)
    lo = min(-A[:, TARGETS.index(LOAD[0])].max(), -B[:, TARGETS.index(LOAD[0])].max())
    hi = max(sum(A[:, TARGETS.index(k)] for k, _, _ in STACK).max(),
             sum(B[:, TARGETS.index(k)] for k, _, _ in STACK).max())
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True)
    fig.patch.set_facecolor(SURF)
    for a in ax:
        a.set_facecolor(SURF)
    stacked(ax[0], A, top_t, hours, shade_free)
    stacked(ax[1], B, bot_t, hours, shade_free)
    if shade_free:
        ax[0].text(FREE_HOURS[0] + 0.2, ax[0].get_ylim()[1],
                   f"shaded = free window {FREE_HOURS[0]}:00–{FREE_HOURS[1]}:00",
                   fontsize=8, color=INK2, va="top")
    for a in ax:
        a.set_ylim(lo * 1.08, hi * 1.06)
    ax[1].set_xlabel("hours from start of the first imputed day", fontsize=9, color=INK2)
    ax[1].set_xticks(np.arange(0, hours + 1, tick))
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncols=6,
                 frameon=False, fontsize=8.5)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", x=0.008, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--chart-days", type=int, default=7)
    ap.add_argument("--soc-window", type=int, default=7)
    ap.add_argument("--zoom-days", type=int, default=2,
                    help="days shown in the zoomed panels")
    ap.add_argument("--rebound", type=float, default=2.4,
                    help="%% induced demand added inside the midday window")
    ap.add_argument("--reduction", type=float, default=2.4,
                    help="%% demand cut outside it and moved into the window")
    a = ap.parse_args()
    W = a.soc_window
    reb, red = a.rebound, a.reduction

    # ---- 1. evaluation: teacher-forced, real nd -----------------------------
    P, Y, st, _ = impute("tf", a.days, soc_window=W)
    mae_row, mre_row = err_table(P, Y, "one-day imputation (TF)")
    nd_real = (Y.reshape(-1, 6) @ SIGN)
    vio = [violations(Y, nd_real, "actual dispatch (reference)", W),
           violations(P, nd_real, "imputed (TF)", W)]
    Pc, Yc, stc, _ = impute("tf", a.chart_days, soc_window=W)
    figure(Yc, Pc, "Actual dispatch", "Imputed", stc,
           f"1. One-day imputation, {stc[0]} to {stc[-1]}",
           OUT / "oneday_eval_stack.png", shade_free=False)
    z = a.zoom_days
    figure(Yc[:z], Pc[:z], "Actual dispatch", "Imputed", stc[:z],
           f"1b. Zoom, {stc[0]} to {stc[z-1]}",
           OUT / "oneday_eval_zoom.png", shade_free=False, tick=6)

    # ---- 2. counterfactual: recursive, shifted nd ---------------------------
    def netE(A):
        f_ = A.reshape(-1, 6)
        return float(((f_[:, 4] * C.ETA - f_[:, 5] / C.ETA) * C.DT).sum())

    runs = {}
    for tag, fix in (("free", False), ("energy-conserved", True)):
        Pf, Yf, _, _ = impute("rec", a.days, soc_window=W, reduction_pct=red,
                                  rebound_pct=reb, energy_fix=fix)
        runs[tag] = (Pf, Yf)
        vio.append(violations(Pf, (Pf.reshape(-1, 6) @ SIGN),
                              f"counterfactual ({tag})", W))
    Pf, Yf = runs["energy-conserved"]
    Pfree = runs["free"][0]
    dE0 = netE(Yf)
    tot = {t: (Pf[..., i].sum() * C.DT, Yf[..., i].sum() * C.DT,
               Pfree[..., i].sum() * C.DT) for i, t in enumerate(TARGETS)}
    Pcf, Ycf, stcf, _ = impute("rec", a.chart_days, soc_window=W, reduction_pct=red,
                                      rebound_pct=reb, energy_fix=True)
    figure(Ycf, Pcf, "Actual dispatch (baseline)",
           f"Counterfactual: {red:g}% shifted into {FREE_HOURS[0]}:00-{FREE_HOURS[1]}:00, "
           f"+{reb:g}% rebound, recursive, energy-conserved", stcf,
           f"2. Counterfactual, {stcf[0]} to {stcf[-1]}",
           OUT / "oneday_cf_stack.png")
    figure(Ycf[:z], Pcf[:z], "Actual dispatch (baseline)",
           f"Counterfactual: {red:g}% shifted into {FREE_HOURS[0]}:00-{FREE_HOURS[1]}:00, "
           f"+{reb:g}% rebound, recursive, energy-conserved", stcf[:z],
           f"2b. Zoom, {stcf[0]} to {stcf[z-1]}",
           OUT / "oneday_cf_zoom.png", tick=6)

    hdr = "| arm | " + " | ".join(SHORT[c] for c in TARGETS) + " | **aggregate** |"
    sep = "| --- |" + " ---: |" * 7
    L = ["# One-day imputation and counterfactual — `blackout` (BiLSTM + hardnet)", "",
         f"{a.days} consecutive test days from {st[0]}. SOC reservoir bound carried "
         f"across days and restarted every {W} days: the actual dispatch needs a "
         f"4,891 MWh swing over 60 days against a 4,536 MWh target, so a 60-day "
         f"chain would declare 41% of real data infeasible; at 7 days the real "
         f"maximum is 4,062 MWh.", "",
         "---", "",
         "## 1. Evaluation — one-day imputation", "",
         "Teacher-forced. Both sides of the gap are observed, which is what makes "
         "this imputation rather than forecasting, so 4 h of real dispatch on each "
         "side is the task definition.", "",
         "### MAE (MW)", "", hdr, sep, mae_row, "",
         "### MRE (%)", "", hdr, sep, mre_row, "",
         "_MRE is unreliable where the truth is mostly zero; gas steam is >90% zero._", "",
         "![evaluation](oneday_eval_stack.png)", "",
         f"Zoomed to the first {a.zoom_days} days, ticks every 6 h:", "",
         "![evaluation zoom](oneday_eval_zoom.png)", "",
         "---", "",
         f"## 2. Counterfactual — load shift into {FREE_HOURS[0]}:00-{FREE_HOURS[1]}:00", "",
         "Recursive. Once demand moves, the dispatch that actually followed no "
         "longer applies, so nothing true can be fed back: day 1 runs with no "
         "dispatch context at all and each later day takes its left context from "
         "its own previous output. The subspace (net demand, demand, wind, solar, "
         "price, calendar) stays known throughout.", "",
         "Scenario definition:", "",
         "```",
         f"outside {FREE_HOURS[0]}:00-{FREE_HOURS[1]}:00 :  d_new = d * (1 - {red/100:.3f})",
         f"inside                :  d_new = d * (1 + {reb/100:.3f}) + (MWh removed that day) / n_free",
         "price inside the window forced to $0/MWh", "",
         "nd_cf = nd_base + (demand_new - demand_old)", "```", "",
         f"The {red:g}% cut outside the window is MOVED into it, so the day very "
         f"nearly conserves energy; the {reb:g}% rebound is the only genuinely "
         f"induced demand. This is lib/shift_model.py::FixedPercentageShift, the "
         f"same scenario planB/counterfactual.py uses.", "",
         "Renewables, curtailment and the interconnector are held fixed, so the "
         "whole change falls on the six dispatchable channels. `dispatch_study/"
         "midday_response_2026.md` measures the real grid serving about 30% of a "
         "midday rise by spilling less, so the dispatchable shares below are "
         "inflated relative to reality by roughly that much.", "",
         "### Energy totals over the window (GWh)", "",
         "Two runs. **free** is the model left alone. **energy-conserved** adds the "
         "constraint that the battery's total net energy over the whole run equals "
         "the baseline's, so the counterfactual cannot manufacture or destroy "
         "stored energy.", "",
         "| fuel | baseline | free | change | energy-conserved | change |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for t in TARGETS:
        cf, bl, fr = tot[t]
        L.append(f"| {SHORT[t]} | {bl/1e3:,.1f} | {fr/1e3:,.1f} | "
                 f"{100*(fr-bl)/max(bl,1e-9):+.1f}% | {cf/1e3:,.1f} | "
                 f"{100*(cf-bl)/max(bl,1e-9):+.1f}% |")
    dEf, dEc = netE(Pfree), netE(Pf)
    L += ["", f"_Battery net energy over {a.days} days: baseline {dE0:,.0f} MWh; "
              f"free {dEf:,.0f} MWh (drift {dEf-dE0:+,.0f}, "
              f"{(dEf-dE0)/a.days:+,.0f} MWh/day); energy-conserved {dEc:,.0f} MWh "
              f"(drift {dEc-dE0:+,.0f})._", "",
          "![counterfactual](oneday_cf_stack.png)", "",
          f"Zoomed to the first {a.zoom_days} days, ticks every 6 h. The shaded band is the free window:", "",
          "![counterfactual zoom](oneday_cf_zoom.png)", "",
          "---", "",
          "## 3. Constraint violations", "",
          "| arm | balance max (MW) | steps >1 MW | ramp overshoot (MW) | below zero (MW) "
          "| above cap (MW) | SOC swing (MWh) | SOC feasible |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *vio, "",
          "_Balance for rows 1-2 is against the real net demand; for the "
          "counterfactual it is against the scenario net demand, which is the "
          "quantity that run is required to serve. The max is a worst single step "
          "-- read it with the `steps >1 MW` column. In the recursive run the seams "
          "are the model's own output rather than observed dispatch, so on a few "
          "steps the ramp tube between them and the balance plane are "
          "incompatible._", "",
          f"_SOC swing is the worst {W}-day chain. Pack "
          f"{C.BATT_CAP_MWH:,.0f} MWh; the projection targets "
          f"{C.BATT_CAP_MWH-200:,.0f} MWh after a 100 MWh margin each side._", ""]
    (OUT / "oneday_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
