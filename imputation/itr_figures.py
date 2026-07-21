"""D2 + D3 for the transformer-imputer ablation.

D2  4-day stacked dispatch figure, blackout mode:
      actual                    the real dispatch
      causal closed-loop        12h chunks imputed left-to-right, each chunk's fill
                                fed forward as the next chunk's observed context
                                (the information set of a causal forecaster; drift
                                and the reconnection seam are the point), raw
      best constrained          each day imputed as a full blackout and passed
                                through the stage-2 winner's guarantee map,
                                pinned to the known real day-edge boundaries

D3  counterfactual change report at q* (default 3.699%%, the all-days
    dispatch-feasible equal shift -- imputation/shift_feasibility.py):
      FixedPercentageShift(q*, q*): off-window demand cut uniformly, the energy
      (+ a q* rebound) lands in the 11:00-14:00 free window, free-window price
      -> 0, renewables fixed so Δnet_demand = Δdemand. Whole-day BLACKOUT
      imputation, Option A: NO dispatch anywhere in the input and NO pinned
      boundaries -- the guarantee map is the free-endpoint projection. Base and
      shifted runs differ only in the drivers. Report: ΔE_i per fuel split
      free/off-window, peak changes, violations (must be 0), and the consistency
      check Σ_i SIGN_i·ΔE_i ≈ shifted net-demand energy; 3-panel figure.

    python3 imputation/itr_figures.py --which d2
    python3 imputation/itr_figures.py --which d3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
from gap_data import load_flats, TARGETS, SIGN, ND_COL, DEM_COL, PRICE_COL   # noqa: E402
import constraints as C                                                      # noqa: E402
import itr_data as D                                                         # noqa: E402
from itr_model import ITransformerImputer, apply_gap_map                     # noqa: E402
from scenario_eval import build_shift_scenario, SHIFT_Q                      # noqa: E402
import stack_plots as sp                                                     # noqa: E402

OUT = HERE / "results" / "itr"
FIG = OUT / "figure"
STACK = ["coal_brown", "gas_steam", "gas_ocgt", "hydro", "battery_discharging"]
FREE_LO, FREE_HI = D.FREE_W


def load_winner(smoke):
    sfx = "_smoke" if smoke else ""
    sel_p = OUT / f"stage2_selection{sfx}.json"
    if not sel_p.exists():
        raise SystemExit("run itr_bench.py --stage2 first (needs stage2_selection.json)")
    sel = json.loads(sel_p.read_text())
    m = ITransformerImputer()
    m.load_state_dict(torch.load(OUT / f"{sel['ckpt_stem']}.pt",
                                 map_location="cpu", weights_only=True))
    m.eval()
    return m, sel


def impute_day(model, f, s, drivers=None, blackout=True):
    """One 288-step window from flat row s -> raw (T,6) MW fill. `drivers`
    optionally overrides (nd_mw, dem_mw, zero_price_mask) for the counterfactual."""
    xs = f.Xte[s:s + D.T].copy()
    if drivers is not None:
        nd_mw, dem_mw, zp = drivers
        xs[:, ND_COL] = (nd_mw - f.x_mean[ND_COL]) / f.x_scale[ND_COL]
        xs[:, DEM_COL] = (dem_mw - f.x_mean[DEM_COL]) / f.x_scale[DEM_COL]
        xs[zp, PRICE_COL] = (0.0 - f.x_mean[PRICE_COL]) / f.x_scale[PRICE_COL]
    m = np.ones((D.T, 1), np.float32)
    if blackout:
        m[:] = 0.0
        xs[:, D._TFI] = 0.0
    with torch.no_grad():
        out = model(torch.from_numpy(xs[None].astype(np.float32)),
                    torch.from_numpy(m[None]))[0].numpy().astype(np.float64)
    return out * f.y_scale + f.y_mean


def pick_run(f, days, rec, n_days):
    """Highest-median-demand run of n_days consecutive test days (visibility)."""
    dem = f.col_mw(f.Xte, DEM_COL)
    best, best_v = None, -1.0
    for i in range(len(days) - n_days + 1):
        if (days[i + n_days - 1] - days[i]).days != n_days - 1:
            continue
        v = float(np.median([dem[rec["s"][i + k]: rec["s"][i + k] + D.T].mean()
                             for k in range(n_days)]))
        if v > best_v:
            best, best_v = i, v
    if best is None:
        raise SystemExit(f"no {n_days} consecutive test days")
    return best


def draw_stack(ax, disp, nd, title):
    t = np.arange(len(disp))
    ti = {s: i for i, s in enumerate(TARGETS)}
    base = np.zeros(len(t))
    for k in STACK:
        top = base + disp[:, ti[k]]
        ax.fill_between(t, base, top, color=sp.COLORS[k], alpha=0.9, label=k)
        base = top
    ax.fill_between(t, 0, -disp[:, ti["battery_charging"]], color=sp.COLORS["battery_charging"],
                    alpha=0.6, label="battery_charging (load)")
    ax.plot(t, nd, "k--", lw=1.0, label="net demand")
    for d0 in range(0, len(t), D.T):                       # shade each day's free window
        ax.axvspan(d0 + FREE_LO * 12, d0 + FREE_HI * 12, color="gold", alpha=0.15)
    ax.axhline(0, color="k", lw=0.6); ax.margins(x=0); ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(title, loc="left", fontsize=10); ax.set_ylabel("MW")


def d2(f, model, sel, n_days, smoke):
    rec, days = D.day_windows(f)
    i0 = pick_run(f, days, rec, n_days)
    ss = rec["s"][i0:i0 + n_days]
    disp = np.concatenate([f.y_to_mw(f.Yte[s:s + D.T]) for s in ss]).astype(np.float64)
    nd = np.concatenate([f.col_mw(f.Xte, ND_COL)[s:s + D.T] for s in ss]).astype(np.float64)

    # causal closed-loop: 144-step chunks; context = previous fill (or real, chunk 0)
    t0 = ss[0]
    state = f.y_to_mw(f.Yte[t0 - 144:t0]).astype(np.float64)         # real warmup context
    causal = []
    for k in range(n_days * 2):
        c0 = t0 + 144 * k
        xs = f.Xte[c0 - 144:c0 + 144].copy()
        xs[:144, D._TFI] = (state - f.x_mean[D._TFI]) / f.x_scale[D._TFI]
        xs[144:, D._TFI] = 0.0
        m = np.ones((288, 1), np.float32); m[144:] = 0.0
        with torch.no_grad():
            o = model(torch.from_numpy(xs[None].astype(np.float32)),
                      torch.from_numpy(m[None]))[0, 144:].numpy().astype(np.float64)
        state = o * f.y_scale + f.y_mean
        causal.append(state)
    causal = np.concatenate(causal)

    # best constrained: per-day blackout, guarantee map pinned to real day edges
    cons = []
    for s in ss:
        raw = impute_day(model, f, s)[None]
        b_nd = f.col_mw(f.Xte, ND_COL)[s:s + D.T][None].astype(np.float64)
        pL = f.y_to_mw(f.Yte[s - 1])[None]; pR = f.y_to_mw(f.Yte[s + D.T])[None]
        cons.append(apply_gap_map(sel["map"], raw, np.zeros(1, int), D.T, pL, pR, b_nd)[0])
    cons = np.concatenate(cons)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, sharey=True)
    draw_stack(axes[0], disp, nd, "actual (real dispatch)")
    draw_stack(axes[1], causal, nd, "causal closed-loop (12h chunks fed forward, raw — "
                                    "drift & seams are the point)")
    draw_stack(axes[2], cons, nd, f"best constrained ({sel['best_guaranteed']}) — "
                                  "per-day blackout imputation, guaranteed feasible")
    ticks = [k * D.T for k in range(n_days)]
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels([days[i0 + k].strftime("%b %d") for k in range(n_days)])
    axes[2].set_xlabel(f"{n_days} consecutive test days (free window 11:00–14:00 shaded)")
    axes[0].legend(loc="upper left", ncol=4, fontsize=7, framealpha=0.9)
    fig.suptitle(f"D2 — blackout-mode dispatch, {days[i0].date()}–{days[i0+n_days-1].date()}")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"d2_stack_{n_days}day{'_smoke' if smoke else ''}.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)
    return i0


def d3(f, model, sel, q, n_days, smoke, max_days=None):
    rec, days = D.day_windows(f)
    scen = build_shift_scenario(f, q)                      # (dem_s, nd_s, P_ref) flat-aligned
    dem_s, nd_s, _ = scen
    dem_b = f.col_mw(f.Xte, DEM_COL).astype(np.float64)
    nd_b = f.col_mw(f.Xte, ND_COL).astype(np.float64)
    hour = np.floor(12 * ((np.arange(D.T)) / 12) / 12).astype(int)   # 0..23 per step-in-day
    hour = (np.arange(D.T) // 12)
    free = (hour >= FREE_LO) & (hour < FREE_HI)

    keep = range(len(days)) if max_days is None else range(min(max_days, len(days)))
    dE_free = np.zeros(6); dE_off = np.zeros(6)
    peak_b = np.zeros(6); peak_s = np.zeros(6)
    viol = {"bal_steps": 0, "ramp_cells": 0, "neg_cells": 0, "soc_days": 0}
    dnd_free = dnd_off = 0.0
    kept = []
    for i in keep:
        s = rec["s"][i]
        rows = np.arange(s, s + D.T)
        Pb = impute_day(model, f, s)[None]                 # base: real drivers
        Ps = impute_day(model, f, s, drivers=(nd_s[rows], dem_s[rows], free))[None]
        # Option A deploy: FREE-endpoint guarantee map for both runs (no pins anywhere)
        Pb = apply_gap_map("free", Pb, np.zeros(1, int), D.T,
                           Pb[:, 0], Pb[:, -1], nd_b[rows][None])[0]
        Ps = apply_gap_map("free", Ps, np.zeros(1, int), D.T,
                           Ps[:, 0], Ps[:, -1], nd_s[rows][None])[0]
        kept.append((rows, Pb, Ps))
        dE_free += (Ps[free] - Pb[free]).sum(0) / 12.0     # MWh
        dE_off += (Ps[~free] - Pb[~free]).sum(0) / 12.0
        peak_b = np.maximum(peak_b, Pb.max(0)); peak_s = np.maximum(peak_s, Ps.max(0))
        dnd_free += (nd_s[rows][free] - nd_b[rows][free]).sum() / 12.0
        dnd_off += (nd_s[rows][~free] - nd_b[rows][~free]).sum() / 12.0
        bal = np.abs((Ps * SIGN).sum(1) - nd_s[rows])
        viol["bal_steps"] += int((bal > 1.0).sum())
        d = np.diff(Ps, axis=0)
        viol["ramp_cells"] += int(((d > C.R_UP + 0.6) | (d < -(C.R_DN + 0.6))).sum())
        viol["neg_cells"] += int((Ps < -0.1).sum())
        viol["soc_days"] += int(C._soc_swing_mwh(Ps) > C.BATT_CAP_MWH + 1e-6)

    dE_sign_free = float(dE_free @ SIGN); dE_sign_off = float(dE_off @ SIGN)
    lines = [f"# D3 — counterfactual change report, FixedPercentageShift(q={q:g}%)", "",
             f"{len(kept)} test days, whole-day blackout imputation, Option A "
             f"(no pinned dispatch anywhere), guarantee map = free-endpoint Π on "
             f"{sel['best_guaranteed']}.", "",
             "| fuel | ΔE free-window (MWh) | ΔE off-window (MWh) | peak base→shift (MW) |",
             "| --- | --- | --- | --- |"]
    for i, t in enumerate(TARGETS):
        lines.append(f"| {t} | {dE_free[i]:+.0f} | {dE_off[i]:+.0f} "
                     f"| {peak_b[i]:.0f} → {peak_s[i]:.0f} |")
    lines += ["",
              f"**Violations (must be 0):** bal>1MW steps {viol['bal_steps']}, ramp cells "
              f"{viol['ramp_cells']}, neg cells {viol['neg_cells']}, SOC days {viol['soc_days']}.",
              "",
              "**Consistency (Σ SIGN·ΔE vs shifted net-demand energy):**",
              f"free-window: dispatch {dE_sign_free:+.0f} MWh vs demand {dnd_free:+.0f} MWh "
              f"(gap {abs(dE_sign_free-dnd_free):.0f});  off-window: {dE_sign_off:+.0f} vs "
              f"{dnd_off:+.0f} MWh (gap {abs(dE_sign_off-dnd_off):.0f})."]
    sfx = "_smoke" if smoke else ""
    (OUT / f"d3_report{sfx}.md").write_text("\n".join(lines) + "\n")
    (OUT / f"d3_report{sfx}.json").write_text(json.dumps(
        {"q": q, "n_days": len(kept), "dE_free_mwh": dE_free.tolist(),
         "dE_off_mwh": dE_off.tolist(), "targets": TARGETS, "violations": viol,
         "consistency_free_mwh": [dE_sign_free, dnd_free],
         "consistency_off_mwh": [dE_sign_off, dnd_off]}, indent=2))
    print("\n".join(lines))

    # 3-panel figure on the D2 span (or the first n_days kept)
    i0 = pick_run(f, days, rec, n_days) if max_days is None else 0
    span = [k for k in range(i0, i0 + n_days) if k < len(kept)]
    if span:
        Pb = np.concatenate([kept[k][1] for k in span])
        Ps = np.concatenate([kept[k][2] for k in span])
        ndb = np.concatenate([nd_b[kept[k][0]] for k in span])
        nds = np.concatenate([nd_s[kept[k][0]] for k in span])
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        draw_stack(axes[0], Pb, ndb, "base — blackout fill, real drivers")
        draw_stack(axes[1], Ps, nds, f"counterfactual — shift q={q:g}% "
                                     "(off-window cut, free-window rebound, price→0)")
        t = np.arange(len(Pb))
        for i, tg in enumerate(TARGETS):
            axes[2].plot(t, Ps[:, i] - Pb[:, i], lw=0.9, label=tg,
                         color=sp.COLORS.get(tg, None))
        for d0 in range(0, len(t), D.T):
            axes[2].axvspan(d0 + FREE_LO * 12, d0 + FREE_HI * 12, color="gold", alpha=0.15)
        axes[2].axhline(0, color="k", lw=0.6); axes[2].margins(x=0)
        axes[2].grid(True, axis="y", alpha=0.3)
        axes[2].set_title("Δ dispatch (counterfactual − base) per fuel", loc="left", fontsize=10)
        axes[2].set_ylabel("MW"); axes[2].legend(ncol=3, fontsize=7)
        fig.suptitle(f"D3 — counterfactual at q={q:g}%, {len(span)} days")
        fig.tight_layout()
        FIG.mkdir(parents=True, exist_ok=True)
        p = FIG / f"d3_counterfactual{sfx}.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["d2", "d3", "both"], default="both")
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--q", type=float, default=SHIFT_Q,
                    help="equal shift %% (default 3.699 = all-days feasible ceiling)")
    ap.add_argument("--smoke", action="store_true",
                    help="use the _smoke selection/ckpts and 12 D3 days")
    args = ap.parse_args()
    f = load_flats()
    model, sel = load_winner(args.smoke)
    print(f"winner: {sel['best_guaranteed']} (map={sel['map']}, ckpt={sel['ckpt_stem']})")
    if args.which in ("d2", "both"):
        d2(f, model, sel, args.days, args.smoke)
    if args.which in ("d3", "both"):
        d3(f, model, sel, args.q, args.days, args.smoke,
           max_days=12 if args.smoke else None)


if __name__ == "__main__":
    main()
