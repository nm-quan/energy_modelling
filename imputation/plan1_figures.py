"""Plan1 deliverables (plan1.md S5-S6).

A   per arm: stacked ACTUAL vs PREDICTED over 4 random test days, one random 3h
    gap per day, the predicted (gap) segment highlighted. figA_<arm>.png
B   counterfactual, rebound 2% / reduce 2%, free window 11:00-14:00, price->0:
      cf_actual.png          actual stack (shared)
      cf_scaled_<arm>.png    off-window dispatch scaled by nd_after/nd_before,
                             free window masked + model-filled
      cf_masked_<arm>.png    off-window stays actual, free window model-filled
    nd bookkeeping per plan1.md S1/S6: the model SEES the demand-side nd feature
    (computed with ACTUAL curtailment; curt inputs -> 0 in the window); the MAP
    balances to the supply-side base + deltas (off: +Ddemand, free: +Ddemand - curt).

    python3 imputation/plan1_figures.py --which a b        # full weights
    python3 imputation/plan1_figures.py --smoke            # smoke weights sanity
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "lib"))
import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "prepared.npz"
from gap_data import TARGETS, SIGN, TARGET_FEAT_IDX                    # noqa: E402
import constraints as C                                                # noqa: E402
from model import BiLSTMImputer                                        # noqa: E402
from constraint_layers import rayen_traj_project                       # noqa: E402
from shift_model import FixedPercentageShift                           # noqa: E402

OUT = HERE / "results" / "plan1"
TABLE = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "table.parquet"
ARMS = ["baseline", "cost", "size_aware"]
CTX, GAP = 48, 36
FREE = (11, 14)
Q = 2.0
COLORS = {"coal_brown": "saddlebrown", "gas_steam": "#d62728", "gas_ocgt": "#ff7f0e",
          "hydro": "royalblue", "battery_discharging": "#9467bd", "battery_charging": "dimgray"}
WIND, SOLAR = "#2e8b40", "#f4c20d"
STACK = ["coal_brown", "gas_steam", "gas_ocgt", "hydro", "battery_discharging"]


def load_test():
    t = pd.read_parquet(TABLE)
    te = t[t.index > pd.Timestamp("2025-12-31 23:55:00+10:00")].copy()
    z = np.load(GD.NPZ, allow_pickle=False)
    feat_cols = [str(c) for c in z["feat_cols"]]
    return te, feat_cols, z["x_mean"], z["x_scale"], z["y_mean"], z["y_scale"]


def model_fill(model, te, feat_cols, xm, xs_, ym, ys_, rows, overrides=None,
               ctx_dispatch=None):
    """Raw MW fill for gap `rows` (window-relative CTX..CTX+GAP). overrides:
    {col: series over the window span} applied before scaling; ctx_dispatch:
    (W,6) MW dispatch to place in the context rows (scaled world)."""
    span = np.arange(rows[0] - CTX, rows[-1] + 1 + CTX)
    vals = te.iloc[span][feat_cols].values.astype(np.float64).copy()
    if overrides:
        for col, series in overrides.items():
            vals[:, feat_cols.index(col)] = series
    if ctx_dispatch is not None:
        for k, tgt in enumerate(TARGETS):
            vals[:, feat_cols.index(tgt)] = ctx_dispatch[:, k]
    X = ((vals - xm) / xs_).astype(np.float32)
    m = np.ones((len(span), 1), np.float32); m[CTX:CTX + GAP] = 0.0
    X[CTX:CTX + GAP, np.asarray(TARGET_FEAT_IDX)] = 0.0
    with torch.no_grad():
        dev = model(torch.from_numpy(X[None]), torch.from_numpy(m[None])
                    )[0, CTX:CTX + GAP].numpy().astype(np.float64)
    pL = vals[CTX - 1, [feat_cols.index(t) for t in TARGETS]]
    pR = vals[CTX + GAP, [feat_cols.index(t) for t in TARGETS]]
    tt = (np.arange(1, GAP + 1) / (GAP + 1))[:, None]
    interp = pL[None] + tt * (pR - pL)[None]
    # dev is the y-scaled deviation; interp is already MW, so fill = interp + dev*scale
    # (equivalent to the training-time (interp_s + dev)*scale + mean)
    return interp + dev * ys_, pL, pR


def rayen(fill, pL, pR, nd, size_aware):
    # counterfactual free-window nd carries the curtailment credit (a ~GW-scale
    # drop at the seam), so the POCS anchor needs many more cycles than the
    # actual-data case -- 400/240 drives the residual to ~0.
    with torch.no_grad():
        return rayen_traj_project(torch.tensor(fill[None]), torch.tensor(pL[None]),
                                  torch.tensor(pR[None]), torch.tensor(nd[None]),
                                  anchor_iters=400 if size_aware else 240,
                                  size_aware=size_aware)[0].numpy()


def draw_stack(ax, te_rows, disp, dem_line, shade=None, title="", curt_solid=None):
    """disp (n,6) MW TARGETS order; renewables/curtailment actual from te_rows.
    curt_solid: optional (n,) bool -- rows where curtailment is DELIVERED under the
    counterfactual credit; drawn solid there (hatched = spilled, elsewhere)."""
    x = np.arange(len(disp))
    ti = {t: i for i, t in enumerate(TARGETS)}
    base = np.zeros(len(x))
    for k in STACK:
        top = base + disp[:, ti[k]]
        ax.fill_between(x, base, top, color=COLORS[k], alpha=0.9, label=k, lw=0)
        base = top
    for col, c_, lab in [("wind", WIND, "wind"), ("solar_utility", SOLAR, "solar")]:
        top = base + te_rows[col].values
        ax.fill_between(x, base, top, color=c_, alpha=0.9, label=lab, lw=0)
        base = top
    for col, c_, lab in [("wind_curtailment", WIND, "wind curt"),
                         ("solar_curtailment", SOLAR, "solar curt")]:
        top = base + te_rows[col].values
        hat = np.ones(len(x), bool) if curt_solid is None else ~curt_solid
        ax.fill_between(x, base, top, where=hat, facecolor="none", edgecolor=c_,
                        hatch="////", lw=0.0, label=lab)
        if curt_solid is not None and curt_solid.any():
            ax.fill_between(x, base, top, where=curt_solid, color=c_, alpha=0.55,
                            lw=0, label=lab.split()[0] + " curt (delivered)")
        base = top
    ax.fill_between(x, 0, -disp[:, ti["battery_charging"]], color=COLORS["battery_charging"],
                    alpha=0.6, label="charging (load)")
    ax.plot(x, dem_line, "k--", lw=1.0, label="demand")
    if shade is not None:
        for s0, s1 in shade:
            ax.axvspan(s0, s1, color="gold", alpha=0.22, zorder=0)
    for d0 in range(288, len(x), 288):                     # day separators
        ax.axvline(d0, color="k", lw=0.7, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5); ax.margins(x=0); ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(title, loc="left", fontsize=10); ax.set_ylabel("MW")


def pick_days(te, n=4, seed=42):
    days = te.index.normalize()
    full = [d for d in pd.unique(days) if (days == d).sum() == 288]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(len(full), n, replace=False)), full


def deliverable_a(te, feat_cols, xm, xs_, ym, ys_, sfx):
    pick, full = pick_days(te)
    rng = np.random.default_rng(7)
    for arm in ARMS:
        p = OUT / f"{arm}{sfx}.pt"
        if not p.exists():
            print(f"[A] skip {arm} (no weights)"); continue
        model = BiLSTMImputer(n_features=len(feat_cols))
        model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        model.eval()
        act_l, pred_l, rows_l, shade = [], [], [], []
        off = 0
        for di in pick:
            d = full[di]
            day_rows = np.where(te.index.normalize() == d)[0]
            g0 = int(rng.integers(CTX, 288 - GAP - CTX))          # gap start within day
            rows = day_rows[g0:g0 + GAP]
            truth_day = te.iloc[day_rows][TARGETS].values.astype(np.float64)
            fill, pL, pR = model_fill(model, te, feat_cols, xm, xs_, ym, ys_, rows)
            nd_bal = te.iloc[rows][TARGETS].values @ SIGN          # supply-side target
            P = rayen(fill, pL, pR, nd_bal, size_aware=(arm == "size_aware"))
            pred_day = truth_day.copy(); pred_day[g0:g0 + GAP] = P
            act_l.append(truth_day); pred_l.append(pred_day); rows_l.append(day_rows)
            shade.append((off + g0, off + g0 + GAP)); off += 288
        act = np.concatenate(act_l); pred = np.concatenate(pred_l)
        rows_all = np.concatenate(rows_l)
        dem = te.iloc[rows_all]["demand_mw"].values
        fig, ax = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True, sharey=True)
        draw_stack(ax[0], te.iloc[rows_all], act, dem, shade, "actual")
        draw_stack(ax[1], te.iloc[rows_all], pred, dem, shade,
                   f"predicted ({arm}) — highlighted = the imputed 3h gap")
        ax[1].set_xticks([k * 288 for k in range(len(pick))])
        ax[1].set_xticklabels([str(full[di].date()) for di in pick])
        ax[0].legend(loc="upper left", ncol=6, fontsize=7, framealpha=0.9)
        fig.suptitle(f"plan1 A — actual vs predicted, {arm}, 4 random test days, random 3h gaps")
        fig.tight_layout()
        fp = OUT / f"figA_{arm}{sfx}.png"
        fig.savefig(fp, dpi=140); plt.close(fig)
        print("[A] wrote", fp)


def deliverable_b(te, feat_cols, xm, xs_, ym, ys_, sfx):
    pick, full = pick_days(te)
    hour = te.index.hour
    free_all = (hour >= FREE[0]) & (hour < FREE[1])
    sh = FixedPercentageShift(Q, Q, free_hours=FREE).transform(
        pd.DataFrame({"demand_mw": te["demand_mw"].values,
                      "price_aud_per_mwh": te["price_aud_per_mwh"].values},
                     index=te.index))
    dem_s = sh["demand_mw"].values
    curt = (te["wind_curtailment"] + te["solar_curtailment"]).values
    nd_bal_base = te[TARGETS].values @ SIGN
    d_dem = dem_s - te["demand_mw"].values
    nd_bal_cf = nd_bal_base + d_dem - np.where(free_all, curt, 0.0)
    # the nd FEATURE the model sees (user recipe: actual curtailment in the formula)
    nd_feat_cf = (dem_s - te["wind"].values - te["solar_utility"].values
                  - te["wind_curtailment"].values - te["solar_curtailment"].values)

    def overrides(span_idx):
        rows = te.iloc[span_idx]
        fr = free_all[span_idx]
        ov = {"demand_mw": dem_s[span_idx], "net_demand": nd_feat_cf[span_idx],
              "price_aud_per_mwh": np.where(fr, 0.0, rows["price_aud_per_mwh"].values),
              "wind_curtailment": np.where(fr, 0.0, rows["wind_curtailment"].values),
              "solar_curtailment": np.where(fr, 0.0, rows["solar_curtailment"].values)}
        return ov

    # shared actual stack
    rows_all = np.concatenate([np.where(te.index.normalize() == full[di])[0] for di in pick])
    shade = [(off * 288 + FREE[0] * 12, off * 288 + FREE[1] * 12) for off in range(len(pick))]
    fig, ax = plt.subplots(figsize=(16, 4.8))
    draw_stack(ax, te.iloc[rows_all], te.iloc[rows_all][TARGETS].values.astype(np.float64),
               te.iloc[rows_all]["demand_mw"].values, shade, "actual (free window shaded)")
    ax.legend(loc="upper left", ncol=6, fontsize=7, framealpha=0.9)
    ax.set_xticks([k * 288 for k in range(len(pick))])
    ax.set_xticklabels([str(full[di].date()) for di in pick])
    fig.suptitle("plan1 B — actual")
    fig.tight_layout(); fig.savefig(OUT / f"cf_actual{sfx}.png", dpi=140); plt.close(fig)
    print("[B] wrote", OUT / f"cf_actual{sfx}.png")

    for arm in ARMS:
        p = OUT / f"{arm}{sfx}.pt"
        if not p.exists():
            print(f"[B] skip {arm} (no weights)"); continue
        model = BiLSTMImputer(n_features=len(feat_cols))
        model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        model.eval()
        size_aware = arm == "size_aware"
        for mode in ("scaled", "masked"):
            days_out, viol = [], {"bal>1MW": 0, "ramp": 0, "neg": 0}
            for di in pick:
                day_rows = np.where(te.index.normalize() == full[di])[0]
                truth_day = te.iloc[day_rows][TARGETS].values.astype(np.float64)
                g0 = FREE[0] * 12
                rows = day_rows[g0:g0 + GAP]
                span = np.arange(rows[0] - CTX, rows[-1] + 1 + CTX)
                if mode == "scaled":
                    ratio = (nd_bal_cf[day_rows] / nd_bal_base[day_rows])[:, None]
                    day_disp = truth_day * ratio       # off-window scaled dispatch
                else:
                    day_disp = truth_day.copy()        # off-window stays actual
                # context dispatch fed to the model = the mode's own off-window world
                ctx_disp = np.zeros((len(span), 6))
                for k, r in enumerate(span):
                    pos = np.where(day_rows == r)[0]
                    ctx_disp[k] = day_disp[pos[0]] if len(pos) else te.iloc[r][TARGETS].values
                fill, pL, pR = model_fill(model, te, feat_cols, xm, xs_, ym, ys_,
                                          rows, overrides(span), ctx_disp)
                P = rayen(fill, pL, pR, nd_bal_cf[rows], size_aware)
                day_disp[g0:g0 + GAP] = P
                days_out.append(day_disp)
                viol["bal>1MW"] += int((np.abs((P * SIGN).sum(-1) - nd_bal_cf[rows]) > 1.0).sum())
                d = np.diff(np.vstack([pL[None], P, pR[None]]), axis=0)
                viol["ramp"] += int(((d > C.R_UP + 0.6) | (d < -(C.R_DN + 0.6))).sum())
                viol["neg"] += int((P < -0.1).sum())
            disp = np.concatenate(days_out)
            fig, ax = plt.subplots(figsize=(16, 4.8))
            title = ("scaled before/after — off-window x nd_after/nd_before, window model-filled"
                     if mode == "scaled" else
                     "actual + masked window — off-window actual, window model-filled")
            draw_stack(ax, te.iloc[rows_all], disp, dem_s[rows_all], shade,
                       f"{title}  [{arm}]  (rebound {Q:g}%, reduce {Q:g}%)",
                       curt_solid=free_all[rows_all])
            ax.legend(loc="upper left", ncol=6, fontsize=7, framealpha=0.9)
            ax.set_xticks([k * 288 for k in range(len(pick))])
            ax.set_xticklabels([str(full[di].date()) for di in pick])
            fig.suptitle(f"plan1 B — counterfactual ({mode}), {arm}")
            fig.tight_layout()
            fp = OUT / f"cf_{mode}_{arm}{sfx}.png"
            fig.savefig(fp, dpi=140); plt.close(fig)
            print(f"[B] wrote {fp}   violations {viol}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["a", "b"], choices=["a", "b"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    te, feat_cols, xm, xs_, ym, ys_ = load_test()
    OUT.mkdir(parents=True, exist_ok=True)
    if "a" in args.which:
        deliverable_a(te, feat_cols, xm, xs_, ym, ys_, sfx)
    if "b" in args.which:
        deliverable_b(te, feat_cols, xm, xs_, ym, ys_, sfx)


if __name__ == "__main__":
    main()
