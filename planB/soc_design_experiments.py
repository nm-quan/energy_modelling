"""Eight experiments that decide how SOC should enter the model.

The question is not "what is the battery's physics" -- that is settled in
quick_findings/vic_battery_constraints/. It is "which encoding of that physics
gets the prediction closer to the truth". A hard constraint only helps when it
binds; a reparameterisation helps whenever it makes the target better
conditioned. These measure both.

    python3 planB/soc_design_experiments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "imputation")); sys.path.insert(0, str(ROOT / "planB"))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
from gap_data import load_flats, TARGETS                               # noqa: E402
from fit import build, CTX, GAP                                        # noqa: E402

ETA = np.sqrt(0.839)          # one-way, the drift-zeroing value
DT = 5 / 60.0
EMAX = 3966.0                 # MWh, largest observed daily swing
CHG = TARGETS.index("battery_charging")
DIS = TARGETS.index("battery_discharging")
TABLE = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/table.parquet"
WINDOW = slice("2025-12-01", None)


def hdr(n, s):
    print("\n" + "=" * 72); print(f"EXP {n} — {s}"); print("=" * 72)


def series():
    W = pd.read_parquet(TABLE).loc[WINDOW]
    return W.battery_charging.to_numpy(), W.battery_discharging.to_numpy()


def gaps():
    f = load_flats()
    W = 2 * CTX + GAP
    rng = np.random.default_rng(123)
    starts = rng.integers(max(0, f.lb_carry - CTX), f.Xte.shape[0] - W, size=400)
    b = build(f, starts, "test")
    return f, starts, b, b["Y"] * f.y_scale + f.y_mean


def main():
    chg, dis = series()
    n_, m_ = dis - chg, np.minimum(chg, dis)
    dE_ = (chg * ETA - dis / ETA) * DT

    hdr(1, "price of assuming complementarity (predict E only)")
    cr = np.maximum(dE_, 0) / (ETA * DT); dr = np.maximum(-dE_, 0) * ETA / DT
    print(f"  chg MAE {np.abs(cr-chg).mean():6.2f}   dis MAE {np.abs(dr-dis).mean():6.2f}  "
          f"  <- floor for any E-only model")

    hdr(2, "does SOC LEVEL predict the next action?")
    W = pd.read_parquet(TABLE).loc[WINDOW]
    idx = pd.to_datetime(W.index); day = idx.floor("D")
    E = pd.Series(np.cumsum(dE_), index=idx)
    soc = (E.groupby(day).transform(lambda v: v - v.min()) / EMAX).to_numpy()
    r = np.corrcoef(soc, n_)[0, 1]
    hm = pd.Series(n_, index=idx).groupby(idx.hour).transform("mean").to_numpy()
    hs = pd.Series(soc, index=idx).groupby(idx.hour).transform("mean").to_numpy()
    print(f"  corr(soc, net) {r:+.3f}; after removing hour-of-day "
          f"{np.corrcoef(soc-hs, n_-hm)[0,1]:+.3f}  <- no incremental signal")

    f, starts, b, truth = gaps()
    pL, pR = b["pL"], b["pR"]
    tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
    interp = pL[:, None] + tt * (pR - pL)[:, None]
    tc, td = truth[..., CHG], truth[..., DIS]

    def mae(c, d):
        return np.abs(c - tc).mean(), np.abs(d - td).mean()

    hdr(3, "same trivial predictor, power space vs energy space")
    c, d = mae(interp[..., CHG], interp[..., DIS])
    print(f"  A interpolate power          {c:6.2f} {d:6.2f}   mean {(c+d)/2:6.2f}")
    dEt = (tc * ETA - td / ETA) * DT
    Et = np.cumsum(dEt, 1)
    sL = (pL[:, CHG] * ETA - pL[:, DIS] / ETA) * DT
    sR = (pR[:, CHG] * ETA - pR[:, DIS] / ETA) * DT
    nn = np.arange(1, GAP + 1)[None, :]
    dEl = sL[:, None] + (nn - 1) / (GAP - 1) * (sR - sL)[:, None]

    def topow(dE):
        return np.maximum(dE, 0) / (ETA * DT), np.maximum(-dE, 0) * ETA / DT
    c, d = mae(*topow(dEl))
    print(f"  B interpolate dE/dt          {c:6.2f} {d:6.2f}   mean {(c+d)/2:6.2f}")
    c, d = mae(*topow(dEl + ((Et[:, -1] - dEl.sum(1)) / GAP)[:, None]))
    print(f"  C B + true gap energy        {c:6.2f} {d:6.2f}   mean {(c+d)/2:6.2f}"
          f"   <- one scalar matches the trained net (47.05)")
    c, d = mae(*topow(dEt))
    print(f"  D true E(t)                  {c:6.2f} {d:6.2f}   mean {(c+d)/2:6.2f}")

    hdr(5, "is that scalar predictable without the leak?")
    tot = Et[:, -1]; trap = (sL + sR) / 2 * GAP
    print(f"  trapezoid on the seam slopes: MAE {np.abs(tot-trap).mean():.0f} MWh, "
          f"corr {np.corrcoef(trap,tot)[0,1]:+.3f}")
    print(f"  assume zero:                  MAE {np.abs(tot).mean():.0f} MWh")

    hdr(6, "does the 24h reservoir bound pin the gap energy?")
    Ymw = f.y_to_mw(f.Yte)
    dEa = (Ymw[:, CHG] * ETA - Ymw[:, DIS] / ETA) * DT
    H, wid = 126, []
    for st in starts:
        g0, g1 = st + CTX, st + CTX + GAP
        if g0 - H < 0 or g1 + H > len(dEa):
            continue
        pre = np.cumsum(dEa[g0-H:g0]); gp = np.cumsum(dEa[g0:g1])
        po = np.cumsum(dEa[g1:g1+H]); t0 = gp[-1]
        ts = np.linspace(t0 - 4000, t0 + 4000, 1601)
        hi = np.maximum.reduce([np.full_like(ts, pre.max()),
                                pre[-1] + gp.max() + np.maximum(0, ts - t0),
                                pre[-1] + ts + po.max()])
        lo = np.minimum.reduce([np.full_like(ts, pre.min()),
                                pre[-1] + gp.min() + np.minimum(0, ts - t0),
                                pre[-1] + ts + po.min()])
        g = (hi - lo) <= EMAX
        if g.any():
            wid.append(ts[g].max() - ts[g].min())
    wid = np.array(wid)
    print(f"  admissible width for gap energy: median {np.median(wid):,.0f} MWh "
          f"(vs 284 MWh trapezoid error) -> {np.mean(wid<568)*100:.0f}% ever binding")

    hdr(7, "conditioning: (n, dE) is a trap, (n, m) is not")
    k = ETA - 1 / ETA
    print(f"  corr(n, dE/DT) = {np.corrcoef(n_, dE_/DT)[0,1]:+.4f}, 1/|k| = {1/abs(k):.1f}x")
    rs = np.random.default_rng(0)
    for s in (10.0, 30.0):
        e = rs.normal(0, s, len(n_))
        a = ((dE_/DT + e) + n_/ETA) / k
        bb = np.maximum(m_ + e, 0) + np.maximum(0, -n_)
        print(f"  perturb 2nd coord by {s:4.0f} MW -> (n,dE) chg MAE {np.abs(a-chg).mean():7.1f} | "
              f"(n,m) {np.abs(bb-chg).mean():6.1f}")

    hdr(8, "the (n, m) parameterisation")
    cr = m_ + np.maximum(0, -n_); dr = m_ + np.maximum(0, n_)
    print(f"  round-trip exact: {np.abs(cr-chg).max():.1e} / {np.abs(dr-dis).max():.1e} MW")
    print(f"  overlap m: mean {m_.mean():.1f} MW, median {np.median(m_):.1f}, "
          f"zero on {np.mean(m_<1e-9)*100:.0f}% of steps")
    print(f"  predicting m=0 always: chg MAE {np.abs(np.maximum(0,-n_)-chg).mean():.2f}, "
          f"dis MAE {np.abs(np.maximum(0,n_)-dis).mean():.2f} MW")


if __name__ == "__main__":
    main()
