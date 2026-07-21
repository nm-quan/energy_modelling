"""Classical imputation baselines — Mean / KNN / MICE / MF — on the SAME general eval
windows as benchmark.py, scored through the SAME projection Π. These are the standard
missing-data imputers (constraints/lit_review.md themes 8-9): they fill the 6 hidden
source channels in each gap from the KNOWN drivers (net_demand, demand, price, calendar)
using a train-set reference, ignoring the two-sided boundary + temporal structure that
interpolation and the bi-LSTM exploit. Π then snaps every fill to balance/ramp/box/SOC,
so the rows are apples-to-apples with interp+projection and bilstm/*.

  mean   SimpleImputer(strategy=mean)   — fill each source with its train-set mean (floor)
  knn    KNNImputer                     — average the k train rows nearest in driver space
  mice   IterativeImputer               — sklearn's MICE: regress each source on the rest
  mf     low-rank (PCA-subspace)        — matrix-factorization completion: project the row's
                                          observed drivers onto the rank-r subspace of the
                                          fully-observed train matrix, reconstruct the sources

Everything is done in the z-scored space the flats already provide (homogeneous scale
for KNN/MF distances), then mapped back to MW for projection + WAPE.

    python3 imputation/classical_baselines.py                       # full (matches benchmark eval)
    python3 imputation/classical_baselines.py --n-ref 3000 --n-eval 100   # quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap_data import load_flats, sample_recon_windows, TARGETS                    # noqa: E402
import constraints as C                                                           # noqa: E402
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer            # noqa: E402

OUT = HERE / "results"
DRIVER_COLS = slice(6, 17)          # net_demand, demand, price, 8 calendar features (always known)
METHODS = ["mean", "knn", "mice", "mf"]


def build_reference(f, n_ref, seed):
    """(n_ref, 17) fully-observed z-scored rows = [6 sources (TARGETS order) | 11 drivers],
    sampled from the train flat. This is the population the imputers learn from."""
    rng = np.random.default_rng(seed)
    j = rng.choice(f.Ytr.shape[0], size=min(n_ref, f.Ytr.shape[0]), replace=False)
    return np.hstack([f.Ytr[j], f.Xtr[j, DRIVER_COLS]]).astype(np.float64)


def impute_sources_z(method, Mref, Dgap, rank=6):
    """Return imputed z-scored sources (M,6) for gap driver rows Dgap (M,11).
    Mref (n_ref,17) is fully observed; the 6 source columns of the gap rows are NaN."""
    M = np.full((Dgap.shape[0], 17), np.nan)
    M[:, 6:] = Dgap
    if method == "mean":
        return np.broadcast_to(Mref[:, :6].mean(0), (Dgap.shape[0], 6)).copy()
    if method == "knn":
        return KNNImputer(n_neighbors=10, weights="distance").fit(Mref).transform(M)[:, :6]
    if method == "mice":
        return IterativeImputer(max_iter=10, random_state=0).fit(Mref).transform(M)[:, :6]
    if method == "mf":
        mu = Mref.mean(0)                                        # (17,)
        _, _, Vt = np.linalg.svd(Mref - mu, full_matrices=False)
        Vr = Vt[:rank].T                                        # (17, r) rank-r column subspace
        pinv_Vo = np.linalg.pinv(Vr[6:])                        # (r, 11) from the observed drivers
        Z = (Dgap - mu[6:]) @ pinv_Vo.T                         # (M, r) latent per gap row
        return mu[:6] + Z @ Vr[:6].T                            # reconstruct sources
    raise ValueError(method)


def score(method, f, gws, Mref):
    """Impute every gap window, project Π, accumulate macro/micro/midday WAPE + the
    constraint columns -- identical accounting to benchmark.score_fill."""
    Dgap = np.vstack([f.Xte[gw.gap_idx][:, DRIVER_COLS] for gw in gws]).astype(np.float64)
    src_mw = f.y_to_mw(impute_sources_z(method, Mref, Dgap))    # (M,6) MW, TARGETS order
    num = np.zeros(6); den = np.zeros(6)
    mnum = np.zeros(6); mden = np.zeros(6)
    ramp_over = 0.0; neg = 0; bal = 0.0
    off = 0
    for gw in gws:
        N = len(gw.gap_idx)
        fill = src_mw[off:off + N]; off += N
        P, resid = C.project_gap(fill, gw.pL_mw, gw.pR_mw, gw.nd_mw)
        e = np.abs(P - gw.truth_mw).sum(0); t = np.abs(gw.truth_mw).sum(0)
        num += e; den += t
        if 11 <= gw.hour < 14:
            mnum += e; mden += t
        ramp_over = max(ramp_over, C._ramp_overshoot_mw(np.vstack([gw.pL_mw, P, gw.pR_mw])))
        neg += int((P < -0.1).sum()); bal = max(bal, float(resid.max()))
    per = num / np.clip(den, 1e-6, None)
    return {"method": method, "macro_WAPE": float(per.mean()),
            "micro_WAPE": float(num.sum() / max(den.sum(), 1e-6)),
            "midday_micro_WAPE": float(mnum.sum() / max(mden.sum(), 1e-6)),
            "per_channel_WAPE": {t: float(per[i]) for i, t in enumerate(TARGETS)},
            "ramp_overshoot_mw": ramp_over, "balance_resid_max_mw": bal, "n_neg": neg,
            "context": 48, "n_eval": len(gws), "eval_seed": 123, "n_ref": int(Mref.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=20000, help="train rows the imputers learn from")
    ap.add_argument("--n-eval", type=int, default=800, help="test windows (match benchmark: 800)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    f = load_flats()
    gws = sample_recon_windows(f, "test", n=args.n_eval, context=48, seed=args.seed)
    Mref = build_reference(f, args.n_ref, seed=args.seed)
    print(f"reference {Mref.shape[0]} train rows | {len(gws)} test windows "
          f"({sum(1 for g in gws if 11 <= g.hour < 14)} midday)")
    OUT.mkdir(exist_ok=True)
    for m in METHODS:
        r = score(m, f, gws, Mref)
        (OUT / f"baseline_{m}.json").write_text(json.dumps(r, indent=2))
        print(f"  {m:5s} macro={r['macro_WAPE']:.4f}  micro={r['micro_WAPE']:.4f}  "
              f"midday={r['midday_micro_WAPE']:.4f}  ramp={r['ramp_overshoot_mw']:.1e}  "
              f"bal={r['balance_resid_max_mw']:.1e}  neg={r['n_neg']}")
    print("wrote baseline_{mean,knn,mice,mf}.json — re-run benchmark.py to tabulate")


if __name__ == "__main__":
    main()
