"""Classical imputation baselines (mean / knn / mice / mf) + interpolation on the
plan1 test set (400 3h gaps, ren flats), scored with MAE + MRE through the same
rayen map as plan1_bench -- directly comparable to the learned arms.

    python3 imputation/plan1_baselines.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_data as GD                                                  # noqa: E402
GD.NPZ = HERE.parent / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
from gap_data import load_flats, sample_recon_windows, TARGETS         # noqa: E402
import plan1_bench as B                                                # noqa: E402  (reuse score())
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import KNNImputer, IterativeImputer               # noqa: E402

OUT = HERE / "results" / "plan1"
NF = 21
DRV = slice(6, NF)                     # drivers: net_demand, demand, price, calendar, wind/solar/curt
METHODS = ["mean", "knn", "mice", "mf", "interp"]


def reference(f, n_ref=20000, seed=123):
    rng = np.random.default_rng(seed)
    j = rng.choice(f.Ytr.shape[0], size=min(n_ref, f.Ytr.shape[0]), replace=False)
    return np.hstack([f.Ytr[j], f.Xtr[j, DRV]]).astype(np.float64)     # (n_ref, 21) z-scored


def impute_z(method, Mref, Dgap, rank=6):
    """z-scored source fills (M,6) for gap driver rows Dgap (M, NF-6)."""
    M = np.full((Dgap.shape[0], NF), np.nan); M[:, 6:] = Dgap
    if method == "mean":
        return np.broadcast_to(Mref[:, :6].mean(0), (Dgap.shape[0], 6)).copy()
    if method == "knn":
        return KNNImputer(n_neighbors=10, weights="distance").fit(Mref).transform(M)[:, :6]
    if method == "mice":
        return IterativeImputer(max_iter=10, random_state=0).fit(Mref).transform(M)[:, :6]
    if method == "mf":
        mu = Mref.mean(0)
        _, _, Vt = np.linalg.svd(Mref - mu, full_matrices=False)
        Vr = Vt[:rank].T
        Z = (Dgap - mu[6:]) @ np.linalg.pinv(Vr[6:]).T
        return mu[:6] + Z @ Vr[:6].T
    raise ValueError(method)


def main():
    f = load_flats()
    gws = sample_recon_windows(f, "test", n=400, context=48, gap=36, seed=123)
    Mref = reference(f)
    print(f"reference {Mref.shape[0]} train rows | {len(gws)} test 3h gaps, {NF} features")

    rows = []
    for m in METHODS:
        if m == "interp":                                             # linear between pinned boundaries
            fills = []
            for gw in gws:
                N = len(gw.gap_idx); tt = (np.arange(1, N + 1) / (N + 1))[:, None]
                fills.append(gw.pL_mw[None] + tt * (gw.pR_mw - gw.pL_mw)[None])
        else:
            Dgap = np.vstack([f.Xte[gw.gap_idx][:, DRV] for gw in gws]).astype(np.float64)
            src = f.y_to_mw(impute_z(m, Mref, Dgap))                  # (M,6) MW
            fills, off = [], 0
            for gw in gws:
                N = len(gw.gap_idx); fills.append(src[off:off + N]); off += N
        r = B.score(m, fills, gws, size_aware=False)                 # rayen map, MAE+MRE
        rows.append(r)
        print(f"  {m:7s} MAE={r['mae_agg']:.1f}  MRE={r['mre_agg']:.2f}%")

    lines = ["# plan1 baselines — classical imputers, 400 test 3h gaps, rayen map", "",
             "| method | MAE agg (MW) | MRE agg (%) | " + " | ".join(f"MRE {t}" for t in TARGETS) + " |",
             "| --- | --- | --- |" + " --- |" * 6]
    for r in rows:
        lines.append(f"| {r['name']} | {r['mae_agg']:.1f} | {r['mre_agg']:.2f} | "
                     + " | ".join(f"{r['mre'][t]:.1f}" for t in TARGETS) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baselines.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("\nwrote", OUT / "baselines.md")


if __name__ == "__main__":
    main()
