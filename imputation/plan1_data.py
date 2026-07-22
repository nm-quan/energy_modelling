"""Plan1 dataset build: 21-feature flats with renewables + curtailment features
and the demand-side net_demand definition.

Extends the hist 5-min table (which already carries wind/solar_utility) with the
full-history curtailment pull (data/vic_curtailment_hist.parquet, clipped >= 0),
appends 4 feature columns AFTER the existing 17 (so every hardcoded index in
gap_data.py stays valid):

    [17] wind  [18] solar_utility  [19] wind_curtailment  [20] solar_curtailment

and REDEFINES net_demand (user decision, plan1.md S1):

    nd = demand_mw - wind - solar_utility - wind_curtailment - solar_curtailment

Splits/scaling replicate lib/pipeline.prepare exactly (hist boundaries, lb 288,
StandardScaler fit on train). Output: data/preprocessed/hist/5min/net_dispatch_ren/
prepared.npz -- same keys as the totdem npz, loadable by gap_data.load_flats via
the PLAN1_NPZ path.

Also measured and printed (because Sigma dispatch != this nd -- interconnector):
the balance residual per split and the FEASIBILITY FLOOR -- MAE(project(truth))
on test 3h gaps -- i.e. what the new nd definition costs a perfect model.

    python3 imputation/plan1_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lib"))

from sklearn.preprocessing import StandardScaler                     # noqa: E402
import pipeline                                                       # noqa: E402
from gap_data import TARGETS, SIGN                                    # noqa: E402

SRC = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_totdem"
DST = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren"
CURT = ROOT / "data" / "vic_curtailment_hist.parquet"
NEW_FEATS = ["wind", "solar_utility", "wind_curtailment", "solar_curtailment"]
LB = 288


def main():
    old = np.load(SRC / "prepared.npz", allow_pickle=False)
    feat17 = [str(c) for c in old["feat_cols"]]
    t = pd.read_parquet(SRC / "table.parquet")

    cur = pd.read_parquet(CURT)
    cur["interval"] = pd.to_datetime(cur["interval"])
    cur = cur.set_index("interval").sort_index()
    wc = cur["curtailment_wind_mw"].reindex(t.index).clip(lower=0)
    sc = cur["curtailment_solar_mw"].reindex(t.index).clip(lower=0)
    n_nan = int(wc.isna().sum() + sc.isna().sum())
    if n_nan:
        print(f"WARNING: {n_nan} curtailment cells missing -> 0")
    t["wind_curtailment"] = wc.fillna(0.0)
    t["solar_curtailment"] = sc.fillna(0.0)

    # demand-side net_demand (plan1.md S1). Keep the old supply-side column around
    # for diagnostics only.
    nd_supply = t["net_demand"].copy()
    t["net_demand"] = (t["demand_mw"] - t["wind"] - t["solar_utility"]
                       - t["wind_curtailment"] - t["solar_curtailment"])

    feat_cols = feat17 + NEW_FEATS
    cfg = pipeline.DATASETS["hist"]
    train_df = t[t.index <= cfg["train_end"]]
    val_df = t[(t.index > cfg["train_end"]) & (t.index <= cfg["val_end"])]
    test_df = t[t.index > cfg["val_end"]]
    print(f"rows: train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")

    xs = StandardScaler().fit(train_df[feat_cols].values)
    ys = StandardScaler().fit(train_df[TARGETS].values)

    def scale(d):
        return (xs.transform(d[feat_cols].values).astype(np.float32),
                ys.transform(d[TARGETS].values).astype(np.float32))

    flats = {"tr": scale(train_df),
             "va": scale(pd.concat([train_df.tail(LB), val_df])),
             "te": scale(pd.concat([val_df.tail(LB), test_df]))}

    DST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DST / "prepared.npz",
        Xtr_flat=flats["tr"][0], Ytr_flat=flats["tr"][1],
        Xva_flat=flats["va"][0], Yva_flat=flats["va"][1],
        Xte_flat=flats["te"][0], Yte_flat=flats["te"][1],
        test_index=np.asarray(test_df.index.astype(str), dtype=str),
        x_mean=xs.mean_, x_scale=xs.scale_, y_mean=ys.mean_, y_scale=ys.scale_,
        feat_cols=np.array(feat_cols), targets=np.array(TARGETS),
        lookback_steps=LB, horizon=1, resolution="5min", dataset="hist",
    )
    t.to_parquet(DST / "table.parquet")
    print(f"wrote {DST/'prepared.npz'}  ({(DST/'prepared.npz').stat().st_size/1e6:.1f} MB, "
          f"{len(feat_cols)} features)")

    # ---- diagnostics: what the demand-side nd costs ----
    disp = t[TARGETS].values @ SIGN
    resid = disp - t["net_demand"].values
    print("\nbalance residual  Sigma SIGN*dispatch - nd_new  (the interconnector slice):")
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        r = (d[TARGETS].values @ SIGN) - d["net_demand"].values
        print(f"  {name:5s}  mean {r.mean():+8.1f}  mean|.| {np.abs(r).mean():7.1f}  "
              f"p95|.| {np.percentile(np.abs(r),95):7.1f}  max|.| {np.abs(r).max():7.1f} MW")
    print(f"  (old supply-side nd residual was exactly 0 by construction; "
          f"nd_new - nd_supply mean {float((t['net_demand']-nd_supply).mean()):+.1f} MW)")

    # feasibility floor: project TRUTH 3h gaps onto {balance to nd_new, ramp, box}
    import constraints as C
    rng = np.random.default_rng(0)
    te = test_df
    nd_te = te["net_demand"].values
    truth = te[TARGETS].values.astype(np.float64)
    G = 36
    maes, resids = [], []
    for _ in range(100):
        s = int(rng.integers(1, len(te) - G - 1))
        P, r = C.project_gap(truth[s:s + G].copy(), truth[s - 1], truth[s + G],
                             nd_te[s:s + G], iters=60)
        maes.append(np.abs(P - truth[s:s + G]).mean())
        resids.append(r.max())
    print(f"\nFEASIBILITY FLOOR (100 random test 3h gaps, truth projected to nd_new):")
    print(f"  MAE(proj(truth), truth) mean {np.mean(maes):.1f} MW  p95 {np.percentile(maes,95):.1f}  "
          f"max {np.max(maes):.1f} MW")
    print(f"  post-projection balance resid max {np.max(resids):.2f} MW")
    print("  -> this MAE is unavoidable for ANY model whose output is balanced to nd_new.")


if __name__ == "__main__":
    main()
