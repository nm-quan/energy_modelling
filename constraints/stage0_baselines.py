"""Stage 0: score the actuals, the GRU CSV baseline, and every existing
checkpoint (bare / +clamp / +anchor) through constraint_report -- the
unconstrained end of the WAPE-vs-violations graph (constraint_research.md).

All rows are teacher-forced test-set predictions (10,656 windows). +clamp is
np.maximum(pred, 0) (what sc.predict applies in production); +anchor is
DemandAnchoredHead(rescale_idx=nosteam, pos_fn=relu) at inference, the demand-
balance winner from demand_simulation/findings.md.

    python3 constraints/stage0_baselines.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402
import models as M              # noqa: E402
import evaluate as ev           # noqa: E402
import harness as H             # noqa: E402

STAGE = "stage0"
NOSTEAM = [0, 1, 3, 5]
BARE = [("lstm_5min_mse", "lstm", ROOT / "ml" / "lstm_5min_mse" / "lstm_5min_mse.pt"),
        ("lstm_revin", "lstm_revin", ROOT / "ml" / "lstm_revin_5min" / "lstm_revin_5min.pt"),
        ("itransformer", "itransformer", ROOT / "ml" / "itransformer_totdem" / "itransformer_totdem.pt")]
GRU_CSV = ROOT / "data" / "gru_predictions_5min.csv"
ORDER = ["actuals", "gru_csv", "lstm_5min_mse",
         "lstm_revin", "lstm_revin+clamp", "lstm_revin+anchor",
         "itransformer", "itransformer+clamp", "itransformer+anchor"]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    fc, xs, ys = data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nd_idx = fc.index("net_demand")
    Xte = data["Xte"]
    true_te = ys.inverse_transform(data["Yte"])
    ti = pd.DatetimeIndex(data["test_index"])
    nd_input = Xte[:, -1, nd_idx] * xs.scale_[nd_idx] + xs.mean_[nd_idx]
    data["Xtr"] = data["Ytr"] = data["Xva"] = data["Yva"] = None   # 8 GB machine

    def report(name, pred):
        row = {"model": name, **H.constraint_report(pred, true_te, ti, nd_input)}
        H.save_row(STAGE, row)
        print(f"  {name:22s} WAPE={row['WAPE']:.4f} neg={row['n_neg']} cap={row['n_cap']} "
              f"ramp={row['n_ramp']} dem@in={row['n_demand_in']} dem@act={row['n_demand_act']} "
              f"SOC={'ok' if row['soc_feasible'] else row['soc_feasible']}", flush=True)

    report("actuals", true_te)

    if GRU_CSV.exists():
        g = pd.read_csv(GRU_CSV)
        g["interval"] = pd.to_datetime(g["interval"])
        g = g.set_index("interval").reindex(ti)
        cols = [f"{t}_pred" for t in H.TARGETS]
        if not g[cols].isna().any().any():
            report("gru_csv", g[cols].to_numpy())
        else:
            print("  gru_csv: skipped (interval mismatch with pipeline test index)")

    for name, arch, weights in BARE:
        t0 = time.time()
        bare = M.make_neural(arch, n_features=len(fc), n_targets=6)
        bare.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        bare = bare.to(device).eval()
        pred = ys.inverse_transform(ev.predict_neural(bare, Xte, device, batch=256))
        report(name, pred)
        if name != "lstm_5min_mse":                      # clamp + anchor variants
            report(f"{name}+clamp", np.maximum(pred, 0.0))
            head = M.DemandAnchoredHead(bare, xs.mean_[nd_idx], xs.scale_[nd_idx],
                                        ys.mean_, ys.scale_, nd_feat_idx=nd_idx,
                                        rescale_idx=NOSTEAM, pos_fn="relu").to(device).eval()
            pa = ys.inverse_transform(ev.predict_neural(head, Xte, device, batch=256))
            report(f"{name}+anchor", np.maximum(pa, 0.0))   # kill y-scaler round-trip -1e-6s
        print(f"    ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + H.write_table(STAGE, "Stage 0 baselines: existing models through constraint_report",
                               order=ORDER))


if __name__ == "__main__":
    main()
