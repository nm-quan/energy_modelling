"""Write teacher-forced test predictions of the anchored model to CSV in the
ml/<name>/predictions.csv format, for ml/check_caps.py.

The anchor guarantees sum(SIGN*pred) == net_demand read off the input window's
LAST step, i.e. nd(t-1) for a step-t prediction -- so the check_caps sum
identity (against actual nd(t)) should show a residual equal to the 5-min nd
ramp (p50 ~53 MW), versus the GRU baseline's max 617 MW / 9,553 rows > 10 MW.
Negative predictions should be exactly zero (ReLU head).

    python demand_simulation/gen_anchor_preds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402
import sim_common as sc         # noqa: E402
import models as M              # noqa: E402
import evaluate as ev           # noqa: E402

NOSTEAM = [0, 1, 3, 5]
DST = ROOT / "ml" / "itransformer_anchor_5min"


def main():
    device = "cpu"
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    fc, xs, ys = data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nd_idx = fc.index("net_demand")

    bare = M.make_neural("itransformer", n_features=len(fc), n_targets=6)
    bare.load_state_dict(torch.load(ROOT / "ml" / "itransformer_totdem" / "itransformer_totdem.pt",
                                    map_location="cpu", weights_only=True))
    model = M.DemandAnchoredHead(bare, xs.mean_[nd_idx], xs.scale_[nd_idx],
                                 ys.mean_, ys.scale_, nd_feat_idx=nd_idx,
                                 rescale_idx=NOSTEAM, pos_fn="relu").to(device).eval()

    pred = ys.inverse_transform(ev.predict_neural(model, data["Xte"], device, batch=256))
    pred = np.maximum(pred, 0.0)   # the head's zeros pick up -1e-6 noise in the y_scaler round trip
    true = ys.inverse_transform(data["Yte"])

    out = pd.DataFrame({"interval": pd.DatetimeIndex(data["test_index"])})
    for i, t in enumerate(sc.TARGETS):
        out[f"{t}_actual"] = true[:, i]
        out[f"{t}_pred"] = pred[:, i]
    DST.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST / "predictions.csv", index=False)
    met = ev.compute_metrics(true, pred, sc.TARGETS)["average"]
    print(f"WAPE={met['WAPE']:.4f} R2={met['R2']:.4f}")
    print("wrote", DST / "predictions.csv")


if __name__ == "__main__":
    main()
