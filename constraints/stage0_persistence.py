"""Naive 5-min persistence baseline: pred(t) = dispatch at t-1, read off the
input window's last step (handles index gaps; exactly one step behind).

Why it belongs in stage 0: persistence inherits the actuals' feasibility with a
one-step shift (ramp trajectory = actual ramps, SOC reservoir = actual swing)
and sum(SIGN*pred) == nd(t-1) EXACTLY by the pipeline identity -- so it is a
zero-cost, perfectly @in-balanced "model". Its WAPE is the floor every learned
model must beat, and per-target WAPE shows where learning actually adds value
at one step (the RevIN redundancy story predicts models ~ persistence).

    python3 constraints/stage0_persistence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402
import models as M              # noqa: E402
import evaluate as ev           # noqa: E402
import harness as H             # noqa: E402

ORDER = ["actuals", "persistence", "gru_csv", "lstm_5min_mse",
         "lstm_revin", "lstm_revin+clamp", "lstm_revin+anchor",
         "itransformer", "itransformer+clamp", "itransformer+anchor"]


def main():
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    fc, xs, ys = data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nd_idx = fc.index("net_demand")
    Xte = data["Xte"]
    true_te = ys.inverse_transform(data["Yte"])
    ti = pd.DatetimeIndex(data["test_index"])
    nd_input = Xte[:, -1, nd_idx] * xs.scale_[nd_idx] + xs.mean_[nd_idx]

    # last input step, feature order -> TARGETS order, x_scaler-inverted to MW
    idx = M.TARGET_FEAT_IDX
    pred = Xte[:, -1, idx] * xs.scale_[idx] + xs.mean_[idx]

    per = ev.compute_metrics(true_te, pred, H.TARGETS)["per_target"]
    row = {"model": "persistence", **H.constraint_report(pred, true_te, ti, nd_input),
           "per_target_WAPE": {t: per[t]["WAPE"] for t in H.TARGETS}}
    H.save_row("stage0", row)
    print(f"persistence  WAPE={row['WAPE']:.4f} R2={row['R2']:.4f} "
          f"neg={row['n_neg']} ramp={row['n_ramp']} dem@in={row['n_demand_in']} "
          f"dem@act={row['n_demand_act']} SOC={row['soc_feasible']}")
    print("per-target WAPE: " + "  ".join(f"{t}={per[t]['WAPE']:.3f}" for t in H.TARGETS))
    H.write_table("stage0", "Stage 0 baselines: existing models through constraint_report",
                  order=ORDER)


if __name__ == "__main__":
    main()
