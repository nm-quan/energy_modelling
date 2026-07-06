"""Scenario report: base vs scenario dispatch, two tables -- response region only
and the entire test set -- plus the demand before/after chart. No RevIN LSTM
(lstm_5min_mse), closed-loop ar_free_rollout (nd_mode="scenario", demand-side
net_demand), same methodology as sweep_eqnd.py. Writes
demand_simulation/sweep_eqnd/result_reb{X}_red{Y}.md (tables only) and
sweep_eqnd/figure/demand_before_after_reb{X}_red{Y}.png.

    python demand_simulation/scenario_report.py --rebound 7 --reduction 7
"""
from __future__ import annotations

import argparse
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
from shift_model import FixedPercentageShift  # noqa: E402
from sweep_eqnd import demand_side_nd, build_ctx, FREE_HOURS, HORIZON, OUT  # noqa: E402
from demand_chart import plot_demand_before_after  # noqa: E402


def table(pred_b, pred_s, dem_b, dem_s):
    rows = []
    rows.append(("demand_mw (input)", dem_b.mean(), dem_s.mean()))
    nb, ns = sc.net_demand(pred_b).mean(), sc.net_demand(pred_s).mean()
    rows.append(("net_demand (pred)", nb, ns))
    for i, t in enumerate(sc.TARGETS):
        rows.append((t, pred_b[:, i].mean(), pred_s[:, i].mean()))
    lines = ["| target | base (MW) | scenario (MW) | change |", "| --- | ---: | ---: | ---: |"]
    for name, b, s in rows:
        pct = 100.0 * (s - b) / b if b else float("nan")
        lines.append(f"| {name} | {b:.1f} | {s:.1f} | {pct:+.1f}% |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebound", type=float, required=True)
    ap.add_argument("--reduction", type=float, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = build_ctx(device)
    df, ni, feat_cols, x_scaler, y_scaler, model, lb, test_index, full_idx, free_test = (
        ctx["df"], ctx["ni"], ctx["feat_cols"], ctx["x_scaler"], ctx["y_scaler"],
        ctx["model"], ctx["lb"], ctx["test_index"], ctx["full_idx"], ctx["free_test"])

    shift = FixedPercentageShift(rebound_pct=args.rebound, reduction_pct=args.reduction,
                                 free_hours=FREE_HOURS)
    base = df.copy(); base["net_demand"] = demand_side_nd(base, ni)
    scen = shift.transform(df); scen["net_demand"] = demand_side_nd(scen, ni)

    fs_base = x_scaler.transform(base.loc[full_idx, feat_cols].values).astype(np.float32)
    fs_scen = x_scaler.transform(scen.loc[full_idx, feat_cols].values).astype(np.float32)
    print("rolling out (closed loop)...")
    pred_b, pred_s = sc.ar_free_rollout(model, fs_base, fs_scen, lb, free_test,
                                        x_scaler, y_scaler, device, nd_mode="scenario")

    mask = sc.response_mask(test_index, HORIZON, FREE_HOURS)
    dem_b_all = base.loc[test_index, "demand_mw"].to_numpy()
    dem_s_all = scen.loc[test_index, "demand_mw"].to_numpy()

    t_response = table(pred_b[mask], pred_s[mask], dem_b_all[mask], dem_s_all[mask])
    t_full = table(pred_b, pred_s, dem_b_all, dem_s_all)

    lines = [f"# rebound {args.rebound:g}%, reduction {args.reduction:g}%\n",
             "## Response region\n", t_response, "",
             "## Entire test set\n", t_full, ""]
    sfx = f"_reb{args.rebound:g}_red{args.reduction:g}"
    out_path = OUT / f"result{sfx}.md"
    out_path.write_text("\n".join(lines))
    print("wrote", out_path)

    fig_dir = OUT / "figure"; fig_dir.mkdir(parents=True, exist_ok=True)
    plot_demand_before_after(test_index, dem_b_all, dem_s_all, FREE_HOURS,
                             fig_dir / f"demand_before_after{sfx}.png",
                             tag=f"shift reb{args.rebound:g}% red{args.reduction:g}%")


if __name__ == "__main__":
    main()
