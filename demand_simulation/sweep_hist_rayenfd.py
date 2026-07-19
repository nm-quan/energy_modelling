"""Closed-loop rebound sweep for the historical fixed-demand RayenFD model.

Example:
    python3 demand_simulation/sweep_hist_rayenfd.py \
        --weights ml/itransformer_rayenfd_hist_noptrain \
        --reduction 0 --rebound 1 5 10 15 20 --device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "ml"))

import evaluate as ev  # noqa: E402
import hist_constrained_shift as hcs  # noqa: E402
import pipeline  # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--reduction", type=float, default=0.0)
    ap.add_argument("--rebound", type=float, nargs="+", required=True)
    ap.add_argument("--demand-cap", type=float, default=hcs.DEMAND_CAP_MW)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(HERE / "sweep_eqnd" /
                                          "hist_rayenfd_rebound_sweep_red0.md"))
    args = ap.parse_args()

    device = ev.pick_device(args.device)
    data = pipeline.load_prepared("5min", 24, 1, "net_dispatch_totdem", "hist")
    xs, ys, fc = data["x_scaler"], data["y_scaler"], data["feat_cols"]
    hcs.data_feat_cols = fc
    lb = data["lookback_steps"]
    entries = hcs.load_entries(Path(args.weights), data, device,
                               model_filter=["itransformer_rayenfd"],
                               rayenfd_steam_pt=True)
    if len(entries) != 1:
        raise RuntimeError(f"expected one RayenFD entry, found {len(entries)}")
    _, n_out, model = entries[0]

    df = pipeline.build_table("5min", "hist")
    val_end = pipeline.DATASETS["hist"]["val_end"]
    val_df = df[(df.index > pipeline.DATASETS["hist"]["train_end"]) &
                (df.index <= val_end)]
    test_df = df[df.index > val_end]
    full = pd.concat([val_df.tail(lb), test_df])
    test_index = test_df.index
    free_test = np.asarray((test_index.hour >= hcs.FREE_HOURS[0]) &
                            (test_index.hour < hcs.FREE_HOURS[1]))
    mask = np.asarray(((test_index - pd.Timedelta(minutes=5)).hour >= hcs.FREE_HOURS[0]) &
                      ((test_index - pd.Timedelta(minutes=5)).hour < hcs.FREE_HOURS[1]))

    base = df.copy()
    base["net_demand"] = hcs.demand_side_nd_hist(base)
    fs_base = xs.transform(base.loc[full.index, fc].values).astype(np.float32)
    actual = test_df[hcs.TARGETS].to_numpy(dtype=np.float64)
    actual_resp = actual[mask]
    dem_b = base.loc[test_index, "demand_mw"].to_numpy(dtype=np.float64)
    prev0 = (fs_base[lb - 1, hcs.M.TARGET_FEAT_IDX] * xs.scale_[hcs.M.TARGET_FEAT_IDX]
             + xs.mean_[hcs.M.TARGET_FEAT_IDX]).astype(np.float64)

    rows = []
    for rebound in args.rebound:
        shift = FixedPercentageShift(rebound_pct=rebound,
                                     reduction_pct=args.reduction,
                                     free_hours=hcs.FREE_HOURS)
        scen = shift.transform(df)
        scen["net_demand"] = hcs.demand_side_nd_hist(scen)
        scen_full = scen.loc[full.index, "demand_mw"]
        if (scen_full < -1e-6).any() or (scen_full > args.demand_cap + 1e-6).any():
            raise ValueError(f"rebound={rebound:g}% violates demand cap/non-negativity")

        fs_scen = xs.transform(scen.loc[full.index, fc].values).astype(np.float32)
        dem_s = scen.loc[test_index, "demand_mw"].to_numpy(dtype=np.float64)
        nd_input_scen = scen.loc[test_index, "net_demand"].to_numpy(dtype=np.float64)
        t0 = time.time()
        print(f"rolling rebound={rebound:g}% reduction={args.reduction:g}% ...",
              flush=True)
        pb, ps, db, ds, alpha = hcs.rollout(
            model, n_out, fs_base, fs_scen, lb, free_test, xs, ys, device)
        b_resp, s_resp = pb[mask], ps[mask]
        base_wape, base_r2 = hcs.wape_r2_response(actual_resp, b_resp)
        nd_b, nd_s = b_resp @ hcs.SIGN, s_resp @ hcs.SIGN
        track = np.abs(ps @ hcs.SIGN - nd_input_scen)
        resid_in = ps @ hcs.SIGN - nd_input_scen
        bal = np.abs(ps @ hcs.SIGN - ds)
        rs = hcs.ramp_counts(ps, prev0)
        soc_ok, soc_pct = hcs.soc_swing_pct(ps)
        response = {}
        for i, target in enumerate(hcs.TARGETS):
            denom = b_resp[:, i].mean()
            response[target] = (100.0 * (s_resp[:, i].mean() - denom) / denom
                                 if abs(denom) > 1e-9 else np.nan)

        rows.append({
            "rebound_pct": rebound,
            "reduction_pct": args.reduction,
            "base_WAPE": base_wape,
            "base_R2": base_r2,
            "demand_in_pct": 100.0 * (dem_s[mask].mean() - dem_b[mask].mean()) /
                             dem_b[mask].mean(),
            "nd_resp_pct": 100.0 * (nd_s.mean() - nd_b.mean()) / nd_b.mean(),
            "coal_resp_pct": response["coal_brown"],
            "hydro_resp_pct": response["hydro"],
            "ocgt_resp_pct": response["gas_ocgt"],
            "batt_dis_resp_pct": response["battery_discharging"],
            "track_p50_mw": float(np.percentile(track, 50)),
            "track_p95_mw": float(np.percentile(track, 95)),
            "bal_own_max_mw": float(bal.max()),
            "n_netdemand_balance_violations": int((bal > hcs.DEMAND_TOL).sum()),
            "n_demand_actual": int((np.abs(resid_in) > hcs.DEMAND_TOL).sum()),
            "mismatch_actual_pct": float(100 * np.abs(resid_in).sum() /
                                           np.abs(nd_input_scen).sum()),
            "n_ramp_violations": rs["n"],
            "ramp_mean_mw": rs["mean_mw"],
            "ramp_max_mw": rs["max_mw"],
            "ramp_total_mwh": rs["total_mwh"],
            "n_negative_dispatch": int((ps < -0.1).sum()),
            "SOC_possible_starting": bool(soc_ok),
            "SOC_swing_pct": soc_pct,
            "alpha_active_pct": float(100 * (alpha > 1e-6).mean()) if len(alpha) else None,
        })
        print(f"  ramp={rs['n']} netd_viol={rows[-1]['n_netdemand_balance_violations']} "
              f"SOC={rows[-1]['SOC_possible_starting']} "
              f"({time.time() - t0:.1f}s)", flush=True)

    cols = list(rows[0])
    lines = [
        "# RayenFD rebound sweep (reduction 0%)\n",
        "Closed-loop historical demand-shift rollout over the full test set. "
        "RayenFD uses `nD(t-1)` and gas-steam passthrough; balance violations use "
        f"a {hcs.DEMAND_TOL:g} MW tolerance.\n",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, bool):
                cells.append(str(value))
            elif isinstance(value, (int, np.integer)):
                cells.append(str(value))
            elif value is None:
                cells.append("—")
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
