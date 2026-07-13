"""Sweep rebound=reduction=q (all WITHIN the fleet ramp-feasibility limit) for one
rayen model, on the hist closed-loop free-window rollout. One markdown row per q,
with demand-ramp feasibility recorded next to the model's response + ramp metrics.

  python3 demand_simulation/sweep_feasible.py --model lstm_rayen --qs 1,2,3,4,4.4
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib")); sys.path.insert(0, str(ROOT / "ml"))
import hist_constrained_shift as H
import pipeline, models as M
from check_caps import RAMPS
from shift_model import FixedPercentageShift

TARGETS, SIGN, FREE = H.TARGETS, H.SIGN, H.FREE_HOURS
# fleet ramp ceiling (best case, all units full headroom)
UP_ALL = (RAMPS["hydro"][1] + RAMPS["coal_brown"][1] + RAMPS["gas_steam"][1] + RAMPS["gas_ocgt"][1]
          + RAMPS["battery_discharging"][1] + (-RAMPS["battery_charging"][0]))
DN_ALL = (-(RAMPS["hydro"][0] + RAMPS["coal_brown"][0] + RAMPS["gas_steam"][0] + RAMPS["gas_ocgt"][0])
          + (-RAMPS["battery_discharging"][0]) + RAMPS["battery_charging"][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm_rayen")
    ap.add_argument("--qs", default="1,2,3,4,4.4")
    ap.add_argument("--out", default=str(HERE / "sweep_eqnd" / "sweep_feasible.md"))
    ap.add_argument("--max-steps", type=int, default=None, help="smoke-test cap on test length")
    args = ap.parse_args()
    qs = [float(x) for x in args.qs.split(",")]

    import evaluate as ev
    device = ev.pick_device(None); print(f"device={device}")
    data = pipeline.load_prepared("5min", 24, 1, "net_dispatch_totdem", "hist")
    xs, ys, fc = data["x_scaler"], data["y_scaler"], data["feat_cols"]
    H.data_feat_cols = fc
    lb = data["lookback_steps"]
    name, n_out, model = H.load_entries(ROOT / "weights", data, device, [args.model])[0]

    df = pipeline.build_table("5min", "hist")
    ve = pipeline.DATASETS["hist"]["val_end"]
    val_df = df[(df.index > pipeline.DATASETS["hist"]["train_end"]) & (df.index <= ve)]
    test_df = df[df.index > ve]
    if args.max_steps is not None:
        test_df = test_df.iloc[:args.max_steps]
    full = pd.concat([val_df.tail(lb), test_df]); test_index = test_df.index
    free_test = np.asarray((test_index.hour >= FREE[0]) & (test_index.hour < FREE[1]))
    mask = np.asarray((test_index - pd.Timedelta(minutes=5)).hour >= FREE[0]) & \
           np.asarray((test_index - pd.Timedelta(minutes=5)).hour < FREE[1])
    actual = test_df[TARGETS].to_numpy(float); actual_resp = actual[mask]

    base = df.copy(); base["net_demand"] = H.demand_side_nd_hist(base)
    fs_base = xs.transform(base.loc[full.index, fc].values).astype(np.float32)
    dem_b = base.loc[test_index, "demand_mw"].to_numpy(float)

    rows = []
    for q in qs:
        t0 = time.time()
        if device == "mps":
            torch.mps.empty_cache()      # release the previous rollout's allocator cache (~9GB) between q's
        scen = FixedPercentageShift(q, q, free_hours=FREE).transform(df)
        scen["net_demand"] = H.demand_side_nd_hist(scen)
        fs_scen = xs.transform(scen.loc[full.index, fc].values).astype(np.float32)
        nd_input = scen.loc[test_index, "net_demand"].to_numpy(float)
        dem_s = scen.loc[test_index, "demand_mw"].to_numpy(float)
        prev0 = (fs_scen[lb - 1, M.TARGET_FEAT_IDX] * xs.scale_[M.TARGET_FEAT_IDX] + xs.mean_[M.TARGET_FEAT_IDX]).astype(float)
        # demand-ramp feasibility for this q
        dnd = np.diff(nd_input); dmax_up, dmax_dn = float(dnd.max()), float(-dnd.min())
        feasible = (dmax_up <= UP_ALL) and (dmax_dn <= DN_ALL)

        pb, ps, db, ds, alpha = H.rollout(model, n_out, fs_base, fs_scen, lb, free_test, xs, ys, device)
        b_resp, s_resp = pb[mask], ps[mask]
        base_wape, _ = H.wape_r2_response(actual_resp, b_resp)
        nd_b, nd_s = b_resp @ SIGN, s_resp @ SIGN
        rs = H.ramp_counts(ps, prev0)
        soc_ok, soc_pct = H.soc_swing_pct(ps)
        resp = {t: (100.0 * (s_resp[:, i].mean() - b_resp[:, i].mean()) / b_resp[:, i].mean()
                    if abs(b_resp[:, i].mean()) > 1e-9 else np.nan) for i, t in enumerate(TARGETS)}
        rows.append(dict(
            q=q, feasible="yes" if feasible else "NO",
            dem_ramp_up=dmax_up, dem_ramp_dn=dmax_dn,
            demand_in_pct=100.0 * (dem_s[mask].mean() - dem_b[mask].mean()) / dem_b[mask].mean(),
            nd_resp_pct=100.0 * (nd_s.mean() - nd_b.mean()) / nd_b.mean(),
            coal_resp_pct=resp["coal_brown"], hydro_resp_pct=resp["hydro"],
            ocgt_resp_pct=resp["gas_ocgt"], battdis_resp_pct=resp["battery_discharging"],
            base_WAPE=base_wape, n_ramp=rs["n"], ramp_mean_mwh=rs["mean_mwh"],
            ramp_max_mw=rs["max_mw"], ramp_total_mwh=rs["total_mwh"],
            soc_feasible="yes" if soc_ok else "no"))
        print(f"  q={q}: feas={rows[-1]['feasible']} dem_in={rows[-1]['demand_in_pct']:+.1f}% "
              f"nd_resp={rows[-1]['nd_resp_pct']:+.2f}% n_ramp={rs['n']} "
              f"ramp_mean={rs['mean_mwh']:.2f}MWh ({time.time()-t0:.0f}s)", flush=True)

    cols = ["q", "feasible", "dem_ramp_up", "dem_ramp_dn", "demand_in_pct", "nd_resp_pct",
            "coal_resp_pct", "hydro_resp_pct", "ocgt_resp_pct", "battdis_resp_pct",
            "base_WAPE", "n_ramp", "ramp_mean_mwh", "ramp_max_mw", "ramp_total_mwh", "soc_feasible"]
    lines = [f"# {name} — feasible demand-shift sweep (rebound = reduction = q)\n",
             f"Fleet ramp ceiling (gens+battery, full headroom): up {UP_ALL:.0f} / down {DN_ALL:.0f} MW per 5-min. "
             f"All q here are within the 4.47% ramp-feasible limit. Closed-loop free-window rollout, hist test set. "
             f"`dem_ramp_*` = worst 5-min net_demand ramp under the shift; `feasible` = within the fleet ceiling.\n",
             "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(H.fmt(r.get(c)) for c in cols) + " |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
