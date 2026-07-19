"""4-day all-energy stacked dispatch chart for the hist itransformer_rayen model.

The hist analogue of stack_shift_4day.py: three panels (actual / baseline
no-shift / scenario shift) over a run of 4 consecutive full test days, stacking
every energy fuel:

    coal_brown, gas_steam, gas_ocgt, hydro, battery_discharging,
    wind (+ wind curtailment, hatched), solar_utility (+ solar curtailment)

battery_charging is drawn below 0 (load). Renewables + curtailment are ACTUAL
(the constrained model only predicts the 6 dispatchable targets), so those
layers are identical in every panel; only the dispatchable layers respond to the
shift. The daily free window (11:00-14:00) is shaded gold.

Baseline / scenario dispatch come from the same closed-loop free-window rollout
used by hist_constrained_shift.py (hist demand-side net_demand, no import
subtraction). Because the shift is capped by the hist demand cap, the default
rebound = reduction = the maximum equal feasible shift (q_max ~ 4.86%).

The last365 renewables parquet only covers through 2026-05-18, so days are
picked from the covered portion of the test set to keep every renewable layer
populated.

    python3 demand_simulation/hist_stack_shift_4day.py
    python3 demand_simulation/hist_stack_shift_4day.py --rebound 2 --reduction 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "ml"))

import evaluate as ev            # noqa: E402
import pipeline                  # noqa: E402
import stack_plots as sp         # noqa: E402
import hist_constrained_shift as hcs  # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402
from stack_shift_4day import panels_multiday  # noqa: E402

OUT = HERE / "sweep_eqnd" / "figure"
FREE_HOURS = hcs.FREE_HOURS       # (11, 14)


def renewables_cov_max() -> pd.Timestamp:
    """Last interval covered by BOTH the generation and curtailment parquets."""
    gen = pd.read_parquet(sp.DATA / "vic_generation_last365.parquet")
    cur = pd.read_parquet(sp.DATA / "vic_curtailment_last365.parquet")
    gmax = pd.to_datetime(gen["interval"]).max()
    cmax = pd.to_datetime(cur["interval"]).max()
    return min(gmax, cmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights"))
    ap.add_argument("--model", default="itransformer_rayenfd",
                    help="model name in hcs.load_entries (default: the fixed-D RAYEN arm)")
    ap.add_argument("--rayenfd-steam-pt", choices=["on", "off"], default="on",
                    help="freeze gas_steam at persistence for rayenfd (matches +spt results)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--rebound", type=float, default=None,
                    help="free-window rebound %% (default: max equal feasible q_max)")
    ap.add_argument("--reduction", type=float, default=None,
                    help="off-window reduction %% (default: max equal feasible q_max)")
    ap.add_argument("--demand-cap", type=float, default=hcs.DEMAND_CAP_MW)
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = ev.pick_device(args.device)
    print(f"device={device}")

    data = pipeline.load_prepared("5min", 24, 1, "net_dispatch_totdem", "hist")
    xs, ys, fc = data["x_scaler"], data["y_scaler"], data["feat_cols"]
    hcs.data_feat_cols = fc                       # rollout() reads this module global
    lb = data["lookback_steps"]

    entries = hcs.load_entries(
        Path(args.weights), data, device, model_filter=[args.model],
        rayenfd_steam_pt=args.rayenfd_steam_pt == "on")
    name, n_out, model = next(e for e in entries if e[0] == args.model)
    print(f"model: {name} (n_out={n_out})")

    df = pipeline.build_table("5min", "hist")
    val_end = pipeline.DATASETS["hist"]["val_end"]
    train_end = pipeline.DATASETS["hist"]["train_end"]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df_full = df[df.index > val_end]
    test_index_full = test_df_full.index

    # Pick the 4-day window from the renewables-covered portion of the test set,
    # then only roll out far enough to cover it (positions stay valid because the
    # covered range is a prefix of the full test set).
    cov_max = renewables_cov_max()
    covered = test_index_full[test_index_full <= cov_max]
    if len(covered) < args.days * 288:
        raise ValueError("not enough renewables-covered test days for the figure")
    days, pos = sp.pick_full_days(pd.DatetimeIndex(covered), n_days=args.days, seed=args.seed)
    print("days:", [str(d) for d in days])

    end = int(pos.max()) + 1
    test_df = test_df_full.iloc[:end]
    test_index = test_df.index
    full = pd.concat([val_df.tail(lb), test_df])
    free_test = np.asarray((test_index.hour >= FREE_HOURS[0])
                           & (test_index.hour < FREE_HOURS[1]))

    # Scenario: capped equal shift on hist demand_mw, hist-style net_demand.
    qmax = hcs.max_equal_shift_pct(full, args.demand_cap, FREE_HOURS)
    rebound = qmax if args.rebound is None else args.rebound
    reduction = qmax if args.reduction is None else args.reduction
    print(f"rebound={rebound:.4f}% reduction={reduction:.4f}% (q_max={qmax:.4f}%)")
    shift = FixedPercentageShift(rebound_pct=rebound, reduction_pct=reduction,
                                 free_hours=FREE_HOURS)

    base = df.copy(); base["net_demand"] = hcs.demand_side_nd_hist(base)
    scen = shift.transform(df); scen["net_demand"] = hcs.demand_side_nd_hist(scen)
    scen_full = scen.loc[full.index, "demand_mw"]
    if (scen_full < -1e-6).any() or (scen_full > args.demand_cap + 1e-6).any():
        raise ValueError("scenario violates demand cap/non-negativity; lower rebound/reduction")

    fs_base = xs.transform(base.loc[full.index, fc].values).astype(np.float32)
    fs_scen = xs.transform(scen.loc[full.index, fc].values).astype(np.float32)

    print(f"rolling out (closed loop, {len(free_test)} steps)...", flush=True)
    t0 = time.time()
    pred_b, pred_s, _, _, _ = hcs.rollout(model, n_out, fs_base, fs_scen, lb,
                                          free_test, xs, ys, device)
    print(f"  done ({time.time() - t0:.0f}s)")

    actual = test_df[sp.TARGETS].to_numpy()
    ren = sp.load_renewables(pd.DatetimeIndex(test_index))

    OUT.mkdir(parents=True, exist_ok=True)
    sfx = f"_reb{rebound:g}_red{reduction:g}"
    out = OUT / f"stacked_all_energy_{args.model}_{args.days}day{sfx}.png"
    panels_multiday(test_index[pos], actual[pos], pred_b[pos], pred_s[pos],
                    ren.iloc[pos], out,
                    tag=f"{name} (hist) — shift reb{rebound:g}% red{reduction:g}%")


if __name__ == "__main__":
    main()
