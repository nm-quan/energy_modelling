"""Rebound sweep -- demand-side net_demand through the no-RevIN LSTM (teacher-forced).

Reconstructed generator for demand_simulation/sweep_eqnd/{result_sweep.md, sweep.csv,
figure/}. The dispatch model (lstm_5min_mse) is run one-shot (teacher-forced) over the
test set under a FixedPercentageShift demand scenario.

Unlike the pipeline's net_demand (= signed sum of dispatchable generation), here the
model's net_demand *input* feature is overridden with a DEMAND-SIDE quantity:

    net_demand = demand_mw - wind - solar_utility - net_import      (net_import = imports - exports)

The shift raises demand_mw inside the free window (11:00-14:00) by rebound_pct and lowers
it elsewhere by reduction_pct (energy-conserving load shift + rebound), which propagates
into this demand-side net_demand. The shifted demand/price/net_demand reach the LSTM via
the saved scaler. Caveat: the model was trained on dispatch-sum net_demand.

Reported per target: base/scenario mean over the response region (window edge inside the
free window) and the % change. The net_demand column is the signed sum of predicted
dispatch (pred @ SIGN), not the demand-side input.

Usage:
    python demand_simulation/sweep_eqnd.py                      # default sweep, reduction 0
    python demand_simulation/sweep_eqnd.py --reduction 10 --rebound 20
    python demand_simulation/sweep_eqnd.py --reduction 10 --rebound 10 20 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent              # demand_simulation/
ROOT = HERE.parent                                  # energy_modelling/
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                          # noqa: E402
import sim_common as sc                  # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

OUT = HERE / "sweep_eqnd"
FIG = OUT / "figure"
FREE_HOURS = (11, 14)                                # free window 11:00-14:00
HORIZON = 1


def demand_side_nd(frame: pd.DataFrame, net_import: pd.Series) -> pd.Series:
    """net_demand = demand_mw - wind - solar_utility - net_import (all aligned)."""
    return (frame["demand_mw"] - frame["wind"] - frame["solar_utility"]
            - net_import.reindex(frame.index))


def run(rebound: float, reduction: float, ctx, device) -> dict:
    """One shift scenario -> per-target base/scen means over the response region.

    Uses the closed-loop ar_free_rollout (nd_mode="scenario"): inside the free
    window the 6 dispatch-history channels feed back from the model's own
    predictions, while demand_mw / price / (demand-side) net_demand stay teacher-
    forced from the scenario frame. Outside the window every step is reseeded.
    """
    df, ni, feat_cols, x_scaler, y_scaler, model, lb, test_index, full_idx, free_test = (
        ctx["df"], ctx["ni"], ctx["feat_cols"], ctx["x_scaler"], ctx["y_scaler"],
        ctx["model"], ctx["lb"], ctx["test_index"], ctx["full_idx"], ctx["free_test"])

    shift = FixedPercentageShift(rebound_pct=rebound, reduction_pct=reduction,
                                 free_hours=FREE_HOURS)

    base = df.copy()
    base["net_demand"] = demand_side_nd(base, ni)
    scen = shift.transform(df)                       # shifts demand_mw + zeroes free price
    scen["net_demand"] = demand_side_nd(scen, ni)

    fs_base = x_scaler.transform(base.loc[full_idx, feat_cols].values).astype(np.float32)
    fs_scen = x_scaler.transform(scen.loc[full_idx, feat_cols].values).astype(np.float32)
    pred_b, pred_s = sc.ar_free_rollout(model, fs_base, fs_scen, lb, free_test,
                                        x_scaler, y_scaler, device, nd_mode="scenario")

    mask = sc.response_mask(test_index, HORIZON, FREE_HOURS)  # response region
    b, s = pred_b[mask], pred_s[mask]

    row = {"rebound": rebound}
    dem_b = base.loc[test_index, "demand_mw"].to_numpy()[mask]
    dem_s = scen.loc[test_index, "demand_mw"].to_numpy()[mask]
    row["demand_in_pct"] = 100.0 * (dem_s.mean() - dem_b.mean()) / dem_b.mean()

    nd_b, nd_s = sc.net_demand(b).mean(), sc.net_demand(s).mean()
    row.update(nd_base=nd_b, nd_scen=nd_s, nd_pct=100.0 * (nd_s - nd_b) / nd_b)
    for i, t in enumerate(sc.TARGETS):
        mb, ms = b[:, i].mean(), s[:, i].mean()
        row[f"{t}_base"] = mb
        row[f"{t}_scen"] = ms
        row[f"{t}_pct"] = 100.0 * (ms - mb) / mb if mb != 0 else np.nan
    return row


def write_md(dfres: pd.DataFrame, reduction: float, path: Path):
    cols = ["coal_brown", "hydro", "gas_ocgt", "battery_discharging"]
    lines = [
        "Rebound sweep -- lstm_5min_mse, DEMAND-SIDE net_demand (shifted, teacher-forced)",
        "",
        "net_demand = demand_mw - wind - solar_utility - net_import (net_import = imports - exports).",
        f"reduction {reduction:g}%, free window {FREE_HOURS[0]:02d}:00-{FREE_HOURS[1]:02d}:00. "
        "Means over the response region. net_demand column = signed sum of predicted dispatch.",
        "Caveat: model trained on dispatch-sum net_demand; demand-side net_demand is fed via the saved scaler.",
        "",
        "| rebound | demand_mw in | net_demand | coal_brown | hydro | gas_ocgt | battery_dis |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in dfres.iterrows():
        lines.append(
            f"| {r['rebound']:g}% | {r['demand_in_pct']:+.1f}% | {r['nd_pct']:+.1f}% | "
            f"{r['coal_brown_pct']:+.1f}% | {r['hydro_pct']:+.1f}% | "
            f"{r['gas_ocgt_pct']:+.1f}% | {r['battery_discharging_pct']:+.1f}% |")
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


def plot(dfres: pd.DataFrame, path: Path):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for t, lbl in [("coal_brown", "coal_brown"), ("hydro", "hydro"),
                   ("gas_ocgt", "gas_ocgt"), ("battery_discharging", "battery_dis"),
                   ("nd", "net_demand")]:
        ax.plot(dfres["rebound"], dfres[f"{t}_pct"], marker="o", label=lbl)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("rebound (%)")
    ax.set_ylabel("response vs base (%)")
    ax.set_title("Demand-side rebound response (LSTM, teacher-forced)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def build_ctx(device):
    df = pipeline.build_table("5min")
    inter = pd.read_parquet(pipeline.DATA_DIR / "vic_interconnector_last365.parquet")
    ni = inter.set_index("interval").sort_index()["net_import_mw"]
    ni = ni.reindex(df.index).interpolate(limit_direction="both")

    data = pipeline.prepare(resolution="5min", lookback=24, horizon=HORIZON,
                            input_mode="net_dispatch_totdem", save=False)
    lb = data["lookback_steps"]
    model = sc.load_lstm("lstm_5min_mse", len(data["feat_cols"]), device)

    val_df = df[(df.index > pipeline.TRAIN_END) & (df.index <= pipeline.VAL_END)]
    test_df = df[df.index > pipeline.VAL_END]
    test_index = test_df.index
    full_idx = pd.concat([val_df.tail(lb), test_df]).index
    free_test = np.asarray((test_index.hour >= FREE_HOURS[0])
                           & (test_index.hour < FREE_HOURS[1]))
    return {"df": df, "ni": ni, "feat_cols": data["feat_cols"],
            "x_scaler": data["x_scaler"], "y_scaler": data["y_scaler"],
            "model": model, "lb": lb, "test_index": test_index,
            "full_idx": full_idx, "free_test": free_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduction", type=float, default=0.0)
    ap.add_argument("--rebound", type=float, nargs="+",
                    default=list(range(10, 101, 10)))
    ap.add_argument("--no-write", action="store_true",
                    help="print only; do not overwrite bundled artifacts")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}; reduction={args.reduction:g}%; rebound={args.rebound}")
    ctx = build_ctx(device)

    rows = [run(reb, args.reduction, ctx, device) for reb in args.rebound]
    dfres = pd.DataFrame(rows)
    print(dfres[["rebound", "demand_in_pct", "nd_pct", "coal_brown_pct",
                 "hydro_pct", "gas_ocgt_pct", "battery_discharging_pct"]].to_string(index=False))

    if not args.no_write:
        OUT.mkdir(parents=True, exist_ok=True)
        # reduction 0 keeps the bundled names; other reductions get their own suffix
        sfx = "" if args.reduction == 0 else f"_red{args.reduction:g}"
        dfres.to_csv(OUT / f"sweep{sfx}.csv", index=False)
        print("wrote", OUT / f"sweep{sfx}.csv")
        write_md(dfres, args.reduction, OUT / f"result_sweep{sfx}.md")
        plot(dfres, FIG / f"response_vs_rebound{sfx}.png")


if __name__ == "__main__":
    main()
