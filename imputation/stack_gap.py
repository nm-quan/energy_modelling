"""Stacked dispatch figure for the gap imputer — N consecutive test days (default 4),
mirroring demand_simulation/study_stack_4day.py. Three panels:

  actual        real dispatch across the N days
  imputed base  real days with each 11:00-14:00 gap replaced by the model's g=0 fill
  imputed +g%   the same, with a +g% demand fill in each gap

Each day's gap is shaded gold; the dashed line is net_demand (raised inside the gap
in the +g% panel). The two things to see: (1) the imputed gaps MATCH the actual
gaps (good reconstruction), and (2) the stack is CONTINUOUS across every gold edge
(11:00 and 14:00) -- no 2:05 seam, because the fill is pinned to both boundaries.

    python3 imputation/stack_gap.py --days 4 --g 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import torch                                 # noqa: E402
from gap_data import load_flats, test_gap_windows, TARGETS             # noqa: E402
from model import BiLSTMImputer              # noqa: E402
from scenario_eval import impute_project     # noqa: E402
import stack_plots as sp                     # noqa: E402

STACK = ["coal_brown", "gas_steam", "gas_ocgt", "hydro", "battery_discharging"]
OUT = HERE / "results" / "figure"


def consecutive_days(gws, n_days):
    """First run of n_days calendar-consecutive gap-days."""
    for i in range(len(gws) - n_days + 1):
        run = gws[i:i + n_days]
        if all((run[k + 1].day - run[k].day).days == 1 for k in range(n_days - 1)):
            return run
    raise SystemExit(f"no run of {n_days} consecutive test gap-days found")


def draw(ax, t, disp, nd_line, gaps):
    ti = {s: i for i, s in enumerate(TARGETS)}
    base = np.zeros(len(t))
    for k in STACK:
        top = base + disp[:, ti[k]]
        ax.fill_between(t, base, top, color=sp.COLORS[k], alpha=0.9, label=k)
        base = top
    ax.fill_between(t, 0, -disp[:, ti["battery_charging"]], color=sp.COLORS["battery_charging"],
                    alpha=0.6, label="battery_charging (load)")
    ax.plot(t, nd_line, "k--", lw=1.0, label="net demand")
    for p0, p1 in gaps:
        ax.axvspan(p0, p1, color="gold", alpha=0.15)
    ax.axhline(0, color="k", lw=0.6); ax.margins(x=0); ax.grid(True, axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--g", type=float, default=10.0)
    ap.add_argument("--context", type=int, default=48)
    ap.add_argument("--start", type=int, default=0, help="skip this many candidate runs")
    ap.add_argument("--ckpt", default=str(HERE / "results" / "bilstm_imputer.pt"))
    args = ap.parse_args()

    f = load_flats()
    gws = test_gap_windows(f, context=args.context)
    run = consecutive_days(gws[args.start:], args.days)
    model = BiLSTMImputer(); model.load_state_dict(torch.load(args.ckpt, map_location="cpu")); model.eval()

    # full-day flat rows for the N-day span
    idx = f.test_index; lb = f.Xte.shape[0] - len(idx)
    day_set = {gw.day for gw in run}
    pos = np.array([p for p in range(len(idx)) if idx[p].normalize() in day_set])
    span_rows = lb + pos
    stamps = idx[pos]
    disp = f.y_to_mw(f.Yte[span_rows])
    nd = f.col_mw(f.Xte, 6)[span_rows]
    base_fill = disp.copy(); scen_fill = disp.copy(); nd_scen = nd.copy()
    gaps = []
    for gw in run:
        Pb, _, _ = impute_project(model, f, gw, args.context, "cpu", 0.0)
        Ps, _, nds = impute_project(model, f, gw, args.context, "cpu", args.g)
        gpos = np.searchsorted(span_rows, gw.gap_idx)     # gap positions within the span
        base_fill[gpos] = Pb; scen_fill[gpos] = Ps; nd_scen[gpos] = nds
        gaps.append((gpos[0], gpos[-1]))

    t = np.arange(len(disp))
    f2, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True, sharey=True)
    for ax, (name, d, ndl) in zip(axes, [
            ("actual (real dispatch)", disp, nd),
            ("imputed — base (g=0): gaps match actual, continuous across every edge", base_fill, nd),
            (f"imputed — +{args.g:g}% demand: stack rises to the raised line, still continuous at 14:00",
             scen_fill, nd_scen)]):
        draw(ax, t, d, ndl, gaps)
        ax.set_title(name, loc="left", fontsize=10); ax.set_ylabel("MW")
    day_starts = [np.searchsorted(pos, np.where(idx.normalize() == d)[0][0]) for d in sorted(day_set)]
    axes[2].set_xticks(day_starts)
    axes[2].set_xticklabels([pd.Timestamp(d).strftime("%b %d") for d in sorted(day_set)])
    axes[2].set_xlabel(f"{args.days} consecutive test days (each 11:00–14:00 gap shaded gold)")
    axes[0].legend(loc="upper left", ncol=4, fontsize=7, framealpha=0.9)
    f2.suptitle(f"Gap imputation, {args.days} days {min(day_set).date()}–{max(day_set).date()} — "
                "real days + imputed 11:00–14:00 fills (bi-LSTM + ramp tube). "
                "No 2:05 seam: each fill is pinned to both boundaries.")
    f2.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"stack_gap_{args.days}day_g{args.g:g}.png"
    f2.savefig(out, dpi=140); plt.close(f2)
    print("wrote", out)


if __name__ == "__main__":
    main()
