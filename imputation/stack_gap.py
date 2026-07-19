"""Stacked dispatch figure for the gap imputer — one day, 3 panels:

  actual        real dispatch across the whole window
  imputed base  real context + the model's g=0 fill inside the 11:00-14:00 gap
  imputed +g%   real context + the model's raised-demand fill inside the gap

Shows the two things the reframe buys: (1) the imputed gap MATCHES the actual gap
(good reconstruction), and (2) the fill connects SMOOTHLY to the real dispatch at
BOTH 11:00 and 14:00 -- no 2:05 seam -- because it is pinned to both boundaries.
The dashed line is net_demand (raised inside the gap in the +g% panel); the stack
rising to meet it there is the response.

    python3 imputation/stack_gap.py --day-index 40 --g 10
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
from gap_data import load_flats, test_gap_windows, TARGETS, SIGN         # noqa: E402
from model import BiLSTMImputer              # noqa: E402
from scenario_eval import impute_project     # noqa: E402
import stack_plots as sp                     # noqa: E402

STACK = ["coal_brown", "gas_steam", "gas_ocgt", "hydro", "battery_discharging"]
OUT = HERE / "results" / "figure"


def window_dispatch(f, gw):
    """Real MW dispatch (TARGETS order) across [ctxL | gap | ctxR] and its stamps."""
    rows = np.concatenate([gw.ctxL_idx, gw.gap_idx, gw.ctxR_idx])
    disp = f.y_to_mw(f.Yte[rows])
    nd = f.col_mw(f.Xte, 6)[rows]
    lb = f.Xte.shape[0] - len(f.test_index)
    stamps = f.test_index[rows - lb]
    return disp, nd, stamps, len(gw.ctxL_idx)


def draw(ax, t, disp, nd_line, gap0, gap1):
    ti = {s: i for i, s in enumerate(TARGETS)}
    base = np.zeros(len(t))
    for k in STACK:
        top = base + disp[:, ti[k]]
        ax.fill_between(t, base, top, color=sp.COLORS[k], alpha=0.9, label=k)
        base = top
    ax.fill_between(t, 0, -disp[:, ti["battery_charging"]], color=sp.COLORS["battery_charging"],
                    alpha=0.6, label="battery_charging (load)")
    ax.plot(t, nd_line, "k--", lw=1.1, label="net demand")
    ax.axvspan(t[gap0], t[gap1 - 1], color="gold", alpha=0.15)
    ax.axvline(t[gap0], color="grey", lw=0.6); ax.axvline(t[gap1 - 1], color="grey", lw=0.6)
    ax.axhline(0, color="k", lw=0.6); ax.margins(x=0); ax.grid(True, axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-index", type=int, default=40)
    ap.add_argument("--g", type=float, default=10.0)
    ap.add_argument("--context", type=int, default=48)
    ap.add_argument("--ckpt", default=str(HERE / "results" / "bilstm_imputer.pt"))
    args = ap.parse_args()

    f = load_flats()
    gws = test_gap_windows(f, context=args.context)
    gw = gws[args.day_index % len(gws)]
    model = BiLSTMImputer(); model.load_state_dict(torch.load(args.ckpt, map_location="cpu")); model.eval()

    disp, nd, stamps, g0 = window_dispatch(f, gw)
    N = len(gw.gap_idx); g1 = g0 + N
    t = np.arange(len(disp))
    Pb, _, ndb = impute_project(model, f, gw, args.context, "cpu", 0.0)
    Ps, _, nds = impute_project(model, f, gw, args.context, "cpu", args.g)

    base_fill = disp.copy(); base_fill[g0:g1] = Pb          # real context + g=0 fill
    scen_fill = disp.copy(); scen_fill[g0:g1] = Ps          # real context + +g% fill
    nd_scen = nd.copy(); nd_scen[g0:g1] = nds

    f2, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, sharey=True)
    for ax, (name, d, ndl) in zip(axes, [
            ("actual (real dispatch)", disp, nd),
            ("imputed — base (g=0): matches actual, smooth at both edges", base_fill, nd),
            (f"imputed — +{args.g:g}% demand: stack rises to the raised line, still smooth at 14:00",
             scen_fill, nd_scen)]):
        draw(ax, t, d, ndl, g0, g1)
        ax.set_title(name, loc="left", fontsize=10); ax.set_ylabel("MW")
    ticks = np.linspace(0, len(t) - 1, 8).astype(int)
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels([pd.Timestamp(stamps[i]).strftime("%H:%M") for i in ticks])
    axes[2].set_xlabel(f"time of day — {pd.Timestamp(stamps[g0]).date()} (gap 11:00–14:00 shaded)")
    axes[0].legend(loc="upper left", ncol=4, fontsize=7, framealpha=0.9)
    f2.suptitle("Gap imputation — real context + imputed 11:00–14:00 fill "
                "(bi-LSTM + ramp tube). No 2:05 seam: the fill is pinned to both boundaries.")
    f2.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"stack_gap_day{args.day_index}_g{args.g:g}.png"
    f2.savefig(out, dpi=140); plt.close(f2)
    print("wrote", out)


if __name__ == "__main__":
    main()
