"""What interpolation does the rayen head actually use, and how far does the
network move away from it?

rayen does NOT use the `interp` residual skeleton from fit.build() -- that is
only wired in when n_disp == 6, and rayen has 7 outputs. Its interpolation lives
INSIDE the head, in two stages, recomputed at every one of the 36 steps:

    glide   c = clip( P_prev + (pR - P_prev)/(k+1),  lo, hi )      k = steps left
    snap    A = c + (delta / sum room) * (room * s)                delta = nd - s.c
    ray     P = A + (alpha * amax) * r_hat                         alpha = sigmoid(logit)

So there are three curves worth separating, and the gap between them is the
whole question:

    actual      the truth
    glide       plain interpolation toward pR, before balance is imposed
    anchor A    after the room-weighted balance snap  <- what the ray starts from
    rayen P     the final output                      <- A plus the network's ray

If P sits on top of A, the network contributed nothing and the answer is the
anchor. That is exactly the alpha* collapse this head was built to fix.

    python3 planB/rayen_anchor.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "imputation"))
sys.path.insert(0, str(ROOT / "dispatch_study"))
sys.path.insert(0, str(HERE))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
from gap_data import load_flats, TARGETS, SIGN, TARGET_FEAT_IDX        # noqa: E402
from common import COLORS, LABEL, INK, MUTED                           # noqa: E402
from nets import build_delta                                           # noqa: E402
from heads import HEADS, rayen_head                                    # noqa: E402
from fit import build, CURT_COLS, CTX, GAP, BACKBONES                  # noqa: E402
from peak_eval import peak_starts                                      # noqa: E402

OUT = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="rayen")
    ap.add_argument("--hour", type=int, default=18)
    args = ap.parse_args()

    f = load_flats()
    nfeat = len(f.feat_cols)
    starts, days = peak_starts(f, args.hour)
    b = build(f, starts, "test")
    n, end_h = len(starts), args.hour + GAP * 5 // 60
    truth = b["Y"] * f.y_scale + f.y_mean

    ck = torch.load(OUT / f"{args.arm}.pt", map_location="cpu", weights_only=False)
    m = BACKBONES["bilstm"](nfeat, ck["n_disp"], hidden=ck["hidden"])
    m.load_state_dict(ck["state"]); m.eval()
    ys_m = torch.tensor(f.y_mean, dtype=torch.float32)
    ys_s = torch.tensor(f.y_scale, dtype=torch.float32)

    P, A, G = [], [], []
    for i in range(0, n, 64):
        sl = slice(i, i + 64)
        with torch.no_grad():
            d_raw, _ = m(torch.from_numpy(b["X"][sl]), torch.from_numpy(b["mask"][sl]), None)
            d_raw = d_raw[:, CTX:CTX + GAP]
            F_ = torch.cat([d_raw[..., :6] * ys_s + ys_m, d_raw[..., 6:]], -1)
            p, dbg = rayen_head(F_, torch.from_numpy(b["pL"][sl]),
                                torch.from_numpy(b["pR"][sl]),
                                torch.from_numpy(b["nd"][sl]), return_debug=True)
        P.append(p.numpy()); A.append(dbg["anchor"].numpy()); G.append(dbg["glide"].numpy())
    P, A, G = (np.concatenate(x, 0) for x in (P, A, G))

    # plain linear interpolation pL -> pR, the no-model reference, for contrast
    tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
    LIN = b["pL"][:, None] + tt * (b["pR"] - b["pL"])[:, None]

    ORDER = ["coal_brown", "hydro", "battery_discharging",
             "gas_ocgt", "gas_steam", "battery_charging"]
    ti = {t: i for i, t in enumerate(TARGETS)}
    hx = np.arange(GAP) * 5.0 / 60.0 + args.hour

    fig, axes = plt.subplots(2, 3, figsize=(14, 6.6), sharex=True)
    for ax, ch in zip(axes.ravel(), ORDER):
        k = ti[ch]
        ax.plot(hx, truth[..., k].mean(0), color=INK, lw=2.2, label="actual", zorder=5)
        ax.plot(hx, LIN[..., k].mean(0), color="#B0B4BA", lw=1.4, ls=":",
                label="plain interp (pL$\\to$pR)", zorder=2)
        ax.plot(hx, G[..., k].mean(0), color="#7FB2E5", lw=1.5, ls="-.",
                label="glide $c$  (stage 1)", zorder=3)
        ax.plot(hx, A[..., k].mean(0), color="#E9A020", lw=1.7, ls="--",
                label="anchor $A$  (stage 2)", zorder=4)
        ax.plot(hx, P[..., k].mean(0), color=COLORS[ch], lw=1.9,
                label="rayen output $P$", zorder=6)
        pull = np.abs(P[..., k] - A[..., k]).mean()
        ax.set_title(f"{LABEL[ch]}    ray moves {pull:.1f} MW off the anchor",
                     loc="left", fontsize=9, color=INK, pad=3)
        ax.grid(True, alpha=0.15, lw=0.5); ax.margins(x=0)
        ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#D6D8DC")
        ax.set_ylabel("MW", fontsize=7.5, color=MUTED)
    axes[0, 0].legend(loc="lower right", fontsize=7.2, frameon=False)
    fig.suptitle(f"The interpolation inside the rayen head — {args.hour:02d}:00-"
                 f"{end_h:02d}:00, day-averaged over {n} test days",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = OUT / f"rayen_anchor_{args.hour:02d}_{end_h:02d}.png"
    fig.savefig(png, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {png}\n")

    # ---- the numbers behind the picture -----------------------------------
    def mae(x):
        return np.abs(x - truth).reshape(-1, 6).mean(0)
    print(f"MAE against truth, MW ({n} days x {GAP} steps)\n")
    print(f"  {'channel':22s} {'plain':>8s} {'glide':>8s} {'anchor':>8s} "
          f"{'rayen P':>8s} {'|P-A|':>8s}")
    for c_, k in ti.items():
        print(f"  {c_:22s} {mae(LIN)[k]:8.1f} {mae(G)[k]:8.1f} {mae(A)[k]:8.1f} "
              f"{mae(P)[k]:8.1f} {np.abs(P[...,k]-A[...,k]).mean():8.1f}")
    print(f"  {'aggregate':22s} {mae(LIN).mean():8.1f} {mae(G).mean():8.1f} "
          f"{mae(A).mean():8.1f} {mae(P).mean():8.1f} "
          f"{np.abs(P-A).mean():8.1f}")
    # how much of the distance from the anchor to the wall does the ray use?
    span = np.abs(P - A).sum(-1)
    print(f"\n  balance check   |s.P - nd| max = {np.abs((P*SIGN).sum(-1)-b['nd']).max():.2e} MW")
    print(f"  the ray is idle on {100*(span < 1.0).mean():.1f}% of steps "
          f"(total move under 1 MW across all six channels)")


if __name__ == "__main__":
    main()
