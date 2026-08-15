"""What happens when the head is given the net demand you could actually measure?

planB feeds the head `nd = SIGN . dispatch`, computed from the six target
channels BEFORE they are masked (fit.py::_supply_side_nd). Its defence is that
net demand "is known at every step even inside the gap -- only the BREAKDOWN is
unknown".

That defence has a measurable hole, and the same docstring states it: the
quantity you can actually meter during an outage is the demand-side net demand
`demand - wind - solar - curtailment`, and it differs from `SIGN . dispatch` by
"mean 391 MW and up to 4,974 MW".

So there are two different numbers and the evaluation uses the one that is a
deterministic function of the labels. This script re-runs the same checkpoints
with the observable one substituted at inference, which is what deployment on a
real outage would have to do.

    python3 planB/ablation_observable_nd.py --n 400 --ckpt-dir colab/planB_results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "imputation")); sys.path.insert(0, str(HERE))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "prepared.npz"
from gap_data import load_flats, TARGETS, SIGN, TARGET_FEAT_IDX, ND_COL  # noqa: E402
from nets import build_delta, n_params                                 # noqa: E402
from heads import HEADS                                                # noqa: E402
import fit as FIT                                                      # noqa: E402
from fit import build, CTX, GAP, apply_residual, BACKBONES             # noqa: E402

OUT = HERE / "results"
SHORT = {"coal_brown": "coal", "gas_ocgt": "ocgt", "gas_steam": "steam",
         "hydro": "hydro", "battery_charging": "bat_chg",
         "battery_discharging": "bat_dis"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ckpt-dir", default="colab/planB_results")
    args = ap.parse_args()
    ckdir = ROOT / args.ckpt_dir

    f = load_flats()
    nfeat = len(f.feat_cols)
    W = 2 * CTX + GAP
    rng = np.random.default_rng(args.seed)
    lo = max(0, f.lb_carry - CTX)
    starts = rng.integers(lo, f.Xte.shape[0] - W, size=args.n)

    # --- observable nd: build WITHOUT the supply-side overwrite ---------------
    real = FIT._supply_side_nd
    FIT._supply_side_nd = lambda X, f_, tfi: X          # disable
    b_obs = build(f, starts, "test")
    FIT._supply_side_nd = real
    b = build(f, starts, "test")                        # the leaked version

    gs = CTX
    nd_leak = b["nd"]                                                   # (n,G)
    nd_obs = (b_obs["X"][:, gs:gs + GAP, ND_COL] * f.x_scale[ND_COL]
              + f.x_mean[ND_COL]).astype(np.float32)
    d = nd_leak - nd_obs
    print(f"nd(supply-side, used) - nd(observable): mean {d.mean():+.1f} MW  "
          f"mean|.| {np.abs(d).mean():.1f}  max|.| {np.abs(d).max():.1f}")

    truth = b["Y"] * f.y_scale + f.y_mean
    ys_m = torch.tensor(f.y_mean, dtype=torch.float32)
    ys_s = torch.tensor(f.y_scale, dtype=torch.float32)
    pL, pR = torch.from_numpy(b["pL"]), torch.from_numpy(b["pR"])

    rows = []
    for arm in ["hardnet"]:
        ck = torch.load(ckdir / f"{arm}.pt", map_location="cpu", weights_only=False)
        head_fn, n_disp = HEADS[ck["head"]]
        bb = ck.get("backbone", "bilstm")
        m = BACKBONES[bb](nfeat, n_disp, hidden=ck["hidden"])
        m.load_state_dict(ck["state"]); m.eval()
        resid = ck.get("residual", False)

        for label, Xsrc, ndv in [
                ("as published (nd = SIGN·truth, in features AND head)",
                 b["X"], nd_leak),
                ("observable nd in the head only", b["X"], nd_obs),
                ("observable nd in features AND head", b_obs["X"], nd_obs)]:
            o = []
            for i in range(0, args.n, 64):
                sl = slice(i, i + 64)
                x = torch.from_numpy(Xsrc[sl]); mk = torch.from_numpy(b["mask"][sl])
                with torch.no_grad():
                    dl = build_delta(mk, nfeat) if bb == "brits" else None
                    d_raw, _ = m(x, mk, dl)
                    d_raw = d_raw[:, CTX:CTX + GAP]
                    if resid:
                        d_raw = apply_residual(d_raw, torch.from_numpy(b["interp"][sl]))
                    F_ = d_raw * ys_s + ys_m
                    P = head_fn(F_, pL[sl], pR[sl], torch.from_numpy(ndv[sl]))
                o.append(P.numpy())
            P = np.concatenate(o, 0)
            e = np.abs(P - truth).reshape(-1, 6)
            bal_true = float(np.abs((P * SIGN).sum(-1) - nd_leak).max())
            rows.append((f"{arm} — {label}", e.mean(0), float(e.mean()), bal_true))

    # parameter-free reference under the same substitution
    tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
    interp = b["pL"][:, None] + tt * (b["pR"] - b["pL"])[:, None]
    head, _ = HEADS["hardnet"]
    for label, ndv in [("interp + head, nd = SIGN·truth", nd_leak),
                       ("interp + head, observable nd", nd_obs)]:
        o = []
        for i in range(0, args.n, 64):
            sl = slice(i, i + 64)
            with torch.no_grad():
                o.append(head(torch.tensor(interp[sl], dtype=torch.float32),
                              pL[sl], pR[sl], torch.from_numpy(ndv[sl])).numpy())
        P = np.concatenate(o, 0)
        e = np.abs(P - truth).reshape(-1, 6)
        rows.append((label, e.mean(0), float(e.mean()),
                     float(np.abs((P * SIGN).sum(-1) - nd_leak).max())))

    L = ["# What the constraint head costs when net demand is not the labels", "",
         f"{args.n} random test 3h gaps, seed {args.seed}, checkpoints from "
         f"`{args.ckpt_dir}`.", "",
         f"`SIGN·dispatch` minus the observable `demand − wind − solar − curtailment`, "
         f"inside the gap: mean {d.mean():+.0f} MW, mean absolute "
         f"{np.abs(d).mean():.0f} MW, max {np.abs(d).max():,.0f} MW.", "",
         "## MAE (MW)", "",
         "| configuration | " + " | ".join(SHORT[c] for c in TARGETS) +
         " | **aggregate** | balance vs true nd |",
         "| --- |" + " ---: |" * (len(TARGETS) + 2)]
    for name, mae, agg, bal in rows:
        L.append(f"| {name} | " + " | ".join(f"{v:,.1f}" for v in mae) +
                 f" | **{agg:,.1f}** | {bal:,.1f} |")
    L.append("")
    (OUT / "ablation_observable_nd.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
