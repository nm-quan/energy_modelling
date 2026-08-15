"""How much of planB's score is the network, and how much is the head plus nd?

The head is handed `nd` INSIDE the gap, and gap_data verifies
`SIGN . truth == net_demand` exactly. So at every masked step the head receives
one exact linear equation in the six dispatch channels, whose right-hand side is
a deterministic function of the very values being predicted. Six unknowns, one
exact equation, per step.

This script removes the network and keeps everything else, so the residual is
attributable:

  interp            linear interpolation between the pinned seams. No nd.
  interp+head       the SAME interpolation pushed through hardnet with the true
                    nd. No learned parameters at all.
  persist+head      pL held flat across the gap, then hardnet. Even weaker.
  zeros+head        F = 0, so the head's projection alone picks the allocation.
  <arm>             the trained checkpoint.

If interp+head is at or below the trained arm, the network is not what is
producing the number.

    python3 planB/ablation_head_vs_network.py --n 400
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
from gap_data import load_flats, TARGETS, SIGN, TARGET_FEAT_IDX        # noqa: E402
from nets import build_delta, n_params                                 # noqa: E402
from heads import HEADS                                                # noqa: E402
from fit import (build, curt_activation, CURT_COLS, CTX, GAP,          # noqa: E402
                 apply_residual, BACKBONES)

OUT = HERE / "results"
SHORT = {"coal_brown": "coal", "gas_ocgt": "ocgt", "gas_steam": "steam",
         "hydro": "hydro", "battery_charging": "bat_chg",
         "battery_discharging": "bat_dis"}
ARMS_TO_TEST = ["hardnet", "rayen", "unconstrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ckpt-dir", default=str(OUT),
                    help="where the .pt files live (colab/planB_results for the "
                         "fully-trained run)")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args()
    ckdir = Path(args.ckpt_dir)

    f = load_flats()
    nfeat = len(f.feat_cols)
    W = 2 * CTX + GAP
    rng = np.random.default_rng(args.seed)
    lo = max(0, f.lb_carry - CTX)
    starts = rng.integers(lo, f.Xte.shape[0] - W, size=args.n)
    b = build(f, starts, "test")
    truth = b["Y"] * f.y_scale + f.y_mean                              # (n,G,6) MW
    ys_m = torch.tensor(f.y_mean, dtype=torch.float32)
    ys_s = torch.tensor(f.y_scale, dtype=torch.float32)

    pL = torch.from_numpy(b["pL"]); pR = torch.from_numpy(b["pR"])
    nd = torch.from_numpy(b["nd"])
    head, _ = HEADS["hardnet"]

    tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
    interp = b["pL"][:, None] + tt * (b["pR"] - b["pL"])[:, None]
    persist = np.repeat(b["pL"][:, None], GAP, 1)

    preds = {"interp (no nd)": interp}

    def run_head(F):
        o = []
        for i in range(0, args.n, 64):
            sl = slice(i, i + 64)
            with torch.no_grad():
                o.append(head(torch.tensor(F[sl], dtype=torch.float32),
                              pL[sl], pR[sl], nd[sl]).numpy())
        return np.concatenate(o, 0)

    preds["interp + head"] = run_head(interp)
    preds["persistence + head"] = run_head(persist)
    preds["zeros + head"] = run_head(np.zeros_like(interp))

    for arm in ARMS_TO_TEST:
        p = ckdir / f"{arm}.pt"
        if not p.exists():
            print(f"  [skip] {p.name} missing"); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        head_fn, n_disp = HEADS[ck["head"]]
        bb = ck.get("backbone", "bilstm")
        m = (BACKBONES[bb](nfeat, n_disp, hidden=ck["hidden"]) if bb == "bilstm"
             else BACKBONES[bb](nfeat, TARGET_FEAT_IDX, CURT_COLS, n_disp,
                                hidden=ck["hidden"]))
        m.load_state_dict(ck["state"]); m.eval()
        resid = ck.get("residual", False)
        o = []
        for i in range(0, args.n, 64):
            sl = slice(i, i + 64)
            x = torch.from_numpy(b["X"][sl]); mk = torch.from_numpy(b["mask"][sl])
            with torch.no_grad():
                dl = build_delta(mk, nfeat) if bb == "brits" else None
                d_raw, _ = m(x, mk, dl)
                d_raw = d_raw[:, CTX:CTX + GAP]
                if resid:
                    d_raw = apply_residual(d_raw, torch.from_numpy(b["interp"][sl]))
                if head_fn is None:
                    P = d_raw[..., :6] * ys_s + ys_m
                else:
                    F_ = (d_raw * ys_s + ys_m if n_disp == 6 else
                          torch.cat([d_raw[..., :6] * ys_s + ys_m, d_raw[..., 6:]], -1))
                    P = head_fn(F_, pL[sl], pR[sl], nd[sl])
            o.append(P.numpy())
        preds[f"{arm} (trained, {n_params(m):,} params)"] = np.concatenate(o, 0)

    L = ["# Ablation — is it the network, or the head plus a leaked net demand?", "",
         f"{args.n} random test 3h gaps, seed {args.seed}. Dispatch channels only "
         "(curtailment is untouched by the head and is not comparable here).", "",
         "`interp + head`, `persistence + head` and `zeros + head` contain **zero "
         "learned parameters**. They differ from the trained arms only in what "
         "direction is handed to the projection.", "",
         "## MAE (MW)", "",
         "| arm | " + " | ".join(SHORT[c] for c in TARGETS) + " | **aggregate** |",
         "| --- |" + " ---: |" * (len(TARGETS) + 1)]
    agg = {}
    for k, P in preds.items():
        e = np.abs(P - truth).reshape(-1, 6)
        mae = e.mean(0)
        agg[k] = float(e.mean())
        L.append(f"| {k} | " + " | ".join(f"{v:,.1f}" for v in mae) +
                 f" | **{agg[k]:,.1f}** |")
    L += ["", "## Balance residual (max |SIGN·P − nd|, MW)", "",
          "| arm | max balance error |", "| --- | ---: |"]
    for k, P in preds.items():
        L.append(f"| {k} | {float(np.abs((P * SIGN).sum(-1) - b['nd']).max()):,.2f} |")

    base = agg.get("interp + head")
    L += ["", "## What this says", ""]
    for k, v in agg.items():
        if k in ("interp (no nd)", "interp + head"):
            continue
        d = 100 * (v - base) / base
        verdict = "**worse than a parameter-free projection**" if d > 0 else "better"
        L.append(f"- `{k}`: {v:,.1f} MW, {d:+.1f}% vs `interp + head` — {verdict}")
    L.append("")

    (OUT / f"ablation_head_vs_network{args.tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
