"""Train the BiLSTM through the RECURSIVE RAYEN head (imputation/recursive_head.py).

Same data, recipe and loss as plan1_train.py so the arms are comparable:
40k random-position 3h-gap windows, AdamW lr 1e-3 wd 1e-5, batch 128, seed 0,
MSE on the MAPPED output, supply-side balance target.

Two differences, both forced by the head:
  * the network emits 7 channels per step (6 direction + 1 step logit), not 6;
  * there is no interp-skeleton residual parameterisation. The head builds its
    own per-step anchor from (P_{t-1}, pR, nd), which already encodes the ramp
    tube, so the network supplies direction and step only.

    python3 imputation/plan1_train_rec.py --epochs 1
    python3 imputation/plan1_train_rec.py --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "prepared.npz"
from gap_data import load_flats, sample_train_windows, TARGETS, SIGN   # noqa: E402
from model import BiLSTMImputer                                        # noqa: E402
from recursive_head import recursive_rayen                             # noqa: E402
import constraints as C                                                # noqa: E402

OUT = HERE / "results" / "plan1"
TABLE = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "table.parquet"


def set_coal_ramp_percentile(pct: float):
    """Replace coal_brown's ramp limits with the pct-th percentile of the historical
    5-min step change. The canon value is the empirical MAX, and check_caps.py flags
    the down-ramp itself as a single unit trip (2024-02-13, -1553.6 vs a p99 of
    -123.3). Max is the right envelope for feasibility validation; a percentile is
    the right one for counterfactual simulation. Patches BOTH the numpy arrays (used
    by the violation audits) and the cached f32 tensors (used by the head)."""
    import pandas as pd
    import torch as _t
    d = pd.read_parquet(TABLE)["coal_brown"].diff().dropna().values
    i = TARGETS.index("coal_brown")
    up, dn = float(np.percentile(d, pct)), float(np.percentile(-d, pct))
    old = (float(C.R_UP[i]), float(C.R_DN[i]))
    C.R_UP[i], C.R_DN[i] = up, dn
    C._RUP_T = _t.tensor(C.R_UP, dtype=_t.float32)
    C._RDN_T = _t.tensor(C.R_DN, dtype=_t.float32)
    print(f"coal_brown ramp: canon {old[0]:.1f}/{old[1]:.1f} -> p{pct:g} {up:.1f}/{dn:.1f} MW/5min")
    return up, dn


def supply_nd(win, f):
    """Balance target = the supply-side sum (exact for actuals), as plan1_train."""
    win["nd_mw"] = ((win["Y"] * f.y_scale + f.y_mean) @ SIGN).astype(np.float32)
    return win


def forward_mapped(model, batch, device, ys_mean, ys_scale, return_debug=False):
    """Model raw (B,W,7) -> gap slice -> recursive head -> y-scaled (B,G,6)."""
    gs, ge = batch["context"], batch["context"] + batch["gap"]
    xb = torch.from_numpy(batch["X"]).to(device)
    mb = torch.from_numpy(batch["mask"]).to(device)
    raw = model(xb, mb)[:, gs:ge]                                      # (B,G,7)
    out = recursive_rayen(raw,
                          torch.from_numpy(batch["pL_mw"]).to(device),
                          torch.from_numpy(batch["pR_mw"]).to(device),
                          torch.from_numpy(batch["nd_mw"]).to(device),
                          return_debug=return_debug)
    P_mw, dbg = (out if return_debug else (out, None))
    return (P_mw - ys_mean) / ys_scale, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--context", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="recursive")
    ap.add_argument("--coal-pct", type=float, default=None,
                    help="use the pct-th percentile of coal's historical 5-min step "
                         "as its ramp limit instead of the empirical max")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_train, args.n_val, args.batch = 1, 1024, 256, 64
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tag = args.tag + ("_smoke" if args.smoke else "")
    if args.coal_pct is not None:
        set_coal_ramp_percentile(args.coal_pct)
        tag += f"_coalp{args.coal_pct:g}"
    print(f"recursive-head arm  device={device} epochs={args.epochs} "
          f"n_train={args.n_train} batch={args.batch}", flush=True)

    f = load_flats()
    n_feat = len(f.feat_cols)
    t0 = time.time()
    tr = supply_nd(sample_train_windows(f, args.n_train, context=args.context,
                                        seed=args.seed, split="train"), f)
    va = supply_nd(sample_train_windows(f, args.n_val, context=args.context,
                                        seed=args.seed + 777, split="val"), f)
    print(f"windows: train {len(tr['X']):,} val {len(va['X']):,} ({time.time()-t0:.0f}s)",
          flush=True)

    ys_mean = torch.tensor(f.y_mean, dtype=torch.float32, device=device)
    ys_scale = torch.tensor(f.y_scale, dtype=torch.float32, device=device)

    model = BiLSTMImputer(n_features=n_feat, n_targets=7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)

    def val_mse():
        model.eval(); tot = n = 0.0
        with torch.no_grad():
            for i in range(0, len(va["X"]), 256):
                b = {k: (v[i:i + 256] if isinstance(v, np.ndarray) else v)
                     for k, v in va.items()}
                pred, _ = forward_mapped(model, b, device, ys_mean, ys_scale)
                yb = torch.from_numpy(b["Y"]).to(device)
                tot += float(((pred - yb) ** 2).sum()); n += yb.numel()
        return tot / n

    hist = {"train_mse": [], "val_mse": [], "mean_step": [], "mean_alpha": []}
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  val_mse before training = {val_mse():.4f}", flush=True)
    for ep in range(1, args.epochs + 1):
        te0 = time.time(); model.train()
        perm = rng.permutation(len(tr["X"]))
        s_mse = 0.0; s_step = 0.0; s_alpha = 0.0; n_win = 0
        for i in range(0, len(perm), args.batch):
            j = perm[i:i + args.batch]
            b = {k: (v[j] if isinstance(v, np.ndarray) else v) for k, v in tr.items()}
            pred, dbg = forward_mapped(model, b, device, ys_mean, ys_scale,
                                       return_debug=True)
            yb = torch.from_numpy(b["Y"]).to(device)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            s_mse += float(loss.detach()) * len(j)
            s_step += float(dbg["step"].detach().mean()) * len(j)
            s_alpha += float(dbg["alpha_max"].detach().mean()) * len(j)
            n_win += len(j)
            if (i // args.batch) % 50 == 0:
                print(f"    batch {i//args.batch:4d}/{len(perm)//args.batch}  "
                      f"mse {float(loss.detach()):.4f}  "
                      f"step {float(dbg['step'].detach().mean()):.3f}  "
                      f"alpha {float(dbg['alpha_max'].detach().mean()):.1f} MW",
                      flush=True)
        vm = val_mse()
        hist["train_mse"].append(s_mse / n_win); hist["val_mse"].append(vm)
        hist["mean_step"].append(s_step / n_win); hist["mean_alpha"].append(s_alpha / n_win)
        (OUT / f"{tag}_history.json").write_text(json.dumps(
            {"arm": tag, "epochs_run": ep, **hist}, indent=1))
        print(f"  ep{ep:03d} train_mse={hist['train_mse'][-1]:.4f} val_mse={vm:.4f} "
              f"mean_step={hist['mean_step'][-1]:.3f} "
              f"mean_alpha={hist['mean_alpha'][-1]:.1f} MW ({time.time()-te0:.0f}s)",
              flush=True)

    torch.save(model.state_dict(), OUT / f"{tag}.pt")
    print(f"\nwrote {OUT / f'{tag}.pt'} + {tag}_history.json")


if __name__ == "__main__":
    main()
