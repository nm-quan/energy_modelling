"""Plan1 training (plan1.md): BiLSTM+RAYEN 3h-gap imputers on the 21-feature
renewables dataset (net_dispatch_ren). Three arms share one recipe:

  baseline    rayen_traj trained through (equal balance split)
  cost        baseline + 0.1 * dispatch-cost term (data/fuel_costs.json prices,
              battery_charging signed NEGATIVE, normalizer 1e5)
  size_aware  rayen_traj with the level-proportional balance split
              (constraints._balance_project size_aware=True) in-graph and at eval

Recipe: epochs 30, patience 10, batch 128, AdamW lr 1e-3 wd 1e-5, seed 0,
n_train 40k random-position 3h-gap windows. TRAIN and VAL loss are MSE (val =
masked-gap MSE through the arm's own map; early stop on it). Per-epoch history
is persisted to <out>/<arm>_history.json (never only notebook prints).

nd bookkeeping (plan1.md S1): the MODEL sees the demand-side nd feature
(col 6 of the new flats). The MAP balances to the supply-side sum
Sigma SIGN*truth -- windows' nd_mw is overridden accordingly here.

    python3 imputation/plan1_train.py --arm baseline
    python3 imputation/plan1_train.py --arm cost --smoke     # CPU sanity
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
from gap_data import load_flats, sample_train_windows, TARGETS, SIGN, ND_COL   # noqa: E402
from model import BiLSTMImputer, masked_loss                           # noqa: E402
from constraint_layers import rayen_traj_project                       # noqa: E402

OUT = HERE / "results" / "plan1"
COSTS = json.loads((ROOT / "data" / "fuel_costs.json").read_text())["dispatch"]
# TARGETS order, battery_charging signed negative (fuel_costs.json convention)
C_SIGNED = np.array([COSTS[t] * (-1.0 if t == "battery_charging" else 1.0)
                     for t in TARGETS], dtype=np.float32)
COST_NORM = 1e5                       # $100/MWh x 1000 MW reference -> O(1) term


def supply_nd(win, f):
    """Override the windows' balance target with the supply-side sum (MW):
    nd_bal = SIGN . truth -- exact for actuals, unlike the demand-side feature."""
    win["nd_mw"] = ((win["Y"] * f.y_scale + f.y_mean) @ SIGN).astype(np.float32)
    return win


def forward_mapped(model, batch, f, device, size_aware, ys_mean, ys_scale):
    """Model fill -> in-graph RAYEN map -> y-scaled mapped gap (B,G,6)."""
    gs, ge = batch["context"], batch["context"] + batch["gap"]
    xb = torch.from_numpy(batch["X"]).to(device)
    mb = torch.from_numpy(batch["mask"]).to(device)
    interp = torch.from_numpy(batch["interp"]).to(device)
    dev = model(xb, mb)[:, gs:ge]
    fill_mw = (interp + dev) * ys_scale + ys_mean
    P_mw = rayen_traj_project(fill_mw,
                              torch.from_numpy(batch["pL_mw"]).to(device),
                              torch.from_numpy(batch["pR_mw"]).to(device),
                              torch.from_numpy(batch["nd_mw"]).to(device),
                              size_aware=size_aware)
    return (P_mw - ys_mean) / ys_scale, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "cost", "size_aware"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-bal", type=float, default=0.1)
    ap.add_argument("--lam-cost", type=float, default=0.1)
    ap.add_argument("--context", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_train, args.n_val, args.batch = 2, 1500, 300, 64
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    size_aware = args.arm == "size_aware"
    use_cost = args.arm == "cost"
    tag = args.arm + ("_smoke" if args.smoke else "")
    print(f"plan1 arm={args.arm} device={device} epochs={args.epochs} "
          f"n_train={args.n_train} size_aware={size_aware} lam_cost="
          f"{args.lam_cost if use_cost else 0}", flush=True)

    f = load_flats()
    n_feat = len(f.feat_cols)
    assert n_feat == 21, f"expected 21-feature ren flats, got {n_feat} -- run plan1_data.py"
    print(f"flats: Xtr={f.Xtr.shape} Xva={f.Xva.shape} Xte={f.Xte.shape}", flush=True)

    t0 = time.time()
    tr = supply_nd(sample_train_windows(f, args.n_train, context=args.context,
                                        seed=args.seed, split="train"), f)
    va = supply_nd(sample_train_windows(f, args.n_val, context=args.context,
                                        seed=args.seed + 777, split="val"), f)
    print(f"windows: train {len(tr['X']):,} val {len(va['X']):,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    ys_mean = torch.tensor(f.y_mean, dtype=torch.float32, device=device)
    ys_scale = torch.tensor(f.y_scale, dtype=torch.float32, device=device)
    sign_t = torch.tensor(SIGN, dtype=torch.float32, device=device)
    cost_t = torch.tensor(C_SIGNED, device=device)
    nd_mean, nd_scale = float(f.x_mean[ND_COL]), float(f.x_scale[ND_COL])
    gs, ge = args.context, args.context + 36

    model = BiLSTMImputer(n_features=n_feat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)

    def val_mse():
        model.eval(); tot = n = 0.0
        with torch.no_grad():
            for i in range(0, len(va["X"]), 256):
                b = {k: (v[i:i + 256] if isinstance(v, np.ndarray) else v)
                     for k, v in va.items()}
                pred, _ = forward_mapped(model, b, f, device, size_aware, ys_mean, ys_scale)
                yb = torch.from_numpy(b["Y"]).to(device)
                tot += float(((pred - yb) ** 2).sum()); n += yb.numel()
        return tot / n

    hist = {"train_mse": [], "train_loss": [], "val_mse": [], "cost_term": []}
    best, best_state, waited = float("inf"), None, 0
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        te0 = time.time(); model.train()
        perm = rng.permutation(len(tr["X"]))
        s_loss = s_mse = s_cost = 0.0; n_win = 0
        for i in range(0, len(perm), args.batch):
            j = perm[i:i + args.batch]
            b = {k: (v[j] if isinstance(v, np.ndarray) else v) for k, v in tr.items()}
            pred, dev = forward_mapped(model, b, f, device, size_aware, ys_mean, ys_scale)
            yb = torch.from_numpy(b["Y"]).to(device)
            # soft balance targets the SUPPLY-side nd: pre-scale it with the nd
            # feature's scaler so masked_loss's de-standardization recovers it.
            nd_sub = torch.from_numpy((b["nd_mw"] - nd_mean) / nd_scale).to(device)
            loss, rec, _ = masked_loss(pred, yb, nd_sub, nd_mean, nd_scale,
                                       ys_mean, ys_scale, sign_t, args.lam_bal, loss="mse")
            if use_cost:
                p_mw = pred * ys_scale + ys_mean
                cterm = (p_mw * cost_t).sum(-1).mean() / COST_NORM
                loss = loss + args.lam_cost * cterm
                s_cost += float(cterm.detach()) * len(j)
            opt.zero_grad(); loss.backward(); opt.step()
            s_loss += float(loss.detach()) * len(j)
            s_mse += float(rec) * len(j); n_win += len(j)
        vm = val_mse()
        hist["train_loss"].append(s_loss / n_win)
        hist["train_mse"].append(s_mse / n_win)
        hist["val_mse"].append(vm)
        hist["cost_term"].append(s_cost / n_win if use_cost else None)
        stop = False
        if vm < best - 1e-6:
            best, waited = vm, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1; stop = waited >= args.patience
        (OUT / f"{tag}_history.json").write_text(json.dumps(
            {"arm": args.arm, "epochs_run": ep, "best_val_mse": best, **hist}, indent=1))
        print(f"  ep{ep:03d} train_mse={hist['train_mse'][-1]:.4f} "
              f"loss={hist['train_loss'][-1]:.4f} val_mse={vm:.4f} "
              f"(best {best:.4f}, waited {waited}/{args.patience}) "
              f"({time.time()-te0:.0f}s){'  EARLY STOP' if stop else ''}", flush=True)
        if stop:
            break

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT / f"{tag}.pt")
    print(f"\nbest val_mse={best:.4f}  wrote {OUT / f'{tag}.pt'} + {tag}_history.json")


if __name__ == "__main__":
    main()
