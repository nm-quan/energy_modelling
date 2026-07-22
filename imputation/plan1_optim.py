"""Plan1 'optimization' arm (plan1.md S2, run LAST) — QP projection trained through.

Replaces the RAYEN ray-shoot with the ACTUAL projection solved as a differentiable
QP over the whole 36-step gap:

    minimize   || P - fill ||^2                         (P : 36 x 6 MW)
    s.t.       SIGN . P_t = nd_t                 (balance, per step)
               -rdn <= P_t - P_{t-1} <= rup      (ramp, incl. seams to pL, pR)
               0 <= P <= cap                     (box)

Because ramp couples adjacent steps, the QP plans across the horizon (pre-ramps
slow units) -- the coupling RAYEN's single global alpha cannot express. Made
differentiable with cvxpylayers (one solve per forward; the layer is FIXED across
the batch, only fill/nd/pL/pR/cap vary, so it compiles once). SOC is left to the
posthoc map (3h gaps are SOC non-binding -- measured 59% of pack worst-case).

Requires cvxpylayers (Colab: `pip install cvxpylayers`). Same recipe/flags as
plan1_train.py; writes optimization.pt + optimization_history.json.

    pip install cvxpylayers
    python3 imputation/plan1_optim.py
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
sys.path.insert(0, str(ROOT / "ml"))          # check_caps lives in ml/ (RAMPS/CAPS canon)
import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "prepared.npz"
from gap_data import load_flats, sample_train_windows, TARGETS, SIGN, ND_COL   # noqa: E402
from model import BiLSTMImputer                                        # noqa: E402
import check_caps as CC                                                # noqa: E402

OUT = HERE / "results" / "plan1"
GAP = 36
R_UP = np.array([CC.RAMPS[t][1] for t in TARGETS], np.float64)
R_DN = np.array([abs(CC.RAMPS[t][0]) for t in TARGETS], np.float64)
CAP = np.array([CC.CAPS[t] for t in TARGETS], np.float64)


def build_qp_layer(device):
    """CvxpyLayer for the gap QP. Params: fill (36,6), nd (36,), pL (6,), pR (6,),
    cap (6,). Var: P (36,6). Returns the layer (solved with a diff-through cone
    solver). Ramp/box use the canon rates as fixed constants."""
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    P = cp.Variable((GAP, 6))
    fill = cp.Parameter((GAP, 6))
    nd = cp.Parameter(GAP)
    pL = cp.Parameter(6); pR = cp.Parameter(6); cap = cp.Parameter(6, nonneg=True)
    sign = np.array(SIGN)
    cons = [P >= 0, P <= cap[None, :]]
    cons += [P @ sign == nd]                              # balance per step
    full_prev = cp.vstack([cp.reshape(pL, (1, 6)), P])    # (37,6): [pL; P]
    full_next = cp.vstack([P, cp.reshape(pR, (1, 6))])    # (37,6): [P; pR]
    dP = full_next - full_prev                            # (37,6) steps incl. both seams
    cons += [dP <= R_UP[None, :], dP >= -R_DN[None, :]]
    obj = cp.Minimize(cp.sum_squares(P - fill))
    prob = cp.Problem(obj, cons)
    return CvxpyLayer(prob, parameters=[fill, nd, pL, pR, cap], variables=[P])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)     # QP solve is the bottleneck
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--context", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_train, args.n_val, args.batch = 2, 800, 200, 32
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tag = "optimization" + ("_smoke" if args.smoke else "")
    print(f"plan1 optimization (QP) device={device} epochs={args.epochs} "
          f"n_train={args.n_train} batch={args.batch}", flush=True)

    f = load_flats()
    n_feat = len(f.feat_cols)
    assert n_feat == 21, f"expected 21-feature ren flats, got {n_feat}"

    def supply_nd(w):
        w["nd_mw"] = ((w["Y"] * f.y_scale + f.y_mean) @ SIGN).astype(np.float32)
        return w
    tr = supply_nd(sample_train_windows(f, args.n_train, context=args.context,
                                        seed=args.seed, split="train"))
    va = supply_nd(sample_train_windows(f, args.n_val, context=args.context,
                                        seed=args.seed + 777, split="val"))
    print(f"windows: train {len(tr['X']):,} val {len(va['X']):,}", flush=True)

    ys_mean = torch.tensor(f.y_mean, dtype=torch.float32, device=device)
    ys_scale = torch.tensor(f.y_scale, dtype=torch.float32, device=device)
    cap_t = torch.tensor(CAP, dtype=torch.float32, device=device)
    qp = build_qp_layer(device)
    gs, ge = args.context, args.context + GAP

    def mapped(b):
        xb = torch.from_numpy(b["X"]).to(device); mb = torch.from_numpy(b["mask"]).to(device)
        interp = torch.from_numpy(b["interp"]).to(device)
        dev = model(xb, mb)[:, gs:ge]
        fill = (interp + dev) * ys_scale + ys_mean               # (B,G,6) MW
        nd = torch.from_numpy(b["nd_mw"]).to(device)
        pL = torch.from_numpy(b["pL_mw"]).to(device); pR = torch.from_numpy(b["pR_mw"]).to(device)
        capb = cap_t.expand(fill.shape[0], 6)
        (P,) = qp(fill, nd, pL, pR, capb)                        # differentiable QP solve
        return (P - ys_mean) / ys_scale

    model = BiLSTMImputer(n_features=n_feat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)

    def val_mse():
        model.eval(); tot = n = 0.0
        with torch.no_grad():
            for i in range(0, len(va["X"]), args.batch):
                b = {k: (v[i:i + args.batch] if isinstance(v, np.ndarray) else v)
                     for k, v in va.items()}
                pred = mapped(b); yb = torch.from_numpy(b["Y"]).to(device)
                tot += float(((pred - yb) ** 2).sum()); n += yb.numel()
        return tot / n

    hist = {"train_mse": [], "val_mse": []}
    best, best_state, waited = float("inf"), None, 0
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        te0 = time.time(); model.train()
        perm = rng.permutation(len(tr["X"])); s_mse = 0.0; nw = 0
        for i in range(0, len(perm), args.batch):
            j = perm[i:i + args.batch]
            b = {k: (v[j] if isinstance(v, np.ndarray) else v) for k, v in tr.items()}
            pred = mapped(b); yb = torch.from_numpy(b["Y"]).to(device)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            s_mse += float(loss.detach()) * len(j); nw += len(j)
        vm = val_mse()
        hist["train_mse"].append(s_mse / nw); hist["val_mse"].append(vm)
        stop = False
        if vm < best - 1e-6:
            best, waited = vm, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1; stop = waited >= args.patience
        (OUT / f"{tag}_history.json").write_text(json.dumps(
            {"arm": "optimization", "epochs_run": ep, "best_val_mse": best, **hist}, indent=1))
        print(f"  ep{ep:03d} train_mse={hist['train_mse'][-1]:.4f} val_mse={vm:.4f} "
              f"(best {best:.4f}, waited {waited}/{args.patience}) "
              f"({time.time()-te0:.0f}s){'  EARLY STOP' if stop else ''}", flush=True)
        if stop:
            break

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT / f"{tag}.pt")
    print(f"\nbest val_mse={best:.4f}  wrote {OUT / f'{tag}.pt'}")


if __name__ == "__main__":
    main()
