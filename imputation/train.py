"""Train the bi-LSTM gap imputer on 5 years of masked windows, evaluate vs the
interpolation baseline on the real test 11:00-14:00 windows.

Training: random-position masked gaps over the 394k-row train flat, with optional
DEMAND PERTURBATION (professor's synthetic-data idea) so the model can't score by
boundary-blending and must read net_demand/demand — the counterfactual-robustness
step. Loss = masked reconstruction + soft balance.

Eval (deployment-matched): mask the real 11-14 window, impute, PROJECT to hard
feasibility (constraints.project_gap), score reconstruction WAPE vs measured truth,
and check violations incl. both seams.

    python3 imputation/train.py --epochs 15 --n-train 40000 --perturb 0.5
    python3 imputation/train.py --smoke        # tiny, CPU, sanity
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
sys.path.insert(0, str(HERE))
from gap_data import (load_flats, test_gap_windows, sample_train_windows,      # noqa: E402
                      TARGETS, SIGN, TARGET_FEAT_IDX, ND_COL, DEM_COL)
import constraints as C                                                        # noqa: E402
from model import BiLSTMImputer, masked_loss                                   # noqa: E402
from baseline import linear_fill                                               # noqa: E402

OUT = HERE / "results"


def perturb_demand(X, mask, f, frac: float, rng, max_g: float = 30.0):
    """On a `frac` share of windows, scale demand+net_demand INSIDE the gap by a
    random +g% (like the deployment counterfactual). Known-subspace cols only, so
    the target (real breakdown) is unchanged -- teaching the model that a raised
    total should change the fill, not be ignored. Returns X (copy modified)."""
    X = X.copy()
    B, W, _ = X.shape
    gmask = (mask[:, :, 0] == 0)                              # (B,W) gap steps
    pick = rng.random(B) < frac
    g = rng.uniform(0, max_g, B) / 100.0
    for col in (ND_COL, DEM_COL):
        mw = X[:, :, col] * f.x_scale[col] + f.x_mean[col]
        mw = np.where(pick[:, None] & gmask, mw * (1 + g[:, None]), mw)
        X[:, :, col] = (mw - f.x_mean[col]) / f.x_scale[col]
    return X


def evaluate(model, f, gws, device, context):
    """Reconstruction WAPE (projected) on the real test gaps + violation counts."""
    N = len(gws[0].gap_idx)
    tfi = np.asarray(TARGET_FEAT_IDX)
    num = np.zeros(6); den = np.zeros(6)
    ramp_bad = neg = 0; bal_max = 0.0
    model.eval()
    with torch.no_grad():
        for gw in gws:
            g0 = gw.gap_idx[0]
            xs = f.Xte[g0 - context: g0 + N + context].copy()   # (W,17)
            m = np.ones((xs.shape[0], 1), np.float32); m[context:context + N] = 0.0
            xs[context:context + N][:, tfi] = 0.0
            xb = torch.from_numpy(xs[None].astype(np.float32)).to(device)
            mb = torch.from_numpy(m[None]).to(device)
            dev = model(xb, mb)[0, context:context + N].cpu().numpy()   # (N,6) y-scaled deviation
            # residual: fill = linear-interp skeleton + learned deviation (y-scaled)
            tt = (np.arange(1, N + 1) / (N + 1))[:, None]
            pL_s, pR_s = f.Yte[g0 - 1], f.Yte[gw.gap_idx[-1] + 1]
            interp = pL_s[None, :] + tt * (pR_s - pL_s)[None, :]
            fill = (interp + dev) * f.y_scale + f.y_mean
            P, resid = C.project_gap(fill, gw.pL_mw, gw.pR_mw, gw.nd_mw)
            num += np.abs(P - gw.truth_mw).sum(0); den += np.abs(gw.truth_mw).sum(0)
            d = np.diff(np.vstack([gw.pL_mw, P, gw.pR_mw]), axis=0)
            ramp_bad += int(((d > C.R_UP + 0.6) | (d < -(C.R_DN + 0.6))).sum())
            neg += int((P < -0.1).sum()); bal_max = max(bal_max, float(resid.max()))
    per = num / np.clip(den, 1e-6, None)
    return {"macro_WAPE": float(per.mean()),
            "per_channel_WAPE": {t: float(per[i]) for i, t in enumerate(TARGETS)},
            "ramp_violations": ramp_bad, "n_neg": neg, "balance_resid_max_mw": bal_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10,
                    help="early stop after this many epochs with no val-recon improvement")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--context", type=int, default=72)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="share of windows with SMALL input-noise augmentation (default off; "
                         "counterfactual responsiveness comes from natural 5-yr demand variation, "
                         "NOT from perturbing demand with an unchanged target -- that teaches the "
                         "model to IGNORE demand)")
    ap.add_argument("--lam-bal", type=float, default=0.1)
    ap.add_argument("--lam-dev", type=float, default=0.0,
                    help="L2 penalty on the deviation from interp (keeps smooth channels "
                         "at the interp baseline; sweep on GPU, e.g. 0.01-0.3)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT / "bilstm_imputer.pt"))
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_train, args.batch = 2, 2000, 64
        if args.out == str(OUT / "bilstm_imputer.pt"):   # never clobber the real ckpt
            args.out = str(OUT / "bilstm_smoke.pt")
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"device={device} epochs={args.epochs} n_train={args.n_train} perturb={args.perturb}")

    f = load_flats()
    gws = test_gap_windows(f, context=args.context)
    print(f"test gap-days: {len(gws)}")
    rng = np.random.default_rng(args.seed)
    ys_mean = torch.tensor(f.y_mean, dtype=torch.float32, device=device)
    ys_scale = torch.tensor(f.y_scale, dtype=torch.float32, device=device)
    sign = torch.tensor(SIGN, dtype=torch.float32, device=device)
    nd_mean, nd_scale = float(f.x_mean[ND_COL]), float(f.x_scale[ND_COL])

    model = BiLSTMImputer(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    gs, ge = args.context, args.context + 36
    y_mw = (f.y_scale, f.y_mean)

    # Train set sampled ONCE (reused each epoch -- standard, and avoids the per-epoch
    # 3.7 GB resample). Held-out VAL set from the val split (random gaps) drives early
    # stopping, so #epochs is NOT selected on test (no leakage); test is scored once
    # at the end on the val-selected best.
    tr = sample_train_windows(f, args.n_train, context=args.context, seed=args.seed, split="train")
    if args.perturb > 0:
        tr["X"] = perturb_demand(tr["X"], tr["mask"], f, args.perturb, rng)
    va = sample_train_windows(f, 4000, context=args.context, seed=args.seed + 777, split="val")
    va_fill_truth = (va["interp"], va["Y"])                  # for raw val recon (interp+dev vs Y)

    def val_recon_wape():
        model.eval()
        with torch.no_grad():
            num = np.zeros(6); den = np.zeros(6)
            for i in range(0, len(va["X"]), 512):
                dev = model(torch.from_numpy(va["X"][i:i + 512]).to(device),
                            torch.from_numpy(va["mask"][i:i + 512]).to(device))[:, gs:ge].cpu().numpy()
                fill = (va["interp"][i:i + 512] + dev) * y_mw[0] + y_mw[1]
                truth = va["Y"][i:i + 512] * y_mw[0] + y_mw[1]
                num += np.abs(fill - truth).sum((0, 1)); den += np.abs(truth).sum((0, 1))
        return float(np.mean(num / np.clip(den, 1e-6, None)))

    best = float("inf"); best_state = None; waited = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train(); perm = rng.permutation(len(tr["X"])); tot = 0.0
        for i in range(0, len(tr["X"]), args.batch):
            j = perm[i:i + args.batch]
            xb = torch.from_numpy(tr["X"][j]).to(device)
            mb = torch.from_numpy(tr["mask"][j]).to(device)
            yb = torch.from_numpy(tr["Y"][j]).to(device)
            interp = torch.from_numpy(tr["interp"][j]).to(device)
            xnd = torch.from_numpy(tr["X"][j][:, gs:ge, ND_COL]).to(device)
            dev = model(xb, mb)[:, gs:ge]
            out = interp + dev                                # residual: interp + learned deviation
            loss, rec, bal = masked_loss(out, yb, xnd, nd_mean, nd_scale,
                                         ys_mean, ys_scale, sign, args.lam_bal)
            loss = loss + args.lam_dev * (dev ** 2).mean()    # keep dev small: stay near interp
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(j)
        vw = val_recon_wape()
        stop = False
        if vw < best - 1e-5:
            best = vw; waited = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1; stop = waited >= args.patience
        print(f"  ep{ep:03d} loss={tot/len(tr['X']):.4f} val_recon_WAPE={vw:.4f} "
              f"(best {best:.4f}, waited {waited}/{args.patience}) ({time.time()-t0:.0f}s)"
              f"{' *' if waited == 0 else ''}{'  EARLY STOP' if stop else ''}", flush=True)
        if stop:
            break

    if best_state:
        model.load_state_dict(best_state)
    ev = evaluate(model, f, gws, device, args.context)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), args.out)
    ev.update(method="bilstm", epochs=args.epochs, n_train=args.n_train,
              perturb=args.perturb, context=args.context)
    # results json follows the checkpoint name, so --smoke (-> bilstm_smoke.pt)
    # writes bilstm_smoke_recon.json and never clobbers the real bilstm_recon.json
    recon_name = "bilstm_recon.json" if Path(args.out).stem == "bilstm_imputer" \
        else Path(args.out).stem + "_recon.json"
    (OUT / recon_name).write_text(json.dumps(ev, indent=2))
    print(f"\nBEST recon WAPE={ev['macro_WAPE']:.4f}  ramp={ev['ramp_violations']} "
          f"neg={ev['n_neg']} bal_max={ev['balance_resid_max_mw']:.1f} MW")
    for t in TARGETS:
        print(f"    {t:20s} {ev['per_channel_WAPE'][t]:.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
