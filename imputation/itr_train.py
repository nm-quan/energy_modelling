"""Train one arm of the transformer-imputer ablation.

Stage 1 (training-method, constraint = none):
    python3 imputation/itr_train.py --arm T1        # .. T2 T3 T4

Stage 2 (constraint-method, training = T*, the stage-1 winner):
    python3 imputation/itr_train.py --arm T* --constraint soft --lam 0.1   # x {0.1,1,10}
    python3 imputation/itr_train.py --arm T* --constraint rayen

Loss = masked gap MAE (y-scaled) [+ 0.1 x aux recon of OBSERVED dispatch, per arm]
[+ lam x soft feasibility penalty (C1)]. The rayen arm projects the gap trajectory
through the differentiable RAYEN ray-shoot IN-GRAPH each step, so the model trains
inside the constraints and is validated/evaluated through the same map.

Early stopping: patience on VALIDATION GAP-MAE (512 fixed windows drawn with the
arm's own gap distribution, scored through the arm's deployed map). The stage-1
winner T* is then picked on VALIDATION BLACKOUT-MAE (256 fixed common blackout
windows) -- both numbers land in the run json for itr_bench.py.

No seeds are swept: per the plan, margins under ~5%% relative MAE between arms are
treated as ties downstream, not wins.
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
from gap_data import load_flats                                               # noqa: E402
import itr_data as D                                                          # noqa: E402
from itr_model import ITransformerImputer, gap_slice, soft_penalty            # noqa: E402
from constraint_layers import rayen_traj_project                              # noqa: E402

OUT = HERE / "results" / "itr"


def to_dev(b, device):
    out = {}
    for k, v in b.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(v))
            if t.dtype == torch.float64:                 # MPS has no float64
                t = t.float()
            out[k] = t.to(device)
        else:
            out[k] = v
    return out


def deployed_gap_mw(model, b, f, device, constraint):
    """(B,glen,6) MW gap trajectory through the arm's DEPLOYED map: the rayen
    ray-shoot for the rayen arm, the raw fill otherwise (C0/C1 deploy raw; the
    posthoc projection is the separate C2 row, applied in itr_bench)."""
    tb = to_dev(b, device)
    out = model(tb["X"], tb["M"])                                # (B,T,6) y-scaled
    ys = torch.tensor(f.y_scale, dtype=out.dtype, device=device)
    ym = torch.tensor(f.y_mean, dtype=out.dtype, device=device)
    G = gap_slice(out, b["g0"], b["glen"]) * ys + ym             # MW
    if constraint == "rayen":
        G = rayen_traj_project(G, tb["pL"].to(G.dtype), tb["pR"].to(G.dtype),
                               tb["nd"].to(G.dtype))
    return G


def val_gap_mae(model, f, split, rec, device, constraint, batch=256):
    """Mean |pred - truth| MW over gap cells, through the deployed map."""
    model.eval()
    tot = n = 0.0
    with torch.no_grad():
        for glen, idx in D.glen_groups(rec, batch):
            b = D.build_batch(f, split, rec, idx)
            G = deployed_gap_mw(model, b, f, device, constraint)
            ys = torch.tensor(f.y_scale, dtype=torch.float32, device=device)
            ym = torch.tensor(f.y_mean, dtype=torch.float32, device=device)
            truth = gap_slice(torch.from_numpy(b["Y"]).to(device), b["g0"], glen) * ys + ym
            tot += float((G - truth).abs().sum()); n += G.numel()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(D.ARMS), required=True)
    ap.add_argument("--constraint", choices=["none", "soft", "rayen"], default="none")
    ap.add_argument("--lam", type=float, default=1.0, help="soft-penalty weight (C1)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    arm = D.ARMS[args.arm]
    n_val_es, n_val_blk = 512, 256
    if args.smoke:
        args.epochs, args.n_train, args.batch, args.patience = 2, 512, 64, 2
        n_val_es, n_val_blk = 96, 32
    stem = f"itr_{args.arm}" + {"none": "", "soft": f"_soft{args.lam:g}",
                                "rayen": "_rayen"}[args.constraint]
    if args.smoke:
        stem += "_smoke"
    out_path = Path(args.out) if args.out else OUT / f"{stem}.pt"

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"arm={args.arm} ({arm}) constraint={args.constraint}"
          f"{f' lam={args.lam:g}' if args.constraint == 'soft' else ''} device={device} "
          f"epochs={args.epochs} n_train={args.n_train}")

    f = load_flats()
    tr = D.sample_windows(f, "train", args.n_train, mode=arm["gaps"], seed=args.seed)
    va_es = D.sample_windows(f, "val", n_val_es, mode=arm["gaps"], seed=4242)
    va_blk = D.sample_windows(f, "val", n_val_blk, mode="blackout", seed=777)
    u, c = np.unique(tr["glen"], return_counts=True)
    print(f"train {len(tr['s'])} windows, glen {dict(zip(u.tolist(), c.tolist()))} | "
          f"val_es {len(va_es['s'])} ({arm['gaps']}) | val_blk {len(va_blk['s'])} (blackout)")

    model = ITransformerImputer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ys = torch.tensor(f.y_scale, dtype=torch.float32, device=device)
    ym = torch.tensor(f.y_mean, dtype=torch.float32, device=device)
    rng = np.random.default_rng(args.seed)

    best = float("inf"); best_state = None; waited = 0; hist = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train(); tot = nb = 0.0
        for glen, idx in D.glen_groups(tr, args.batch, rng=rng):
            b = D.build_batch(f, "train", tr, idx, aug=arm["flank_aug"], rng=rng)
            tb = to_dev(b, device)
            out = model(tb["X"], tb["M"])                        # (B,T,6) y-scaled
            gapm = (tb["M"] == 0).float()                        # (B,T,1)
            loss = ((out - tb["Y"]).abs() * gapm).sum() / (gapm.sum() * 6 + 1e-6)
            obs = 1.0 - gapm
            if arm["aux"] and float(obs.sum()) > 0:              # SAITS aux recon, weight 0.1
                loss = loss + 0.1 * ((out - tb["Y"]).abs() * obs).sum() / (obs.sum() * 6 + 1e-6)
            if args.constraint in ("soft", "rayen"):
                G = gap_slice(out, b["g0"], glen) * ys + ym      # MW gap trajectory
                pL = tb["pL"].float(); pR = tb["pR"].float(); nd = tb["nd"].float()
                if args.constraint == "soft":
                    loss = loss + args.lam * soft_penalty(G, nd, pL, pR)
                else:                                            # rayen: train INSIDE the map
                    Gp = rayen_traj_project(G, pL, pR, nd)
                    truth = gap_slice(tb["Y"], b["g0"], glen) * ys + ym
                    loss = loss + ((Gp - truth).abs() / ys.mean()).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx); nb += len(idx)
        vw = val_gap_mae(model, f, "val", va_es, device, args.constraint)
        hist.append(vw)
        stop = False
        if vw < best - 1e-4:
            best, waited = vw, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1; stop = waited >= args.patience
        print(f"  ep{ep:03d} loss={tot/max(nb,1):.4f} val_gap_MAE={vw:.1f} MW "
              f"(best {best:.1f}, waited {waited}/{args.patience}) ({time.time()-t0:.0f}s)"
              f"{' *' if waited == 0 else ''}{'  EARLY STOP' if stop else ''}", flush=True)
        if stop:
            break

    if best_state:
        model.load_state_dict(best_state)
    blk = val_gap_mae(model, f, "val", va_blk, device, args.constraint)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    meta = {"arm": args.arm, "constraint": args.constraint,
            "lam": args.lam if args.constraint == "soft" else None,
            "best_val_gap_mae_mw": best, "val_blackout_mae_mw": blk,
            "epochs_ran": len(hist), "n_train": args.n_train, "seed": args.seed,
            "smoke": args.smoke}
    Path(str(out_path).replace(".pt", ".json")).write_text(json.dumps(meta, indent=2))
    print(f"\nBEST val_gap_MAE={best:.1f} MW | val_BLACKOUT_MAE={blk:.1f} MW "
          f"(the T* selection metric)\nwrote {out_path}")


if __name__ == "__main__":
    main()
