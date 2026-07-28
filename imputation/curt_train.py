"""Two 3h-gap imputers that ALSO predict wind and solar curtailment.

  --arm plain      BiLSTM -> 8 raw channels (6 dispatch + 2 curtailment).
                   No constraints at all. The unconstrained reference.
  --arm recursive  BiLSTM -> 9 channels (6 direction + 1 step logit + 2
                   curtailment). Dispatch goes through recursive_head; curtailment
                   through relu in z-space (see curt_activation).

WHY CURTAILMENT IS A TARGET AND NOT AN INPUT. Until now curtailment was a feature
the model read and could not change, so a counterfactual could never express
"demand rose, so we spilled less wind". Making it an output lets the model say
exactly that -- and in the recursive arm it is COUPLED to the dispatch, which is
the point:

    nd_effective = nd_target - (curt_actual - curt_pred)

If the model predicts less curtailment than actually happened, that energy is
delivered instead of spilled, so the six fuels have less to cover. At
curt_pred == curt_actual this reduces to nd_target, so training is consistent.
The coupling is one-directional (curtailment does not depend on the dispatch
head), so it is still a single forward pass and fully differentiable.

Both arms are masked identically: the six dispatch columns AND the two curtailment
columns are zeroed inside the gap, so neither model can read the answer.

    python3 imputation/curt_train.py --arm plain --epochs 1
    python3 imputation/curt_train.py --arm recursive --epochs 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "prepared.npz"
from gap_data import load_flats, TARGETS, SIGN, TARGET_FEAT_IDX        # noqa: E402
from model import BiLSTMImputer                                        # noqa: E402
from recursive_head import recursive_rayen                             # noqa: E402

OUT = HERE / "results" / "plan1"
CURT_COLS = [19, 20]


def curt_activation(raw2, c_mean, c_scale):
    """Map the two raw curtailment logits to MW.

    relu, not softplus, and in Z-SPACE, not raw MW. Both parts matter:

      relu      the target is EXACTLY zero in 56% (wind) and 66% (solar) of gap
                cells. softplus is strictly positive and can never produce that
                value; its kink is smoothed away exactly where the target's mass
                sits. relu's dead zone is not a problem here -- when the truth is
                zero and the prediction is zero, zero gradient is correct.
      z-space   softplus(raw) forced the head to reach ~102 in RAW MW from a
                Linear(256,.) initialised at +-0.062, while its other outputs only
                needed to reach ~1. At raw=0 it emitted 0.683 MW against a training
                mean of 102.4. Mapping through the scaler puts it at the training
                mean at initialisation, before any gradient step.
    """
    return torch.relu(raw2 * c_scale + c_mean)


CTX, GAP = 48, 36


def build(f, starts, split="train", context=CTX, gap=GAP):
    """Windows with 8 targets. Mirrors gap_data._build_windows but additionally
    blanks the curtailment inputs in the gap and returns their truth in MW."""
    Xflat, Yflat = ((f.Xtr, f.Ytr) if split == "train" else
                    (f.Xva, f.Yva) if split == "val" else (f.Xte, f.Yte))
    W = 2 * context + gap
    gs, ge = context, context + gap
    tfi = np.asarray(TARGET_FEAT_IDX)
    X = np.stack([Xflat[s:s + W] for s in starts]).astype(np.float32)
    Y = np.stack([Yflat[s + gs:s + ge] for s in starts]).astype(np.float32)   # (n,G,6) z
    cm = f.x_mean[CURT_COLS].astype(np.float32)
    cs = f.x_scale[CURT_COLS].astype(np.float32)
    C = X[:, gs:ge][:, :, CURT_COLS] * cs + cm                                # (n,G,2) MW
    mask = np.ones((len(starts), W, 1), np.float32); mask[:, gs:ge] = 0.0
    X[:, gs:ge, tfi] = 0.0                       # blank the dispatch we must infer
    X[:, gs:ge, CURT_COLS] = 0.0                 # and the curtailment we must infer
    pL = f.y_to_mw(Yflat[starts + gs - 1]).astype(np.float32)
    pR = f.y_to_mw(Yflat[starts + ge]).astype(np.float32)
    nd = ((Y * f.y_scale + f.y_mean) @ SIGN).astype(np.float32)                # supply-side
    return {"X": X, "mask": mask, "Y": Y, "C": C.astype(np.float32),
            "pL": pL, "pR": pR, "nd": nd,
            "c_mean": cm, "c_scale": cs}


def sample(f, n, split, seed):
    Xflat = f.Xtr if split == "train" else f.Xva
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, Xflat.shape[0] - (2 * CTX + GAP), size=n)
    return build(f, starts, split)


def forward(model, b, arm, dev, ys_m, ys_s, c_m, c_s):
    """-> (pred_dispatch_z (B,G,6), pred_curt_mw (B,G,2))"""
    raw = model(torch.from_numpy(b["X"]).to(dev),
                torch.from_numpy(b["mask"]).to(dev))[:, CTX:CTX + GAP]
    curt = curt_activation(raw[..., -2:], c_m, c_s)            # MW, >= 0
    if arm == "plain":
        return raw[..., :6], curt
    # recursive: couple curtailment into the balance target, then shoot
    c_act = torch.from_numpy(b["C"]).to(dev)
    nd = torch.from_numpy(b["nd"]).to(dev) - (c_act - curt).sum(-1)
    P = recursive_rayen(raw[..., :7],
                        torch.from_numpy(b["pL"]).to(dev),
                        torch.from_numpy(b["pR"]).to(dev), nd)
    return (P - ys_m) / ys_s, curt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["plain", "recursive"], required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_val, args.batch = 1024, 256, 64
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"curt_{args.arm}" + ("_smoke" if args.smoke else "")
    n_out = 8 if args.arm == "plain" else 9
    print(f"arm={args.arm} outputs={n_out} device={dev} epochs={args.epochs} "
          f"n_train={args.n_train}", flush=True)

    f = load_flats()
    t0 = time.time()
    tr = sample(f, args.n_train, "train", args.seed)
    va = sample(f, args.n_val, "val", args.seed + 777)
    print(f"windows: train {len(tr['X']):,} val {len(va['X']):,} ({time.time()-t0:.0f}s)",
          flush=True)

    ys_m = torch.tensor(f.y_mean, dtype=torch.float32, device=dev)
    ys_s = torch.tensor(f.y_scale, dtype=torch.float32, device=dev)
    c_m = torch.tensor(tr["c_mean"], device=dev)
    c_s = torch.tensor(tr["c_scale"], device=dev)

    model = BiLSTMImputer(n_features=len(f.feat_cols), n_targets=n_out).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)

    def loss_of(b):
        pz, cmw = forward(model, b, args.arm, dev, ys_m, ys_s, c_m, c_s)
        yz = torch.from_numpy(b["Y"]).to(dev)
        cz_t = (torch.from_numpy(b["C"]).to(dev) - c_m) / c_s
        cz_p = (cmw - c_m) / c_s
        ld = ((pz - yz) ** 2).mean()                      # dispatch, z-scored
        lc = ((cz_p - cz_t) ** 2).mean()                  # curtailment, z-scored
        return ld + lc, ld, lc

    def val():
        model.eval(); s = np.zeros(3); n = 0
        with torch.no_grad():
            for i in range(0, len(va["X"]), 256):
                b = {k: (v[i:i + 256] if isinstance(v, np.ndarray) and v.ndim > 1 else v)
                     for k, v in va.items()}
                l, ld, lc = loss_of(b)
                s += np.array([float(l), float(ld), float(lc)]) * len(b["X"]); n += len(b["X"])
        model.train(); return s / n

    hist = {"train": [], "val": [], "val_disp": [], "val_curt": []}
    OUT.mkdir(parents=True, exist_ok=True)
    v0 = val(); print(f"  before training: val {v0[0]:.4f} (dispatch {v0[1]:.4f} "
                      f"curt {v0[2]:.4f})", flush=True)
    for ep in range(1, args.epochs + 1):
        te0 = time.time(); perm = rng.permutation(len(tr["X"])); s = 0.0; n = 0
        for i in range(0, len(perm), args.batch):
            j = perm[i:i + args.batch]
            b = {k: (v[j] if isinstance(v, np.ndarray) and v.ndim > 1 else v)
                 for k, v in tr.items()}
            l, ld, lc = loss_of(b)
            opt.zero_grad(); l.backward(); opt.step()
            s += float(l.detach()) * len(j); n += len(j)
            if (i // args.batch) % 60 == 0:
                print(f"    batch {i//args.batch:4d}/{len(perm)//args.batch} "
                      f"loss {float(l.detach()):.4f} (disp {float(ld):.4f} "
                      f"curt {float(lc):.4f})", flush=True)
        v = val()
        hist["train"].append(s / n); hist["val"].append(v[0])
        hist["val_disp"].append(v[1]); hist["val_curt"].append(v[2])
        print(f"  ep{ep:03d} train {s/n:.4f}  val {v[0]:.4f} "
              f"(dispatch {v[1]:.4f}  curt {v[2]:.4f})  ({time.time()-te0:.0f}s)",
              flush=True)

    torch.save(model.state_dict(), OUT / f"{tag}.pt")
    (OUT / f"{tag}_history.json").write_text(json.dumps(
        {"arm": args.arm, "epochs": args.epochs, **hist}, indent=1))
    print(f"\nwrote {OUT/f'{tag}.pt'}")


if __name__ == "__main__":
    main()
