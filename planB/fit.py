"""planB training (file is `fit.py`, not `train.py`, so imputation/train.py on the
shared sys.path cannot shadow it). One script, four arms, one loss.

    python3 planB/fit.py --arm unconstrained
    python3 planB/fit.py --arm rayen
    python3 planB/fit.py --arm hardnet
    python3 planB/fit.py --arm brits          # BRITS backbone + hardnet head

DELIBERATELY SIMPLE, per the planB brief:
  * ONE MSE over all 8 z-scored channels. No lambda, no separate curtailment
    term. The planA loss was `ld + lc` -- two means added -- which silently gave
    curtailment 3x the per-channel weight.
  * relu in z-space for curtailment (`curt_activation`), never softplus. The
    target is exactly zero in 56%/66% of gap cells; softplus is strictly positive
    and cannot produce that value, and in raw MW it also forced the head to travel
    to ~102 from an init of +-0.06.
  * SPLIT output heads for dispatch and curtailment (fix #5).
  * NO curtailment credit in the balance target. planA coupled
    nd_eff = nd - (curt_actual - curt_pred), which leaks the true in-gap
    curtailment at eval time and cost accuracy on every dispatch channel.
  * p99.9 ramp envelope (planB/limits.py), not the all-time max.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# imputation FIRST so that planB ends up at position 0 and its own
# nets.py / train.py are not shadowed by the imputation ones
sys.path.insert(0, str(ROOT / "imputation")); sys.path.insert(0, str(HERE))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "prepared.npz"
from gap_data import load_flats, TARGETS, SIGN, TARGET_FEAT_IDX        # noqa: E402
from nets import (BiLSTMImputer, BRITSImputer, SAITSImputer,            # noqa: E402
                  build_delta, n_params)
from heads import HEADS                                                # noqa: E402

OUT = HERE / "results"
CURT_COLS = [19, 20]
CTX, GAP = 48, 36

# arm -> (backbone, head, gap length). "blackout" masks a WHOLE DAY (288 steps)
# instead of 3 hours, which is the task quick_findings/ and itr_bench.py use.
ARMS = {
    "unconstrained": ("bilstm", "none",    36),
    "rayen":         ("bilstm", "rayen",   36),
    "hardnet":       ("bilstm", "hardnet", 36),
    "brits":         ("brits",  "hardnet", 36),
    "brits_rayen":   ("brits",  "rayen",   36),
    "blackout":      ("bilstm", "hardnet", 288),
    "saits":         ("saits",  "hardnet", 36),
}
BACKBONES = {"bilstm": BiLSTMImputer, "brits": BRITSImputer, "saits": SAITSImputer}


def curt_activation(raw2, c_mean, c_scale):
    """Raw logits -> curtailment in MW. relu (the target is exactly zero most of
    the time) applied in z-space (so raw=0 lands on the training mean)."""
    return torch.relu(raw2 * c_scale + c_mean)


ND_COL = 6


def _supply_side_nd(X, f, tfi):
    """Overwrite the net_demand FEATURE with the supply-side quantity SIGN.dispatch.

    The models were being fed `net_demand = demand - wind - solar - curtailment`
    (plan1_data.py) while the constraint head balances them to `SIGN.dispatch`.
    Those differ by the curtailment term plus an export residual -- measured, mean
    391 MW and up to 4,974 MW, e.g. +1,104 MW on 2026-06-04. So the model read one
    net demand and was scored against another.

    Under the isolated-Victoria assumption (no interconnector, no import/export)
    the correct net demand is what the local fleet must supply, which IS
    SIGN.dispatch. It is known at every step even inside the gap -- only the
    BREAKDOWN is unknown -- so computing it here, before the dispatch columns are
    masked, is legitimate and not leakage.

    Reuses the existing column-6 scaler. That is an affine offset of ~0.4 z-units,
    which the network absorbs; refitting would invalidate the saved scalers for no
    benefit.
    """
    disp = X[:, :, tfi] * f.x_scale[tfi] + f.x_mean[tfi]        # (n,W,6) MW
    nd = disp @ SIGN.astype(np.float32)                          # (n,W) MW
    X[:, :, ND_COL] = ((nd - f.x_mean[ND_COL]) / f.x_scale[ND_COL]).astype(np.float32)
    return X


TABLE = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "table.parquet"


def flats_by_date(f, lo, hi):
    """Standardised (X, Y) flats for an arbitrary date range, straight from the
    table. Uses the npz's ORIGINAL scalers -- refitting on a shorter window would
    make the numbers incomparable with every other arm.

    Needed because the npz splits are fixed (train <=2025-06-30, val <=2025-12-31,
    test from 2026-01-01), so 'train on 2025 plus part of 2026' spans all three.
    """
    import pandas as pd
    t = pd.read_parquet(TABLE)
    m = (t.index >= pd.Timestamp(lo, tz=t.index.tz)) & (t.index <= pd.Timestamp(hi, tz=t.index.tz))
    d = t[m]
    X = ((d[f.feat_cols].values - f.x_mean) / f.x_scale).astype(np.float32)
    Y = ((d[TARGETS].values - f.y_mean) / f.y_scale).astype(np.float32)
    return X, Y, d.index


def build(f, starts, split, gap=GAP, flats=None):
    Xflat, Yflat = (flats if flats is not None else
                    (f.Xtr, f.Ytr) if split == "train" else
                    (f.Xva, f.Yva) if split == "val" else (f.Xte, f.Yte))
    W = 2 * CTX + gap
    gs, ge = CTX, CTX + gap
    tfi = np.asarray(TARGET_FEAT_IDX)
    X = np.stack([Xflat[s:s + W] for s in starts]).astype(np.float32)
    X = _supply_side_nd(X, f, tfi)          # BEFORE masking: nd is known in the gap
    Y = np.stack([Yflat[s + gs:s + ge] for s in starts]).astype(np.float32)
    cm = f.x_mean[CURT_COLS].astype(np.float32)
    cs = f.x_scale[CURT_COLS].astype(np.float32)
    Cw = X[:, gs:ge][:, :, CURT_COLS] * cs + cm
    mask = np.ones((len(starts), W, 1), np.float32); mask[:, gs:ge] = 0.0
    X[:, gs:ge, tfi] = 0.0
    X[:, gs:ge, CURT_COLS] = 0.0
    # linear interpolation skeleton in Z-SPACE. planA parameterised the fill as
    # interp + deviation; planB dropped it and made every arm learn the whole
    # trajectory from scratch, which is a large part of why they trail plain
    # interpolation (50.6 MW). Restored here as the default.
    tt = (np.arange(1, gap + 1) / (gap + 1))[None, :, None].astype(np.float32)
    pLz, pRz = Yflat[starts + gs - 1], Yflat[starts + ge]
    interp = (pLz[:, None] + tt * (pRz - pLz)[:, None]).astype(np.float32)
    return {"X": X, "mask": mask, "Y": Y, "C": Cw.astype(np.float32), "interp": interp,
            "pL": f.y_to_mw(Yflat[starts + gs - 1]).astype(np.float32),
            "pR": f.y_to_mw(Yflat[starts + ge]).astype(np.float32),
            "nd": ((Y * f.y_scale + f.y_mean) @ SIGN).astype(np.float32),
            "c_mean": cm, "c_scale": cs}


def sample(f, n, split, seed, gap=GAP, flats=None):
    Xflat = (flats[0] if flats is not None else
             f.Xtr if split == "train" else f.Xva)
    rng = np.random.default_rng(seed)
    return build(f, rng.integers(0, Xflat.shape[0] - (2 * CTX + gap), size=n),
                 split, gap, flats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--no-residual", action="store_true",
                    help="predict the dispatch level directly instead of a "
                         "deviation from the interpolation skeleton")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many epochs without a val "
                         "improvement; 0 disables. The saved checkpoint "
                         "is always the BEST epoch, not the last.")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="mse", choices=["mse", "sum_mw", "wape"],
                    help="mse    (default) mean squared error over all 8 channels "
                         "AFTER z-scoring each by its own scaler. Note this is "
                         "identical to 'sum of the per-source MSEs' up to a factor "
                         "of 8, so that is not a separate option. | "
                         "sum_mw  per-source MSE in RAW MW, summed. No z-scoring, so "
                         "channels are weighted by scale^2 and coal (scale 681) "
                         "swamps battery (52) by ~170x. | "
                         "wape   per-source sum|err|/sum|truth| in MW, meaned. Matches "
                         "the reported metric, so no channel is down-weighted for "
                         "being small.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the BACKBONE only. The constraint heads are "
                         "left eager: they carry data-dependent control flow (a "
                         "36-step sweep with a 40-step bisection inside) that either "
                         "graph-breaks or produces a huge unrolled graph. Helps SAITS "
                         "and BRITS most, nn.LSTM least (cuDNN is already fused).")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast around the BACKBONE only; the head always runs "
                         "in fp32. Running the head in bf16 would put the balance "
                         "residual around 1e-2 relative -- tens of MW on a 5 GW stack "
                         "-- which destroys the exactness the layer exists for.")
    ap.add_argument("--train-range", nargs=2, metavar=("FROM", "TO"), default=None,
                    help="train on this date range instead of the npz train split, "
                         "e.g. --train-range 2025-01-01 2026-03-31. Spans the fixed "
                         "npz splits, so it is built from the table with the npz "
                         "scalers.")
    ap.add_argument("--val-range", nargs=2, metavar=("FROM", "TO"), default=None)
    ap.add_argument("--tag", default=None, help="checkpoint name override")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_val, args.batch = 1024, 256, 64
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        # TF32 is safe here: the heads are elementwise (clamp/add/mul/sum), not
        # matmul, so the exact arithmetic is untouched. Only the backbone gemms
        # are affected.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"gpu: {torch.cuda.get_device_name(0)}  "
              f"tf32 on  amp={args.amp}  compile={args.compile}", flush=True)
    backbone, head_name, gap = ARMS[args.arm]
    head_fn, n_disp = HEADS[head_name]
    # the rayen head consumes a DIRECTION, not a dispatch level, so the interp
    # residual has no meaning there -- its anchor already plays that role.
    residual = (not args.no_residual) and n_disp == 6
    tag = args.tag or (args.arm + ("" if args.loss == "mse" else f"_{args.loss}")
                       + ("_smoke" if args.smoke else ""))

    f = load_flats()
    nfeat = len(f.feat_cols)
    t0 = time.time()
    if args.train_range:
        trf = flats_by_date(f, *args.train_range)
        vaf = flats_by_date(f, *(args.val_range or args.train_range))
        print(f"train window {args.train_range[0]} -> {args.train_range[1]}  "
              f"{len(trf[2]):,} rows ({len(trf[2])//288} days)", flush=True)
        print(f"val   window {(args.val_range or args.train_range)[0]} -> "
              f"{(args.val_range or args.train_range)[1]}  {len(vaf[2]):,} rows "
              f"({len(vaf[2])//288} days)", flush=True)
        tr = sample(f, args.n_train, "train", args.seed, gap, trf[:2])
        va = sample(f, args.n_val, "val", args.seed + 777, gap, vaf[:2])
    else:
        tr = sample(f, args.n_train, "train", args.seed, gap)
        va = sample(f, args.n_val, "val", args.seed + 777, gap)

    if backbone == "bilstm":
        model = BiLSTMImputer(nfeat, n_disp, hidden=args.hidden).to(dev)
    else:
        model = BACKBONES[backbone](nfeat, TARGET_FEAT_IDX, CURT_COLS, n_disp,
                                    hidden=args.hidden).to(dev)
    print(f"arm={args.arm}  backbone={backbone}  head={head_name}  gap={gap}  "
          f"residual={residual}  loss={args.loss}  outputs={n_disp}  hidden={args.hidden}  "
          f"params={n_params(model):,}  device={dev}", flush=True)
    print(f"windows: train {len(tr['X']):,} val {len(va['X']):,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Move the whole window set to the device ONCE. 40k x 132 x 21 float32 is
    # ~450 MB, trivial on an A100, and it removes a host-to-device copy per batch
    # -- which is the actual bottleneck at large batch, not the matmuls.
    def to_dev(dd):
        return {k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray)
                    and v.ndim > 1 else v) for k, v in dd.items()}
    tr, va = to_dev(tr), to_dev(va)

    ys_m = torch.tensor(f.y_mean, dtype=torch.float32, device=dev)
    ys_s = torch.tensor(f.y_scale, dtype=torch.float32, device=dev)
    c_m = torch.tensor(tr["c_mean"], device=dev)
    c_s = torch.tensor(tr["c_scale"], device=dev)
    if args.compile:
        model = torch.compile(model)
        print("  backbone compiled (first batch will be slow)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)

    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if args.amp and dev == "cuda"
               else contextlib.nullcontext())

    def forward(b):
        x, mk = b["X"], b["mask"]
        dl = build_delta(mk, x.shape[-1]) if backbone == "brits" else None
        with amp_ctx:                                  # BACKBONE only
            d_raw, c_raw = model(x, mk, dl)
        d_raw = d_raw[:, CTX:CTX + gap].float()        # back to fp32 for the head
        c_raw = c_raw[:, CTX:CTX + gap].float()
        curt = curt_activation(c_raw, c_m, c_s)
        if residual:                       # deviation from the interp skeleton
            d_raw = d_raw + b["interp"]
        if head_fn is None:
            return d_raw[..., :6], curt
        F_ = (d_raw * ys_s + ys_m if n_disp == 6 else
              torch.cat([d_raw[..., :6] * ys_s + ys_m, d_raw[..., 6:]], -1))
        P = head_fn(F_, b["pL"], b["pR"], b["nd"])
        return (P - ys_m) / ys_s, curt

    all_scale = torch.cat([ys_s, c_s])           # (8,) MW per z-unit
    all_mean = torch.cat([ys_m, c_m])

    def loss_of(b):
        pz, cmw = forward(b)
        cz_p = (cmw - c_m) / c_s
        pred_z = torch.cat([pz, cz_p], -1)                       # (B,G,8) z
        true_z = torch.cat([b["Y"], (b["C"] - c_m) / c_s], -1)
        if args.loss == "mse":
            # one MSE over all eight z-scored channels. Identical to summing the
            # per-source MSEs up to a factor of 8, which Adam absorbs.
            return ((pred_z - true_z) ** 2).mean()
        err_mw = (pred_z - true_z) * all_scale                   # back to MW
        if args.loss == "sum_mw":
            # per-source MSE in MW, summed: channels weighted by scale^2, so coal
            # dominates battery ~170x. This is what "sum the per-source errors"
            # means if you do NOT normalise first.
            return (err_mw ** 2).mean((0, 1)).sum()
        # wape: each source scored against its own total, so a 50 MW battery error
        # and a 700 MW coal error are comparable
        true_mw = true_z * all_scale + all_mean
        num = err_mw.abs().sum((0, 1))
        den = true_mw.abs().sum((0, 1)).clamp_min(1.0)
        return (num / den).mean()

    def val():
        model.eval(); s = n = 0.0
        with torch.no_grad():
            vb = max(256, args.batch)
            for i in range(0, len(va["X"]), vb):
                b = {k: (v[i:i + vb] if torch.is_tensor(v) and v.dim() > 1 else v)
                     for k, v in va.items()}
                s += float(loss_of(b)) * len(b["X"]); n += len(b["X"])
        model.train(); return s / n

    hist = {"train": [], "val": []}
    OUT.mkdir(parents=True, exist_ok=True)
    best, best_state, waited = float("inf"), None, 0
    print(f"  val before training: {val():.4f}", flush=True)
    for ep in range(1, args.epochs + 1):
        te0 = time.time(); s = n = 0.0
        perm = torch.from_numpy(rng.permutation(len(tr["X"]))).to(dev)
        for i in range(0, len(perm), args.batch):
            j = perm[i:i + args.batch]
            b = {k: (v[j] if torch.is_tensor(v) and v.dim() > 1 else v)
                 for k, v in tr.items()}
            l = loss_of(b)
            opt.zero_grad(); l.backward(); opt.step()
            s += float(l.detach()) * len(j); n += len(j)
            if (i // args.batch) % 60 == 0:
                print(f"    batch {i//args.batch:4d}/{len(perm)//args.batch} "
                      f"loss {float(l.detach()):.4f}", flush=True)
        v = val(); hist["train"].append(s / n); hist["val"].append(v)
        stop = False
        if v < best - 1e-6:
            best, waited = v, 0
            best_state = {k: t.detach().clone()
                          for k, t in model.state_dict().items()}
        else:
            waited += 1
            stop = args.patience > 0 and waited >= args.patience
        print(f"  ep{ep:03d} train {s/n:.4f}  val {v:.4f}  "
              f"(best {best:.4f}, waited {waited}"
              f"{'/' + str(args.patience) if args.patience else ''})  "
              f"({time.time()-te0:.0f}s){'  EARLY STOP' if stop else ''}", flush=True)
        # persist every epoch so a dropped Colab session loses at most one
        (OUT / f"{tag}_history.json").write_text(json.dumps(
            {"arm": args.arm, "backbone": backbone, "head": head_name,
             "gap": gap, "residual": residual, "loss": args.loss,
             "params": n_params(model),
             "best_val": best, "epochs_run": ep, **hist}, indent=1))
        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    sd = model.state_dict()
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}   # undo compile wrap

    torch.save({"state": sd, "arm": args.arm, "head": head_name,
                "backbone": backbone, "gap": gap, "residual": residual,
                "loss": args.loss,
                "hidden": args.hidden, "n_disp": n_disp}, OUT / f"{tag}.pt")
    print(f"\nbest val {best:.4f} -> wrote {OUT/f'{tag}.pt'}")


if __name__ == "__main__":
    main()
