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
import heads as HD                                                     # noqa: E402
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
    "hardnet_alloc": ("bilstm", "hardnet_alloc", 36),
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


ND_FEAT, DEM_FEAT = 6, 7          # net_demand, demand_mw columns of feat_cols


def response_targets(P, h=GAP, W=36):
    """Per-timestep marginal response a(t) (T,6) from history, by local OLS.

        dP_i(t) = P_i(t+h) - P_i(t),   dnd(t) = nd(t+h) - nd(t),   nd = SIGN.P
        a_i(t)  = sum_{|u-t|<=W} dP_i(u) dnd(u) / sum dnd(u)^2

    This is the supervision the counterfactual has no label for. Two properties
    make it the right target rather than a hand-picked constant:

      * sum_i s_i a_i(t) = 1 EXACTLY and for free -- sum_i s_i dP_i = dnd is the
        balance identity, so the numerator sums to sum dnd^2. The target is
        therefore always consistent with what the head already guarantees.
      * it is state-conditional. At W=+-36 (3h) the estimate at 08:00 is
        coal 0.53 / hydro 0.25; at 20:00 it is coal 0.23 / hydro 0.45 / ocgt 0.14
        -- the real merit-order shift into the evening peak. Pooled over all
        history it reproduces dispatch_study Study A to ~2pp.

    Widening W washes the conditioning out: at W=+-288 (24h) the between-hour
    spread collapses to <0.01 and the target IS a constant. Keep W small.
    """
    nd = P @ SIGN
    dP = np.zeros_like(P); dnd = np.zeros(len(P))
    dP[:-h] = P[h:] - P[:-h]; dnd[:-h] = nd[h:] - nd[:-h]
    cy = np.vstack([np.zeros(6), np.cumsum(dP * dnd[:, None], 0)])
    cx = np.concatenate([[0.0], np.cumsum(dnd * dnd)])
    lo = np.clip(np.arange(len(P)) - W, 0, len(P))
    hi = np.clip(np.arange(len(P)) + W + 1, 0, len(P))
    return (cy[hi] - cy[lo]) / np.maximum(cx[hi] - cx[lo], 1e-6)[:, None]


def resp_by_date(lo, hi, h=GAP, W=36):
    """response_targets over the FULL table, then sliced -- so the h-step
    lookahead and the local window are never truncated at the range edges."""
    import pandas as pd
    t = pd.read_parquet(TABLE)
    a = response_targets(t[TARGETS].values.astype(np.float64), h=h, W=W)
    m = ((t.index >= pd.Timestamp(lo, tz=t.index.tz))
         & (t.index <= pd.Timestamp(hi, tz=t.index.tz)))
    return a[m].astype(np.float32)


def apply_residual(d_raw, interp):
    """The interp skeleton is a DISPATCH LEVEL, so it applies to the first 6
    outputs only. Anything after them is a logit -- rayen's step, or
    hardnet_alloc's 6 allocation logits -- and passes through untouched."""
    if d_raw.shape[-1] == 6:
        return d_raw + interp
    return torch.cat([d_raw[..., :6] + interp, d_raw[..., 6:]], -1)


def build(f, starts, split, gap=GAP, flats=None, resp=None):
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
    out = {"X": X, "mask": mask, "Y": Y, "C": Cw.astype(np.float32), "interp": interp,
           "pL": f.y_to_mw(Yflat[starts + gs - 1]).astype(np.float32),
           "pR": f.y_to_mw(Yflat[starts + ge]).astype(np.float32),
           "nd": ((Y * f.y_scale + f.y_mean) @ SIGN).astype(np.float32),
           "c_mean": cm, "c_scale": cs}
    if resp is not None:
        # demand_mw inside the gap, in MW -- the perturbation is q * demand, the
        # same quantity FixedPercentageShift moves in the counterfactual.
        out["dem"] = (X[:, gs:ge, DEM_FEAT] * f.x_scale[DEM_FEAT]
                      + f.x_mean[DEM_FEAT]).astype(np.float32)
        # one response target per window: the mean over its gap steps
        out["a"] = np.stack([resp[s + gs:s + ge].mean(0) for s in starts])
        out["a_step"] = np.stack([resp[s + gs:s + ge] for s in starts])   # (n,G,6)
    return out


def peak_start_pool(index, n_starts, gap=GAP, hours=(17, 18)):
    """Start positions whose masked GAP opens inside `hours`.

    The default (17,18) brackets the 18:00-21:00 evaluation window: gaps run from
    17:00-20:00 through 18:55-21:55. Returns an int array of valid starts.

    Why this and not a sliding window: coverage is already ~11x per timestep and
    uniform over hour of day (quick_findings/planb_head_vs_network/coverage.py), so
    the peak is not UNSEEN -- it is 12.2% of the gradient because it is 12.5% of
    the clock. Stride-1 sliding keeps that ratio; over-drawing peak windows does
    not.
    """
    gs = np.arange(n_starts) + CTX                      # gap opens here
    hr = index.hour.values[gs]
    return np.flatnonzero((hr >= hours[0]) & (hr <= hours[1]))


def sample(f, n, split, seed, gap=GAP, flats=None, index=None, peak_frac=0.0,
           peak_hours=(17, 18), peak_only=False, resp=None):
    Xflat = (flats[0] if flats is not None else
             f.Xtr if split == "train" else f.Xva)
    n_starts = Xflat.shape[0] - (2 * CTX + gap)
    rng = np.random.default_rng(seed)
    if (peak_frac > 0.0 or peak_only) and index is not None:
        pool = peak_start_pool(index, n_starts, gap, peak_hours)
        n_pk = n if peak_only else int(round(n * peak_frac))
        starts = np.concatenate([rng.choice(pool, size=n_pk, replace=True),
                                 rng.integers(0, n_starts, size=n - n_pk)])
        rng.shuffle(starts)
    else:
        starts = rng.integers(0, n_starts, size=n)
    return build(f, starts, split, gap, flats, resp=resp)


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
    ap.add_argument("--peak-frac", type=float, default=0.0,
                    help="fraction of TRAIN windows forced to open inside "
                         "--peak-hours. 0 (default) is the uniform sampler every "
                         "existing arm used. Needs a date range so the index is "
                         "available. Coverage is NOT the reason to use this -- the "
                         "uniform sampler already masks every timestep ~11x; this "
                         "changes the peak's SHARE OF THE GRADIENT (12.2% -> frac).")
    ap.add_argument("--peak-hours", nargs=2, type=int, default=(17, 18),
                    metavar=("FROM", "TO"),
                    help="hour range in which the masked gap OPENS. Default 17 18 "
                         "brackets the 18:00-21:00 peak_eval window.")
    ap.add_argument("--pert-weight", type=float, default=0.0,
                    help="weight on the PERTURBATION loss. 0 (default) is off. "
                         "Runs a second forward with net demand shifted by "
                         "--pert-q inside the gap and supervises the resulting "
                         "response SHARES against the local marginal response "
                         "measured from history (response_targets). This is the "
                         "only term in the objective that ever sees a demand "
                         "CHANGE -- ordinary reconstruction cannot, because "
                         "nd == SIGN.P holds by identity on historical data.")
    ap.add_argument("--soc", action="store_true",
                    help="enforce the battery SOC swing inside the head "
                         "(heads.set_soc). Folds into the same per-step box, so "
                         "balance/ramp/cap stay exact. Over a 3 h gap with the "
                         "full reservoir it rarely binds -- it matters at "
                         "inference when seeded with the day's history.")
    ap.add_argument("--alloc-weight", type=float, default=0.0,
                    help="weight on the ALLOCATION cross-entropy for the "
                         "hardnet_alloc head. 0 (default) = UNSUPERVISED: the "
                         "shares are learned from reconstruction alone, via the "
                         "corrections the projection makes on historical windows. "
                         ">0 additionally supervises them against the measured "
                         "marginal response (response_targets).")
    ap.add_argument("--pert-q", type=float, default=0.024,
                    help="demand shift as a fraction, applied over the gap. "
                         "0.024 matches the counterfactual protocol's rebound.")
    ap.add_argument("--resp-w", type=int, default=36,
                    help="half-width (steps) of the local regression for the "
                         "response target. Small keeps it state-conditional: at "
                         "+-288 the between-hour spread collapses to <0.01 and "
                         "the target degenerates into a constant.")
    ap.add_argument("--early-on", default="global", choices=["global", "peak"],
                    help="which validation loss drives best-checkpoint and "
                         "patience. 'global' (default) is a whole-clock average in "
                         "which the evening peak is ~12% of the cells.")
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
    soc_cap = HD.set_soc(True if args.soc else None)
    # the rayen head consumes a DIRECTION, not a dispatch level, so the interp
    # residual has no meaning there -- its anchor already plays that role.
    residual = (not args.no_residual) and n_disp in (6, 12)
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
        need_resp = args.pert_weight > 0 or args.alloc_weight > 0
        rs_tr = (resp_by_date(*args.train_range, h=gap, W=args.resp_w)
                 if need_resp else None)
        rs_va = (resp_by_date(*(args.val_range or args.train_range), h=gap,
                              W=args.resp_w) if need_resp else None)
        tr = sample(f, args.n_train, "train", args.seed, gap, trf[:2],
                    index=trf[2], peak_frac=args.peak_frac,
                    peak_hours=tuple(args.peak_hours), resp=rs_tr)
        va = sample(f, args.n_val, "val", args.seed + 777, gap, vaf[:2],
                    resp=rs_va)
        # a SECOND validation set of peak-only windows, always built when the
        # index is available. The global val loss is a whole-clock average in
        # which the evening peak is ~12% of the cells, so it can improve while
        # the peak gets worse -- which is exactly what peak_18_21.md shows.
        va_pk = sample(f, args.n_val, "val", args.seed + 778, gap, vaf[:2],
                       index=vaf[2], peak_only=True,
                       peak_hours=tuple(args.peak_hours))
        if args.peak_frac > 0:
            print(f"peak sampling: {args.peak_frac:.0%} of train windows open in "
                  f"{args.peak_hours[0]:02d}:00-{args.peak_hours[1]:02d}:59 "
                  f"(uniform would be {100*(args.peak_hours[1]-args.peak_hours[0]+1)/24:.1f}%)",
                  flush=True)
    else:
        tr = sample(f, args.n_train, "train", args.seed, gap)
        va = sample(f, args.n_val, "val", args.seed + 777, gap)
        va_pk = None
        if args.peak_frac > 0 or args.early_on == "peak":
            raise SystemExit("--peak-frac/--early-on peak need --train-range "
                             "(the npz splits carry no datetime index)")

    if backbone == "bilstm":
        model = BiLSTMImputer(nfeat, n_disp, hidden=args.hidden).to(dev)
    else:
        model = BACKBONES[backbone](nfeat, TARGET_FEAT_IDX, CURT_COLS, n_disp,
                                    hidden=args.hidden).to(dev)
    print(f"arm={args.arm}  backbone={backbone}  head={head_name}  gap={gap}  "
          f"residual={residual}  loss={args.loss}  soc={soc_cap}  outputs={n_disp}  hidden={args.hidden}  "
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
    va_pk = to_dev(va_pk) if va_pk is not None else None

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
            d_raw = apply_residual(d_raw, b["interp"])
        if head_fn is None:
            return d_raw[..., :6], curt, d_raw
        F_ = (d_raw * ys_s + ys_m if n_disp == 6 else
              torch.cat([d_raw[..., :6] * ys_s + ys_m, d_raw[..., 6:]], -1))
        P = head_fn(F_, b["pL"], b["pR"], b["nd"])
        return (P - ys_m) / ys_s, curt, d_raw

    all_scale = torch.cat([ys_s, c_s])           # (8,) MW per z-unit
    all_mean = torch.cat([ys_m, c_m])
    xs_nd = float(f.x_scale[ND_FEAT]); xs_dem = float(f.x_scale[DEM_FEAT])

    def pert_from(b, pz0):
        """Second forward with net demand shifted by q*demand INSIDE the gap;
        supervise the resulting response shares against the measured local
        marginal response.

        The head already forces sum_i s_i dE_i = sum delta, and the target obeys
        sum_i s_i a_i = 1 by the balance identity, so the two are consistent and
        the term is drivable to zero. What it teaches is the ALLOCATION of a
        demand change -- the one thing gap reconstruction can never supply,
        because on historical data nd == SIGN.P and the change is always zero.
        """
        g0, g1 = CTX, CTX + gap
        d = args.pert_q * b["dem"]                       # (B,G) MW, per step
        X2 = b["X"].clone()
        X2[:, g0:g1, ND_FEAT] += d / xs_nd               # both features are z-scored
        X2[:, g0:g1, DEM_FEAT] += d / xs_dem
        b2 = dict(b); b2["X"] = X2; b2["nd"] = b["nd"] + d
        pz1, _, _ = forward(b2)
        dE = ((pz1 - pz0) * ys_s).sum(1)                 # (B,6) MW-steps of response
        share = dE / d.sum(1, keepdim=True).clamp_min(1e-3)
        return ((share - b["a"]) ** 2).mean()

    sgn_t = torch.tensor(SIGN, dtype=torch.float32, device=dev)

    def alloc_loss(b, d_raw):
        """Cross-entropy between the head's predicted allocation and the measured
        one. Target is s_i*a_i -- the fraction of a demand change channel i
        actually absorbed -- clamped at 0 and renormalised, so it is a point on
        the same simplex the softmax lives on. Used only for the SUPERVISED
        variant; with --alloc-weight 0 the shares are learned from the
        reconstruction loss alone, through the corrections the projection makes
        on ordinary historical windows."""
        p_ = torch.softmax(d_raw[..., 6:], -1)
        tgt = (b["a_step"] * sgn_t).clamp_min(0.0)
        tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-6)
        return -(tgt * p_.clamp_min(1e-8).log()).sum(-1).mean()

    def loss_of(b, pert=True):
        # `pert` gates every TRAINING-ONLY auxiliary term (perturbation and
        # allocation). val() passes pert=False so the reported val loss stays the
        # same quantity every other arm reports.
        pz, cmw, d_raw = forward(b)
        extra = (args.pert_weight * pert_from(b, pz)
                 if (pert and args.pert_weight > 0 and "a" in b) else 0.0)
        if pert and args.alloc_weight > 0 and n_disp == 12 and "a_step" in b:
            extra = extra + args.alloc_weight * alloc_loss(b, d_raw)
        cz_p = (cmw - c_m) / c_s
        pred_z = torch.cat([pz, cz_p], -1)                       # (B,G,8) z
        true_z = torch.cat([b["Y"], (b["C"] - c_m) / c_s], -1)
        if args.loss == "mse":
            # one MSE over all eight z-scored channels. Identical to summing the
            # per-source MSEs up to a factor of 8, which Adam absorbs.
            return ((pred_z - true_z) ** 2).mean() + extra
        err_mw = (pred_z - true_z) * all_scale                   # back to MW
        if args.loss == "sum_mw":
            # per-source MSE in MW, summed: channels weighted by scale^2, so coal
            # dominates battery ~170x. This is what "sum the per-source errors"
            # means if you do NOT normalise first.
            return (err_mw ** 2).mean((0, 1)).sum() + extra
        # wape: each source scored against its own total, so a 50 MW battery error
        # and a 700 MW coal error are comparable
        true_mw = true_z * all_scale + all_mean
        num = err_mw.abs().sum((0, 1))
        den = true_mw.abs().sum((0, 1)).clamp_min(1.0)
        return (num / den).mean() + extra

    def val(ds=None, want_resp=False):
        ds = va if ds is None else ds
        model.eval(); s = n = r = 0.0
        with torch.no_grad():
            vb = max(256, args.batch)
            for i in range(0, len(ds["X"]), vb):
                b = {k: (v[i:i + vb] if torch.is_tensor(v) and v.dim() > 1 else v)
                     for k, v in ds.items()}
                s += float(loss_of(b, pert=False)) * len(b["X"])
                if want_resp and "a" in b:
                    r += float(pert_from(b, forward(b)[0])) * len(b["X"])
                n += len(b["X"])
        model.train()
        return (s / n, r / n) if want_resp else s / n

    hist = {"train": [], "val": [], "val_peak": []}
    OUT.mkdir(parents=True, exist_ok=True)
    best, best_state, waited = float("inf"), None, 0
    print(f"  val before training: {val():.4f}"
          + (f"  peak {val(va_pk):.4f}" if va_pk is not None else ""), flush=True)
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
        vv = val(want_resp=args.pert_weight > 0)
        v, vr = vv if args.pert_weight > 0 else (vv, float("nan"))
        vp = val(va_pk) if va_pk is not None else float("nan")
        hist["train"].append(s / n); hist["val"].append(v)
        hist["val_peak"].append(vp); hist.setdefault("val_resp", []).append(vr)
        sel = vp if args.early_on == "peak" else v      # what drives the checkpoint
        stop = False
        if sel < best - 1e-6:
            best, waited = sel, 0
            best_state = {k: t.detach().clone()
                          for k, t in model.state_dict().items()}
        else:
            waited += 1
            stop = args.patience > 0 and waited >= args.patience
        print(f"  ep{ep:03d} train {s/n:.4f}  val {v:.4f}"
              + (f"  val_peak {vp:.4f}" if va_pk is not None else "")
              + (f"  val_resp {vr:.4f}" if args.pert_weight > 0 else "")
              + f"  (best[{args.early_on}] {best:.4f}, waited {waited}"
              f"{'/' + str(args.patience) if args.patience else ''})  "
              f"({time.time()-te0:.0f}s){'  EARLY STOP' if stop else ''}", flush=True)
        # persist every epoch so a dropped Colab session loses at most one
        (OUT / f"{tag}_history.json").write_text(json.dumps(
            {"arm": args.arm, "backbone": backbone, "head": head_name,
             "gap": gap, "residual": residual, "loss": args.loss,
             "params": n_params(model),
             "train_range": args.train_range, "val_range": args.val_range,
             "peak_frac": args.peak_frac, "peak_hours": list(args.peak_hours),
             "early_on": args.early_on, "pert_weight": args.pert_weight,
             "pert_q": args.pert_q, "resp_w": args.resp_w,
                "alloc_weight": args.alloc_weight,
                "soc": soc_cap,
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
                # provenance: without these a checkpoint's training window is
                # unrecoverable (hardnet_2025 has this problem).
                "train_range": args.train_range, "val_range": args.val_range,
                "peak_frac": args.peak_frac, "peak_hours": list(args.peak_hours),
                "early_on": args.early_on, "pert_weight": args.pert_weight,
                "pert_q": args.pert_q, "resp_w": args.resp_w,
                "alloc_weight": args.alloc_weight,
                "soc": soc_cap,
                "hidden": args.hidden, "n_disp": n_disp}, OUT / f"{tag}.pt")
    print(f"\nbest val {best:.4f} -> wrote {OUT/f'{tag}.pt'}")


if __name__ == "__main__":
    main()
