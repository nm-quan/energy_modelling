"""rayen vs hardnet: how much of each head's answer is the network?

Run under BOTH envelopes, because that is itself a discriminator:
  hardnet  = argmin ||P - F|| over the feasible set. Widen the set and the
             projection simply moves less; the answer stays anchored to F.
  rayen    = A + step * alpha_max * r, where the anchor A and alpha_max are both
             BUILT FROM the box. Change the box and the answer changes even with
             the network frozen.

Fixed part of each head (zero network contribution):
  hardnet  hardnet_head(interp skeleton)      -- the head's own default input
  rayen    the anchor A                       -- returned by rayen_head debug
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/Users/nguyenminhquan/Downloads/energy_modelling")
sys.path.insert(0, str(ROOT / "imputation"))
sys.path.insert(0, str(ROOT / "planB"))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
from gap_data import load_flats, TARGET_FEAT_IDX                       # noqa: E402
import limits as LIM                                                   # noqa: E402
from fit import build, CTX, GAP, CURT_COLS, BACKBONES                  # noqa: E402

CKPT = ROOT / "colab" / "planB_results"
N, SEED = 400, 123
f = load_flats(); nfeat = len(f.feat_cols); W = 2 * CTX + GAP
rng = np.random.default_rng(SEED)
starts = rng.integers(max(0, f.lb_carry - CTX), f.Xte.shape[0] - W, size=N)
b = build(f, starts, "test")
ys_m = torch.tensor(f.y_mean, dtype=torch.float32)
ys_s = torch.tensor(f.y_scale, dtype=torch.float32)
pL, pR, nd = b["pL"], b["pR"], b["nd"]
truth = b["Y"] * f.y_scale + f.y_mean
mae = lambda P: float(np.abs(P - truth).mean())
tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
interp = pL[:, None] + tt * (pR - pL)[:, None]


def raw_of(arm, n_disp, bb):
    ck = torch.load(CKPT / f"{arm}.pt", map_location="cpu", weights_only=False)
    m = (BACKBONES[bb](nfeat, n_disp, hidden=ck["hidden"]) if bb == "bilstm"
         else BACKBONES[bb](nfeat, TARGET_FEAT_IDX, CURT_COLS, n_disp, hidden=ck["hidden"]))
    m.load_state_dict(ck["state"]); m.eval()
    out = []
    for i in range(0, N, 64):
        sl = slice(i, i + 64)
        with torch.no_grad():
            d, _ = m(torch.from_numpy(b["X"][sl]), torch.from_numpy(b["mask"][sl]), None)
            d = d[:, CTX:CTX + GAP]
            if ck.get("residual", False):
                d = d + torch.from_numpy(b["interp"][sl])
            out.append((d[..., :6] * ys_s + ys_m if n_disp == 6 else
                        torch.cat([d[..., :6] * ys_s + ys_m, d[..., 6:]], -1)).numpy())
    return torch.tensor(np.concatenate(out, 0))


RAW = {a: raw_of(a, n, bb) for a, n, bb in
       (("rayen", 7, "bilstm"), ("hardnet", 6, "bilstm"), ("saits", 6, "saits"))}
tT = lambda a: torch.tensor(a)

for label, pct, ov in (("p99.9 (what they were TRAINED and evaluated under)", 99.9, {}),
                       ("empirical max + p99.5 coal/steam (CURRENT limits.py)", None, None)):
    up, dn = LIM.ramp_percentile(pct=pct, override=ov)
    LIM.R_UP, LIM.R_DN = up, dn
    LIM._RUP_T = torch.tensor(up, dtype=torch.float32)
    LIM._RDN_T = torch.tensor(dn, dtype=torch.float32)
    import importlib, heads
    importlib.reload(heads)
    print(f"\n=== envelope: {label} ===")
    P, dbg = heads.rayen_head(RAW["rayen"], tT(pL), tT(pR), tT(nd), return_debug=True)
    A = dbg["anchor"].numpy().astype(np.float64)
    C = dbg["glide"].numpy().astype(np.float64)
    Pr = P.numpy().astype(np.float64)
    st, am = dbg["step"].numpy(), dbg["alpha_max"].numpy()
    Ph = heads.hardnet_head(RAW["hardnet"], tT(pL), tT(pR), tT(nd)).numpy().astype(np.float64)
    Pi = heads.hardnet_head(tT(interp), tT(pL), tT(pR), tT(nd)).numpy().astype(np.float64)
    Pt = heads.hardnet_head(tT(truth), tT(pL), tT(pR), tT(nd)).numpy().astype(np.float64)

    print(f"  {'interp (no head, no network)':46s} {mae(interp):6.1f}")
    print(f"  {'HARDNET fixed part  hardnet_head(interp)':46s} {mae(Pi):6.1f}")
    print(f"  {'hardnet full':46s} {mae(Ph):6.1f}"
          f"   network buys {mae(Pi)-mae(Ph):+6.1f}")
    print(f"  {'RAYEN fixed part 1  glide c (geometry only)':46s} {mae(C):6.1f}")
    print(f"  {'RAYEN fixed part 2  anchor A (zero network)':46s} {mae(A):6.1f}")
    print(f"  {'rayen full':46s} {mae(Pr):6.1f}"
          f"   network buys {mae(A)-mae(Pr):+6.1f}")
    d = np.abs(Pr - A).sum(-1)
    print(f"  rayen ray: sigmoid(step) med {np.median(st):.3f}  alpha_max med "
          f"{np.median(am):6.1f}  |P-A|_1 med {np.median(d):6.1f} MW  "
          f"no-op steps {100*(d<1).mean():.1f}%")
    print(f"  hardnet on the TRUTH: mean |out-truth| {np.abs(Pt-truth).mean():.3f} MW "
          f"({100*(np.abs(Pt-truth).sum(-1)<1).mean():.1f}% of steps returned unchanged)")
