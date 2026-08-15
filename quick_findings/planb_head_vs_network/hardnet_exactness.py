"""How exact is hardnet, and where exactly does it stop being exact?

eval.md reports the WORST cell. This reports the distribution, and separates the
two possible causes of a nonzero balance residual:

  (a) solver error                 -> would show up as residual with nd INSIDE
                                      the box's attainable range
  (b) empty feasible set           -> the hyperplane s.P = nd misses the box
                                      entirely, so no P can satisfy balance

For a box [lo,hi] and s with entries +-1, the attainable range of s.P is
    [ sum_i min(s_i*lo_i, s_i*hi_i),  sum_i max(s_i*lo_i, s_i*hi_i) ]
so (b) is directly checkable.
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
from gap_data import load_flats, TARGET_FEAT_IDX, SIGN                 # noqa: E402
from nets import build_delta                                           # noqa: E402
from heads import HEADS, hardnet_head, _box                            # noqa: E402
from limits import limits as _limits, R_UP, R_DN, CAP                  # noqa: E402
from fit import build, CURT_COLS, CTX, GAP, BACKBONES                  # noqa: E402

CKPT = ROOT / "colab" / "planB_results"
N, SEED = 400, 123

f = load_flats()
nfeat = len(f.feat_cols)
W = 2 * CTX + GAP
rng = np.random.default_rng(SEED)
starts = rng.integers(max(0, f.lb_carry - CTX), f.Xte.shape[0] - W, size=N)
b = build(f, starts, "test")
ys_m = torch.tensor(f.y_mean, dtype=torch.float32)
ys_s = torch.tensor(f.y_scale, dtype=torch.float32)
pL, pR, nd = b["pL"], b["pR"], b["nd"]
truth = b["Y"] * f.y_scale + f.y_mean

tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
interp = pL[:, None] + tt * (pR - pL)[:, None]


def reachable(F64):
    """Walk the same greedy sweep and record, per step, whether nd_t lies inside
    the box's attainable balance range."""
    dev, dt = torch.device("cpu"), torch.float64
    sign, rup, rdn, cap = _limits(dev, dt)
    tF = torch.tensor(F64); tpL = torch.tensor(pL, dtype=dt)
    tpR = torch.tensor(pR, dtype=dt); tnd = torch.tensor(nd, dtype=dt)
    P_prev, gaps = tpL, []
    Ps = []
    from heads import project_box_plane
    for j in range(GAP):
        lo, hi = _box(P_prev, tpR, GAP - j, rup, rdn, cap)
        a = torch.minimum(lo * sign, hi * sign).sum(-1)     # min attainable s.P
        c = torch.maximum(lo * sign, hi * sign).sum(-1)     # max attainable s.P
        gaps.append(torch.maximum(torch.maximum(a - tnd[:, j], tnd[:, j] - c),
                                  torch.zeros_like(a)))
        P_t = project_box_plane(tF[:, j], lo, hi, tnd[:, j], sign)
        Ps.append(P_t); P_prev = P_t
    return torch.stack(Ps, 1).numpy(), torch.stack(gaps, 1).numpy()


def report(name, F64):
    P, unreach = reachable(F64)
    res = np.abs(P @ SIGN - nd)                      # (n,G) post-head residual
    ex = 100 * (res < 1e-6).mean()
    # is the residual explained by the hyperplane missing the box?
    bad = res > 1e-6
    expl = 100 * (np.abs(res[bad] - unreach[bad]) < 1e-6).mean() if bad.any() else 100.0
    ramp = np.concatenate([pL[:, None], P, pR[:, None]], 1)
    d = np.diff(ramp, axis=1)
    ro = (np.maximum(d - R_UP, 0) + np.maximum(-d - R_DN, 0)).max()
    print(f"{name:34s} exact {ex:6.2f}%  mean {res.mean():7.3f}  p99 "
          f"{np.percentile(res,99):7.2f}  max {res.max():8.2f} | "
          f"unreachable-explained {expl:6.1f}%  neg {max(0,-P.min()):5.2f}  "
          f"cap {np.maximum(P-CAP,0).max():5.2f}  ramp {ro:5.2f}")


print(f"{N} test 3h gaps, seed {SEED}. Residual = |SIGN.P - nd| AFTER the head.\n")
print("input to the head".ljust(34), "balance residual (MW)".center(48), "| diagnosis")
report("TRUTH (the feasibility floor)", truth.astype(np.float64))
report("interp skeleton, zero network", interp.astype(np.float64))

for arm in ("hardnet", "saits"):
    ck = torch.load(CKPT / f"{arm}.pt", map_location="cpu", weights_only=False)
    head_fn, n_disp = HEADS[ck["head"]]
    bb = ck.get("backbone", "bilstm")
    m = (BACKBONES[bb](nfeat, n_disp, hidden=ck["hidden"]) if bb == "bilstm"
         else BACKBONES[bb](nfeat, TARGET_FEAT_IDX, CURT_COLS, n_disp, hidden=ck["hidden"]))
    m.load_state_dict(ck["state"]); m.eval()
    raws = []
    for i in range(0, N, 64):
        sl = slice(i, i + 64)
        with torch.no_grad():
            x = torch.from_numpy(b["X"][sl]); mk = torch.from_numpy(b["mask"][sl])
            dl = build_delta(mk, nfeat) if bb == "brits" else None
            d_raw, _ = m(x, mk, dl)
            d_raw = d_raw[:, CTX:CTX + GAP]
            if ck.get("residual", False):
                d_raw = d_raw + torch.from_numpy(b["interp"][sl])
            raws.append((d_raw[..., :6] * ys_s + ys_m).numpy())
    report(f"{arm} ({bb}) raw", np.concatenate(raws, 0).astype(np.float64))
