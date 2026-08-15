"""Did the model learn the merit order?

Compares the hardnet_alloc head's PREDICTED allocation p = softmax(logits)
against the measured marginal response share s_i*a_i, on held-out test gaps.

This is the metric that answers the objection directly: p is a network output
computed from the inputs at inference, so if it tracks the measured response the
allocation is learned, not looked up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/Users/nguyenminhquan/Downloads/energy_modelling")
sys.path.insert(0, str(ROOT / "imputation"))
sys.path.insert(0, str(ROOT / "planB"))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
from gap_data import load_flats, TARGETS, SIGN, TARGET_FEAT_IDX        # noqa: E402
from heads import HEADS                                                # noqa: E402
from fit import (build, CTX, GAP, CURT_COLS, BACKBONES, apply_residual,
                 response_targets, TABLE)                              # noqa: E402

N, SEED = 400, 123
f = load_flats(); nfeat = len(f.feat_cols); W = 2 * CTX + GAP
rng = np.random.default_rng(SEED)
starts = rng.integers(max(0, f.lb_carry - CTX), f.Xte.shape[0] - W, size=N)
b = build(f, starts, "test")
ys_m = torch.tensor(f.y_mean, dtype=torch.float32)
ys_s = torch.tensor(f.y_scale, dtype=torch.float32)

# measured target, computed directly on the test dispatch so the alignment with
# f.Xte is exact by construction (f.test_index is short by lb_carry rows).
P_te = (f.Yte * f.y_scale + f.y_mean).astype(np.float64)
a_te = response_targets(P_te, h=GAP, W=36)
tgt = np.stack([a_te[s + CTX:s + CTX + GAP] for s in starts]) * SIGN     # (N,G,6)
tgt = np.clip(tgt, 0, None)
tgt = tgt / np.maximum(tgt.sum(-1, keepdims=True), 1e-6)

print(f"{N} test gaps. p = softmax(head logits) vs measured s_i*a_i.\n")
print(f"{'arm':30s}" + "".join(f"{c.split('_')[0][:6]:>9s}" for c in TARGETS)
      + f"{'MAE':>8s}{'CE':>8s}")
print(f"{'MEASURED (target)':30s}" + "".join(f"{v:9.3f}" for v in tgt.mean((0, 1)))
      + f"{'-':>8s}{'-':>8s}")
eq = np.full(6, 1 / 6)
print(f"{'equal split 1/n (plain hardnet)':30s}" + "".join(f"{v:9.3f}" for v in eq)
      + f"{np.abs(eq - tgt).mean():8.4f}"
      + f"{-(tgt * np.log(np.maximum(eq, 1e-8))).sum(-1).mean():8.4f}")

for arm in sys.argv[1:]:
    p = ROOT / "planB" / "results" / f"{arm}.pt"
    if not p.exists():
        print(f"{arm:30s} [missing]"); continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    head_fn, n_disp = HEADS[ck["head"]]
    m_ = BACKBONES[ck.get("backbone", "bilstm")](nfeat, n_disp, hidden=ck["hidden"])
    m_.load_state_dict(ck["state"]); m_.eval()
    ps = []
    for i in range(0, N, 64):
        sl = slice(i, i + 64)
        with torch.no_grad():
            d, _ = m_(torch.from_numpy(b["X"][sl]), torch.from_numpy(b["mask"][sl]), None)
            d = d[:, CTX:CTX + GAP]
            if ck.get("residual", False):
                d = apply_residual(d, torch.from_numpy(b["interp"][sl]))
            ps.append(torch.softmax(d[..., 6:], -1).numpy())
    P = np.concatenate(ps, 0).astype(np.float64)
    mae = np.abs(P - tgt).mean()
    ce = -(tgt * np.log(np.maximum(P, 1e-8))).sum(-1).mean()
    print(f"{arm:30s}" + "".join(f"{v:9.3f}" for v in P.mean((0, 1)))
          + f"{mae:8.4f}{ce:8.4f}")
