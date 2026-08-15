"""Is the backbone itself constraint-following, or is the head doing all of it?

For every 6-output arm, take the RAW dispatch F (interp skeleton + network
residual, in MW) BEFORE hardnet_head, and audit it against the same p99.9
envelope the head enforces:

    balance   |SIGN.F_t - nd_t|
    box       max(-F, 0) and max(F - CAP, 0)
    ramp      overshoot of R_UP / R_DN across pL -> F -> pR
    move      MW the projection then has to shift, from hardnet's own debug hook

hardnet is the identity on an already-feasible input (heads.py self-test), so
`move` is a direct read of how far off the network was.
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
from heads import HEADS, hardnet_head                                  # noqa: E402
from limits import R_UP, R_DN, CAP                                     # noqa: E402
from fit import build, CURT_COLS, CTX, GAP, BACKBONES                  # noqa: E402

CKPT = ROOT / "colab" / "planB_results"          # the TRAINED checkpoints
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


def audit(F, name, move=None):
    """F (n,G,6) MW. Returns a dict of violation magnitudes."""
    bal = np.abs(F @ SIGN - nd)                                   # (n,G)
    neg = np.maximum(-F, 0.0)
    cap = np.maximum(F - CAP, 0.0)
    full = np.concatenate([pL[:, None], F, pR[:, None]], 1)       # (n,G+2,6)
    d = np.diff(full, axis=1)
    ramp = np.maximum(d - R_UP, 0.0) + np.maximum(-d - R_DN, 0.0)
    r = {"name": name,
         "bal_mean": bal.mean(), "bal_p99": np.percentile(bal, 99), "bal_max": bal.max(),
         "neg_mean": neg.sum(-1).mean(), "neg_max": neg.max(),
         "cap_max": cap.max(),
         "ramp_mean": ramp.sum(-1).mean(), "ramp_max": ramp.max(),
         "ramp_pct": 100 * (ramp > 0.1).any(-1).mean(),
         "bal_pct": 100 * (bal > 1.0).mean()}
    r["move_mean"] = move.mean() if move is not None else np.nan
    r["move_max"] = move.max() if move is not None else np.nan
    r["move_id"] = 100 * (move < 1.0).mean() if move is not None else np.nan
    return r


rows = []

# references -------------------------------------------------------------
tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
interp = pL[:, None] + tt * (pR - pL)[:, None]
_, dbg = hardnet_head(torch.tensor(interp), torch.tensor(pL), torch.tensor(pR),
                      torch.tensor(nd), return_debug=True)
rows.append(audit(interp, "interp (no network)", dbg["correction"].numpy()))
_, dbg = hardnet_head(torch.tensor(truth), torch.tensor(pL), torch.tensor(pR),
                      torch.tensor(nd), return_debug=True)
rows.append(audit(truth, "TRUTH", dbg["correction"].numpy()))

# arms -------------------------------------------------------------------
for arm in ("unconstrained", "hardnet", "brits", "saits"):
    p = CKPT / f"{arm}.pt"
    if not p.exists():
        print(f"[skip] {arm}"); continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    head_fn, n_disp = HEADS[ck["head"]]
    bb = ck.get("backbone", "brits" if arm == "brits" else "bilstm")
    m = (BACKBONES[bb](nfeat, n_disp, hidden=ck["hidden"]) if bb == "bilstm"
         else BACKBONES[bb](nfeat, TARGET_FEAT_IDX, CURT_COLS, n_disp, hidden=ck["hidden"]))
    m.load_state_dict(ck["state"]); m.eval()
    raws = []
    for i in range(0, N, 64):
        sl = slice(i, i + 64)
        x = torch.from_numpy(b["X"][sl]); mk = torch.from_numpy(b["mask"][sl])
        with torch.no_grad():
            dl = build_delta(mk, nfeat) if bb == "brits" else None
            d_raw, _ = m(x, mk, dl)
            d_raw = d_raw[:, CTX:CTX + GAP]
            if ck.get("residual", False):
                d_raw = d_raw + torch.from_numpy(b["interp"][sl])
            raws.append((d_raw[..., :6] * ys_s + ys_m).numpy())
    F = np.concatenate(raws, 0).astype(np.float64)
    _, dbg = hardnet_head(torch.tensor(F), torch.tensor(pL), torch.tensor(pR),
                          torch.tensor(nd), return_debug=True)
    rows.append(audit(F, f"{arm} ({bb}) RAW", dbg["correction"].numpy()))

hdr = (f"{'':28s} {'balance MW':>26s}   {'box MW':>16s}   {'ramp MW':>22s}   "
       f"{'head move MW/step':>26s}")
sub = (f"{'raw output, pre-head':28s} {'mean':>8s}{'p99':>9s}{'max':>9s}  "
       f"{'%>1MW':>7s} {'neg max':>8s}{'cap max':>8s}  {'mean':>8s}{'max':>8s}"
       f"{'% win':>7s}  {'mean':>9s}{'max':>10s}{'% ident':>8s}")
print(hdr); print(sub); print("-" * len(sub))
for r in rows:
    print(f"{r['name']:28s} {r['bal_mean']:8.1f}{r['bal_p99']:9.1f}{r['bal_max']:9.1f}  "
          f"{r['bal_pct']:6.1f}% {r['neg_max']:8.1f}{r['cap_max']:8.1f}  "
          f"{r['ramp_mean']:8.2f}{r['ramp_max']:8.1f}{r['ramp_pct']:6.1f}%  "
          f"{r['move_mean']:9.1f}{r['move_max']:10.1f}{r['move_id']:7.1f}%")
