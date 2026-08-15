"""How much of the dispatch answer is the head, and how much is the network?

The hardnet arms are trained on the POST-head output, so the raw F is free to sit
anywhere on the projection's fibre. To size the network's actual contribution,
project the interp skeleton with the network residual set to ZERO and compare:

    interp                 no head, no network      (the standing reference)
    hardnet_head(interp)   head only, no network    <- the hard-coded floor
    <arm>                  head + network

Anything an arm gains over row 2 is the network. Anything it does not is the head.
Curtailment is reported alongside because it never touches a head.
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
from gap_data import load_flats, TARGETS, TARGET_FEAT_IDX              # noqa: E402
from nets import build_delta                                           # noqa: E402
from heads import HEADS, hardnet_head                                  # noqa: E402
from fit import build, curt_activation, CURT_COLS, CTX, GAP, BACKBONES  # noqa: E402

CKPT = ROOT / "colab" / "planB_results"
N, SEED = 400, 123
SHORT = {"coal_brown": "coal", "gas_ocgt": "ocgt", "gas_steam": "steam",
         "hydro": "hydro", "battery_charging": "bat_chg",
         "battery_discharging": "bat_dis"}

f = load_flats()
nfeat = len(f.feat_cols)
W = 2 * CTX + GAP
rng = np.random.default_rng(SEED)
starts = rng.integers(max(0, f.lb_carry - CTX), f.Xte.shape[0] - W, size=N)
b = build(f, starts, "test")
ys_m = torch.tensor(f.y_mean, dtype=torch.float32)
ys_s = torch.tensor(f.y_scale, dtype=torch.float32)
c_m = torch.tensor(b["c_mean"]); c_s = torch.tensor(b["c_scale"])
pL, pR, nd = b["pL"], b["pR"], b["nd"]
truth = b["Y"] * f.y_scale + f.y_mean
ctruth = b["C"]

tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
interp = pL[:, None] + tt * (pR - pL)[:, None]

rows = []


def add(name, P, C=None):
    mae = np.abs(P - truth).mean((0, 1))
    rows.append((name, mae, np.abs(P - truth).mean(),
                 np.abs(C - ctruth).mean() if C is not None else np.nan))


add("interp (no head, no net)", interp)
Pi = hardnet_head(torch.tensor(interp), torch.tensor(pL), torch.tensor(pR),
                  torch.tensor(nd))[0:].numpy()
add("hardnet_head(interp)  <- HEAD ONLY", Pi)

for arm in ("hardnet", "brits", "saits"):
    p = CKPT / f"{arm}.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    head_fn, n_disp = HEADS[ck["head"]]
    bb = ck.get("backbone", "brits" if arm == "brits" else "bilstm")
    m = (BACKBONES[bb](nfeat, n_disp, hidden=ck["hidden"]) if bb == "bilstm"
         else BACKBONES[bb](nfeat, TARGET_FEAT_IDX, CURT_COLS, n_disp, hidden=ck["hidden"]))
    m.load_state_dict(ck["state"]); m.eval()
    Ps, Cs = [], []
    for i in range(0, N, 64):
        sl = slice(i, i + 64)
        x = torch.from_numpy(b["X"][sl]); mk = torch.from_numpy(b["mask"][sl])
        with torch.no_grad():
            dl = build_delta(mk, nfeat) if bb == "brits" else None
            d_raw, c_raw = m(x, mk, dl)
            d_raw = d_raw[:, CTX:CTX + GAP]
            Cs.append(curt_activation(c_raw[:, CTX:CTX + GAP], c_m, c_s).numpy())
            if ck.get("residual", False):
                d_raw = d_raw + torch.from_numpy(b["interp"][sl])
            F_ = d_raw[..., :6] * ys_s + ys_m
            Ps.append(head_fn(F_, torch.from_numpy(pL[sl]), torch.from_numpy(pR[sl]),
                              torch.from_numpy(nd[sl])).numpy())
    add(f"{arm} ({bb}) = head + net", np.concatenate(Ps, 0).astype(np.float64),
        np.concatenate(Cs, 0).astype(np.float64))

ch = [SHORT[t] for t in TARGETS]
print(f"{'':38s} " + "".join(f"{c:>9s}" for c in ch) + f"{'disp agg':>10s}{'curt agg':>10s}")
print("-" * (38 + 9 * len(ch) + 20))
for name, mae, agg, cagg in rows:
    print(f"{name:38s} " + "".join(f"{v:9.1f}" for v in mae) +
          f"{agg:10.1f}" + (f"{cagg:10.1f}" if np.isfinite(cagg) else f"{'—':>10s}"))
