"""Does a weighted projection fix the gas-steam over-allocation?

The equal-MW split comes from minimising ||P - F||^2. Minimise the WEIGHTED norm
instead:

    P* = argmin sum_i w_i (P_i - F_i)^2   s.t.  s.P = nd,  lo <= P <= hi

KKT gives the same shape with one scalar lambda:

    P = clip(F + lambda * s / w, lo, hi)

and g(lambda) = s.clip(...) is still monotone (slope sum_{free} 1/w_i > 0), so the
same bisection works. The differentiable active-set formula divides by
sum_{i in A} 1/w_i instead of |A|.

Three weightings, run at INFERENCE on hardnet_2025 (trained through the equal
head, so this is indicative of the allocation rule only, not a trained result):

    equal    w = 1                 the current head
    z-space  w = 1/sigma^2         min-norm in the units the LOSS uses
    sigma    w = 1/sigma           halfway
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
from gap_data import load_flats, TARGETS, SIGN                         # noqa: E402
import heads                                                           # noqa: E402
from heads import _box                                                 # noqa: E402
from limits import limits as _limits                                   # noqa: E402

SIG = load_flats().y_scale.astype(np.float64)
EMP = {"coal_brown": 43.4, "hydro": 29.3, "gas_ocgt": 9.7, "gas_steam": 3.7,
       "battery_discharging": 7.4, "battery_charging": 6.6}


def make_head(w_np):
    """w_np (6,) positive weights -> a drop-in replacement for hardnet_head."""
    W = torch.tensor(w_np, dtype=torch.float64)

    def proj(F, lo, hi, nd, sign, iters=60):
        w = W.to(F.device, F.dtype)
        step = sign / w                                   # direction of travel
        with torch.no_grad():
            R = (torch.maximum((hi - F).abs(), (F - lo).abs()).amax(-1, keepdim=True)
                 + 1.0) / step.abs().min() + 1.0
            a, b = -R, R
            for _ in range(iters):
                m = 0.5 * (a + b)
                g = (torch.clamp(F + m * step, lo, hi) * sign).sum(-1, keepdim=True)
                low = g < nd.unsqueeze(-1)
                a = torch.where(low, m, a); b = torch.where(low, b, m)
            lam0 = 0.5 * (a + b)
            P0 = torch.clamp(F + lam0 * step, lo, hi)
            free = (P0 > lo + 1e-9) & (P0 < hi - 1e-9)
            denom = (free.to(F.dtype) / w).sum(-1, keepdim=True)
        Pc = torch.clamp(F + lam0 * step, lo, hi)
        fixed = (~free).to(F.dtype) * Pc * sign
        lam = (nd.unsqueeze(-1) - fixed.sum(-1, keepdim=True)
               - (free.to(F.dtype) * F * sign).sum(-1, keepdim=True)) / denom.clamp_min(1e-12)
        lam = torch.where(denom > 0, lam, lam0)
        return torch.clamp(F + lam * step, lo, hi)

    def head(raw, pL, pR, nd, return_debug=False):
        dev, dt = raw.device, raw.dtype
        sign, rup, rdn, cap = _limits(dev, dt)
        B, N, _ = raw.shape
        P_prev, outs = pL, []
        for j in range(N):
            lo, hi = _box(P_prev, pR, N - j, rup, rdn, cap)
            P_t = proj(raw[:, j], lo, hi, nd[:, j], sign)
            outs.append(P_t); P_prev = P_t
        P = torch.stack(outs, 1)
        return (P, {"correction": torch.zeros(B, N)}) if return_debug else P
    return head


WEIGHTS = {"equal  w=1        (current)": np.ones(6),
           "sigma  w=1/sigma": 1.0 / SIG,
           "zspace w=1/sigma^2 (loss units)": 1.0 / SIG ** 2}

if __name__ == "__main__":
    import contextlib, io
    import counterfactual as CF
    CF.OUT = ROOT / "planB" / "results"
    rows = {}
    for name, w in WEIGHTS.items():
        h = make_head(w)
        heads.hardnet_head = h
        CF.HEADS = dict(CF.HEADS); CF.HEADS["hardnet"] = (h, 6)
        sys.argv = ["cf", "--days", "10", "--arms", "hardnet_2025",
                    "--suffix", "_wtmp"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CF.main()
        md = (CF.OUT / "counterfactual_wtmp.md").read_text()
        blk = md.split("## Response, masked")[1].split("## Curtailment")[0]
        vals, tot = {}, None
        for line in blk.splitlines():
            p = [c.strip() for c in line.split("|")]
            if len(p) > 3 and p[1] not in ("channel", "---", ""):
                if "Σ" in p[1]:
                    tot = p[2]
                else:
                    vals[p[1]] = p[2]
        rows[name] = (vals, tot)

    chans = list(rows[list(rows)[0]][0])
    print(f"hardnet_2025, masked mode, 10 days — allocation under three norms\n")
    print(f"{'channel':20s}" + "".join(f"{n:>34s}" for n in rows) + f"{'REAL':>8s}")
    for c in chans:
        emp = [v for k, v in EMP.items() if k.split("_")[0] in c.replace(" ", "_")
               or c.split()[0] in k]
        print(f"{c:20s}" + "".join(f"{rows[n][0][c]:>34s}" for n in rows))
    print(f"{'Σ signed':20s}" + "".join(f"{rows[n][1]:>34s}" for n in rows))
    print("\nREAL GRID: hydro 29.3  coal 43.4  gas steam 3.7  gas OCGT 9.7  "
          "battery in 6.6  battery out 7.4")
