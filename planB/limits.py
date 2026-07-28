"""Constraint envelope for planB: p99.9 ramps, empirical-max caps.

RAMPS = 99.9th percentile of the historical 5-minute step change, per channel and
per direction, NOT the all-time max. The max is a single-event envelope -- the
canon coal down-ramp of -1553.6 MW/5min is one unit trip on 2024-02-13 -- which
makes the ramp constraint effectively non-binding. p99.9 tightens it 8.6x and
still admits 99.77% of real steps.

CAPS stay at the empirical max. A percentile cap would forbid output levels the
fleet demonstrably reaches, which is a different and unjustified claim.

WHY p99.9 AND NOT p99. Both were measured by projecting the TRUTH onto each
envelope; what is left is error no model can remove:

    envelope   real steps over   floor MAE   balance shortfall
    max                  0.00%      0.00 MW             0.0 MW
    p99.9                0.23%      0.21 MW            12.8 MW
    p99                  2.42%      2.63 MW           189.3 MW

The accuracy cost of p99 is small (2.63 MW against a ~50 MW scale). The
feasibility cost is not: at p99 the balance target becomes unreachable inside the
tightened box on 0.1% of steps, so "balance exact by construction" -- the whole
point of the constraint layer -- degrades to "exact except where the envelope
forbids it". p99.9 keeps that claim intact for a floor of 0.21 MW.

ONE ENVELOPE, used for training, evaluation and counterfactuals alike. An earlier
plan was max for imputation and a percentile for counterfactuals, on the grounds
that max validates history while a percentile simulates behaviour. The
counterfactual evidence argued against it: at p99 the recursive head put 16% of a
demand rise on coal, where dispatch_study measures the real grid at 43%. p99 was
over-correcting coal in the direction that was already wrong.

    python3 planB/limits.py        # print the envelope + the feasibility check
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "imputation"))
from gap_data import TARGETS                                          # noqa: E402
import constraints as C                                               # noqa: E402

TABLE = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/table.parquet"
PCT = 99.9
CAP = C.CAP.copy()                       # empirical max, unchanged


def ramp_percentile(pct: float = PCT):
    """(up, dn) MW per 5-min step at the pct-th percentile, TARGETS order."""
    t = pd.read_parquet(TABLE)
    d = t[TARGETS].diff().dropna()
    up = np.array([np.percentile(d[c].values, pct) for c in TARGETS])
    dn = np.array([np.percentile(-d[c].values, pct) for c in TARGETS])
    return up, dn


R_UP, R_DN = ramp_percentile()
_RUP_T = torch.tensor(R_UP, dtype=torch.float32)
_RDN_T = torch.tensor(R_DN, dtype=torch.float32)
_CAP_T = torch.tensor(CAP, dtype=torch.float32)
_SIGN_T = C._SIGN_T


def limits(dev, dt):
    return (_SIGN_T.to(dev, dt), _RUP_T.to(dev, dt),
            _RDN_T.to(dev, dt), _CAP_T.to(dev, dt))


if __name__ == "__main__":
    import gap_data as GD
    GD.NPZ = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
    from gap_data import load_flats, sample_recon_windows, SIGN

    print(f"planB envelope: ramps at p{PCT:g}, caps at the empirical max\n")
    print(f"  {'channel':22s}{'up pct':>9s}{'up max':>9s}{'':4s}{'dn pct':>9s}{'dn max':>9s}"
          f"{'':4s}{'cap':>9s}")
    for i, c in enumerate(TARGETS):
        print(f"  {c:22s}{R_UP[i]:9.1f}{C.R_UP[i]:9.1f}    {R_DN[i]:9.1f}{C.R_DN[i]:9.1f}"
              f"    {CAP[i]:9.1f}")
    print(f"\n  tightening factor (max / pct): up "
          f"{np.round(C.R_UP / R_UP, 1)}\n{'':35s}dn {np.round(C.R_DN / R_DN, 1)}")

    # ---- does a feasible path still exist between every pinned pair? ----
    f = load_flats()
    gws = sample_recon_windows(f, "test", n=400, context=48, gap=36, seed=123)
    N = 36
    bad_bridge = 0
    worst_slack = np.full(6, np.inf)
    for gw in gws:
        need_up = gw.pR_mw - gw.pL_mw
        slack_up = (N + 1) * R_UP - need_up
        slack_dn = (N + 1) * R_DN + need_up
        worst_slack = np.minimum(worst_slack, np.minimum(slack_up, slack_dn))
        if (slack_up < 0).any() or (slack_dn < 0).any():
            bad_bridge += 1
    print(f"\n  bridge feasibility over {len(gws)} test 3h gaps "
          f"(|pR - pL| <= 37 * ramp, per channel):")
    print(f"    windows with NO feasible path: {bad_bridge}/{len(gws)}")
    print(f"    tightest remaining slack per channel (MW):")
    for i, c in enumerate(TARGETS):
        print(f"      {c:22s}{worst_slack[i]:10.1f}")

    # ---- how often does the ACTUAL data exceed this envelope? ----
    t = pd.read_parquet(TABLE)
    d = t[TARGETS].diff().dropna().values
    over = (d > R_UP) | (d < -R_DN)
    print(f"\n  the actual historical dispatch exceeds this envelope on "
          f"{100*over.mean():.2f}% of channel-steps")
    print(f"    (expected: a p{PCT:g} envelope is exceeded ~{2*(100-PCT):.0f}% of the "
          f"time by construction, counting both directions)")
