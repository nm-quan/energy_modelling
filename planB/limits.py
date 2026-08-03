"""Constraint envelope for planB: empirical-max ramps, with two channels tightened.

WHY NOT A PERCENTILE. A percentile envelope is, by construction, violated by the
tail it excluded -- measured over 3,004,986 channel-steps:

    envelope   cells over   rate      coal down   gas_steam down
    p99            59,943   1.995%       -123.3            -28.9
    p99.9           5,990   0.199%       -180.2            -54.8
    p99.99            610   0.020%       -275.5           -107.6
    p99.999            72   0.002%       -560.6           -135.0
    max                 0   0.000%      -1553.6           -499.9

Only the max admits every recorded step. "The envelope should match the data" and
"the envelope should be tighter than the max" are mutually exclusive requirements,
and matching the data wins: you cannot hold a model to a standard the grid itself
does not meet, and a figure that reports 50 ramp violations coming from untouched
historical dispatch is unreadable.

WHAT THIS GIVES UP, stated plainly. coal_brown down = -1553.6 MW/5min is a single
unit trip on 2024-02-13, 8.6x its own p99.9, so the coal down-ramp is effectively
non-binding. gas_steam down is 9.1x its p99.9 for the same reason.

WHY THAT IS NOW ACCEPTABLE. The tightening was introduced because coal absorbed
82% of the counterfactual response -- but that was the INCUMBENT size_aware head,
whose alpha* collapsed to zero in 30/40 windows, so a hand-written weight rule was
choosing the allocation, not the model. The current projection head under-uses
coal (38% against a measured 44.1%). The symptom the percentile was treating no
longer exists.

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
PCT = None      # default for every channel: the empirical max

# Per-channel overrides. Everything not listed keeps the max, so the historical
# data stays feasible on those channels by construction.
#
#   coal_brown  down = -1553.6 MW/5min is a single unit trip (2024-02-13), 10.9x
#               its own p99.5, which leaves the coal down-ramp non-binding.
#   gas_steam   same story on the down side (12.2x), and on the UP side the max of
#               82.5 let the model ramp it to 12% of the counterfactual response
#               against a measured 3.7%. p99.5 tightens the up-ramp 2.6x.
#
# Cost, stated: these two channels' own history exceeds their limit ~1.0% of the
# time. Every other channel remains at 0.00%.
PCT_OVERRIDE = {"coal_brown": 99.5, "gas_steam": 99.5}
CAP = C.CAP.copy()                       # empirical max, unchanged


def ramp_percentile(pct=PCT, override=None):
    """(up, dn) MW per 5-min step, TARGETS order.

    pct=None means the empirical max; `override` maps a channel name to its own
    percentile and wins over pct for that channel."""
    override = PCT_OVERRIDE if override is None else override
    up, dn = C.R_UP.copy(), C.R_DN.copy()
    if pct is None and not override:
        return up, dn
    t = pd.read_parquet(TABLE)
    d = t[TARGETS].diff().dropna()
    for i, c in enumerate(TARGETS):
        q = override.get(c, pct)
        if q is None:
            continue
        up[i] = np.percentile(d[c].values, q)
        dn[i] = np.percentile(-d[c].values, q)
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

    NAME = "empirical max" if PCT is None else f"p{PCT:g}"
    ov = ", ".join(f"{c} p{q:g}" for c, q in PCT_OVERRIDE.items()) or "none"
    print(f"planB envelope: ramps at the {NAME}; overrides: {ov}; "
          f"caps at the empirical max\n")
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
    per = ((d > R_UP) | (d < -R_DN)).mean(0)
    print("    per channel: " + ", ".join(
        f"{c.split('_')[0]} {100*per[i]:.2f}%" for i, c in enumerate(TARGETS)))
    if PCT is None and not PCT_OVERRIDE:
        print("    (the max envelope admits every recorded step by construction)")
    elif PCT is None:
        print("    (only the overridden channels can exceed; the rest are at the max)")
    else:
        print(f"    (expected: a p{PCT:g} envelope is exceeded ~{2*(100-PCT):.0f}% of "
              f"the time by construction, counting both directions)")
