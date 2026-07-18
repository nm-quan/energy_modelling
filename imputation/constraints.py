"""Hard constraints for the gap fill: the two-sided RAMP TUBE + balance + box.

The seam problem (2:05 violations) came from one-sided ramp enforcement: the old
head guaranteed each step within ramp of the PAST only, so the closed-loop fill
drifted and snapped back to the actual 14:00 value. With both boundaries known
(bidirectional imputation) we enforce ramp against BOTH ends at once:

  step t of an N-step gap, known endpoints p_L (10:55) and p_R (14:00):
    forward cone : |P(t) - p_L| <= t * r          (reachable from the left)
    backward cone: |P(t) - p_R| <= (N+1 - t) * r  (able to still reach the right)
    box          : 0 <= P(t) <= cap
  tube(t) = intersection of both cones and the box.

Projecting every fill into its tube makes EVERY consecutive pair -- including the
entry seam (p_L -> P(1)) and the exit seam (P(N) -> p_R) -- ramp-feasible by
construction. A ramp-feasible bridge exists iff |p_R - p_L| <= (N+1) * r per
channel (endpoints not farther apart than the fleet can ramp across the gap).

Balance: after the tube, snap the signed sum onto net_demand by moving along the
flexible-channel directions within the remaining tube room (an anchored rescale,
same family as RayenHeadFixedD). SOC is handled statefully by the caller.

Ramps are asymmetric (r_up != r_dn); the cones use the direction-appropriate rate.
"""
from __future__ import annotations

import numpy as np

from gap_data import SIGN, TARGETS
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from check_caps import RAMPS, CAPS          # noqa: E402

R_UP = np.array([RAMPS[t][1] for t in TARGETS], dtype=np.float64)         # (6,)
R_DN = np.array([abs(RAMPS[t][0]) for t in TARGETS], dtype=np.float64)    # (6,)
CAP = np.array([CAPS[t] for t in TARGETS], dtype=np.float64)             # (6,)


def bridge_feasible(pL: np.ndarray, pR: np.ndarray, N: int) -> np.ndarray:
    """Per-channel: does a ramp-feasible path from pL to pR over N steps exist?
    Up-move needs (N+1)*r_up >= (pR-pL); down-move needs (N+1)*r_dn >= (pL-pR)."""
    up = (N + 1) * R_UP >= (pR - pL) - 1e-6
    dn = (N + 1) * R_DN >= (pL - pR) - 1e-6
    return up & dn


def ramp_tube(pL: np.ndarray, pR: np.ndarray, N: int):
    """(lo, hi) each (N,6): the feasible band per gap step from both boundaries+box.
    t = 1..N. Uses asymmetric ramps: the max you can be ABOVE pL after t steps is
    t*r_up, below is t*r_dn; symmetric logic from the right with (N+1-t)."""
    t = np.arange(1, N + 1)[:, None].astype(np.float64)      # (N,1)
    tb = (N + 1 - t)                                         # steps remaining to pR
    lo_fwd = pL[None, :] - t * R_DN[None, :]
    hi_fwd = pL[None, :] + t * R_UP[None, :]
    lo_bwd = pR[None, :] - tb * R_UP[None, :]                # to still climb to pR
    hi_bwd = pR[None, :] + tb * R_DN[None, :]
    lo = np.maximum.reduce([lo_fwd, lo_bwd, np.zeros_like(lo_fwd)])
    hi = np.minimum.reduce([hi_fwd, hi_bwd, np.broadcast_to(CAP, lo_fwd.shape)])
    return lo, np.maximum(hi, lo)                            # guard lo<=hi (feasible => holds)


def _ramp_pocs(P: np.ndarray, pL: np.ndarray, pR: np.ndarray, iters: int = 8):
    """Alternating forward/backward ramp + box projection with the endpoints
    pinned. Each sweep enforces |P(t)-P(t-1)| within [-r_dn, r_up]; iterating the
    two directions (POCS) converges to a trajectory that is ramp-feasible on EVERY
    consecutive pair incl. the entry seam pL->P(1) and exit seam P(N)->pR. Requires
    bridge_feasible (guaranteed when |pR-pL| <= (N+1)*r)."""
    N = P.shape[0]
    full = np.vstack([pL, P, pR])                           # (N+2,6); rows 0 and N+1 fixed
    for _ in range(iters):
        for t in range(1, N + 1):                           # forward
            full[t] = np.clip(full[t], full[t - 1] - R_DN, full[t - 1] + R_UP)
        for t in range(N, 0, -1):                           # backward
            full[t] = np.clip(full[t], full[t + 1] - R_UP, full[t + 1] + R_DN)
        full[1:N + 1] = np.clip(full[1:N + 1], 0.0, CAP)    # box
    return full[1:N + 1]


def project_gap(fill_mw: np.ndarray, pL: np.ndarray, pR: np.ndarray,
                nd_mw: np.ndarray, free=(0, 1, 3)):
    """Project a raw (N,6) fill to a hard-feasible gap trajectory.

    Priority (from the study: ramps/box win over balance): first snap the signed
    sum onto net_demand per step by moving the `free` channels (default
    hydro/coal/ocgt), then enforce ramp+box exactly via forward/backward POCS with
    the boundaries pinned. The final projection may pull slightly off the balance
    plane where the demand change exceeds one-step ramp capacity -- that residual
    is physical and is returned. Returns (proj_mw (N,6), balance_resid_mw (N,))."""
    free = list(free)
    s = SIGN[free]                                          # +1 for hydro/coal/ocgt
    P = fill_mw.copy()
    resid = nd_mw - P @ SIGN                                # (N,)
    P[:, free] = P[:, free] + (resid / (s @ s))[:, None] * s[None, :]   # onto SIGN.P = nd
    P = _ramp_pocs(np.clip(P, 0.0, CAP), pL, pR)           # hard ramp+box, both seams
    return P, np.abs(nd_mw - P @ SIGN)


if __name__ == "__main__":
    # sanity on a REAL feasible day: project a noisy fill; must be ramp-clean on
    # every pair incl. both seams (the 2:05 fix), box-clean, balance near-exact.
    from gap_data import load_flats, test_gap_windows
    rng = np.random.default_rng(0)
    gw = test_gap_windows(load_flats())[100]
    pL, pR, nd, N = gw.pL_mw, gw.pR_mw, gw.nd_mw, len(gw.gap_idx)
    print("bridge feasible per channel:", bridge_feasible(pL, pR, N))
    P, resid = project_gap(gw.truth_mw + rng.normal(0, 120, (N, 6)), pL, pR, nd)
    d = np.diff(np.vstack([pL, P, pR]), axis=0)             # incl. both seams
    bad = int(((d > R_UP + 1e-6) | (d < -(R_DN + 1e-6))).sum())
    print(f"ramp violations incl. both seams: {bad};  n_neg {int((P < -1e-6).sum())};  "
          f"balance resid max {resid.max():.2f} MW")
