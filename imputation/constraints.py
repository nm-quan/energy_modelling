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
import torch

from gap_data import SIGN, TARGETS
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from check_caps import RAMPS, CAPS, BATT_CAP_MWH          # noqa: E402

R_UP = np.array([RAMPS[t][1] for t in TARGETS], dtype=np.float64)         # (6,)
R_DN = np.array([abs(RAMPS[t][0]) for t in TARGETS], dtype=np.float64)    # (6,)
CAP = np.array([CAPS[t] for t in TARGETS], dtype=np.float64)             # (6,)

# ---- SOC (battery energy reservoir) constants, shared with scenario_eval ----
ETA = float(np.sqrt(0.834))      # one-side round-trip efficiency
DT = 5.0 / 60.0                  # hours per 5-min step
# battery channel indices in TARGETS order (charging=load, discharging=source)
CHG_IDX, DIS_IDX = 4, 5


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


# =====================================================================================
# Torch cyclic projection over {balance} ∩ {ramp tube} ∩ {box} ∩ {SOC}
# -------------------------------------------------------------------------------------
# One `cyclic_project` serves BOTH constraint modes:
#   posthoc   run it under no_grad at eval (numpy wrapper project_gap below)
#   unrolled  call it INSIDE the model forward with a small `iters` so gradients flow
#             (every op is a differentiable clamp/linear move -- no in-place writes).
# Cycling the four projections converges to a point in their intersection whenever it
# is non-empty; the feasibility_certificate proves it is non-empty on all test days,
# so balance reaches ~0 MW while ramp/box/SOC stay satisfied (validated).
# =====================================================================================
_SIGN_T = torch.tensor(SIGN, dtype=torch.float32)
_RUP_T = torch.tensor(R_UP, dtype=torch.float32)
_RDN_T = torch.tensor(R_DN, dtype=torch.float32)
_CAP_T = torch.tensor(CAP, dtype=torch.float32)


def _balance_project(P, nd, sign, free_mask):
    """Move the free channels along SIGN so Σ SIGN·P == nd at every step (exact,
    per step, linear). P (B,N,6), nd (B,N), sign/free_mask (6,)."""
    resid = nd - (P * sign).sum(-1)                       # (B,N)
    s = sign * free_mask                                  # only free channels move
    return P + (resid / (s * s).sum()).unsqueeze(-1) * s


def _ramp_project(P, pL, pR, rup, rdn):
    """Forward then backward ramp clamp with boundaries pinned, built functionally
    (no in-place writes -> autograd-safe). One fwd+bwd sweep; the outer cycle repeats
    it, so every consecutive pair incl. both seams converges to ramp-feasible."""
    N = P.shape[1]
    prev = pL                                             # forward from pL
    fwd = []
    for t in range(N):
        x = torch.maximum(torch.minimum(P[:, t], prev + rup), prev - rdn)
        fwd.append(x); prev = x
    nxt = pR                                              # backward from pR
    back = [None] * N
    for t in range(N - 1, -1, -1):
        x = torch.maximum(torch.minimum(fwd[t], nxt + rdn), nxt - rup)
        back[t] = x; nxt = x
    return torch.stack(back, dim=1)                       # (B,N,6)


def _soc_project(P, cap_eff):
    """Damp battery throughput so the cumulative-energy SWING fits the pack. With
    SOC unknown at t=0, feasibility <=> swing(cum energy) <= capacity (check_caps).
    Scaling both battery channels by cap_eff/swing scales the swing to cap_eff; a
    no-op (scale=1) whenever SOC is non-binding, which is every test day."""
    chg = P[..., CHG_IDX].clamp_min(0.0)
    dis = P[..., DIS_IDX].clamp_min(0.0)
    dE = (chg * ETA - dis / ETA) * DT                     # (B,N) MWh per step
    E = torch.cumsum(dE, dim=1)
    Efull = torch.cat([torch.zeros_like(E[:, :1]), E], dim=1)   # include start (0)
    swing = Efull.max(dim=1).values - Efull.min(dim=1).values   # (B,)
    scale = torch.clamp(cap_eff / (swing + 1e-6), max=1.0).view(-1, 1, 1)
    batt = P[..., CHG_IDX:DIS_IDX + 1] * scale
    return torch.cat([P[..., :CHG_IDX], batt], dim=-1)


def cyclic_project(P, pL, pR, nd, iters=40, soc=True, free=None, margin_mwh=100.0):
    """Project raw dispatch P (B,N,6) MW onto {balance}∩{ramp}∩{box}∩{SOC}, with the
    boundaries pL,pR (B,6) pinned and Σ SIGN·P snapped to nd (B,N). Differentiable."""
    dev, dt = P.device, P.dtype                          # match input dtype (f64 posthoc, f32 train)
    sign = _SIGN_T.to(dev, dt); rup = _RUP_T.to(dev, dt)
    rdn = _RDN_T.to(dev, dt); cap = _CAP_T.to(dev, dt)
    free_mask = torch.ones(6, device=dev, dtype=dt)
    if free is not None:
        free_mask = torch.zeros(6, device=dev, dtype=dt); free_mask[list(free)] = 1.0
    cap_eff = BATT_CAP_MWH - 2.0 * margin_mwh
    for _ in range(iters):
        P = _balance_project(P, nd, sign, free_mask)
        P = _ramp_project(P, pL, pR, rup, rdn)
        P = torch.minimum(P.clamp_min(0.0), cap)
        if soc:
            P = _soc_project(P, cap_eff)
    return P


def project_gap(fill_mw: np.ndarray, pL: np.ndarray, pR: np.ndarray,
                nd_mw: np.ndarray, free=None, iters: int = 40, soc: bool = True):
    """POSTHOC mode: numpy wrapper over cyclic_project (no grad). Projects a raw
    (N,6) fill to a hard-feasible trajectory with EXACT balance + SOC. Returns
    (proj_mw (N,6), balance_resid_mw (N,)); residual is ~0 where a feasible dispatch
    exists (all test days) and non-zero only on genuinely ramp-limited steps."""
    with torch.no_grad():
        P = cyclic_project(
            torch.tensor(np.asarray(fill_mw)[None], dtype=torch.float64),
            torch.tensor(np.asarray(pL)[None], dtype=torch.float64),
            torch.tensor(np.asarray(pR)[None], dtype=torch.float64),
            torch.tensor(np.asarray(nd_mw)[None], dtype=torch.float64),
            iters=iters, soc=soc, free=free)[0].numpy()
    return P, np.abs(nd_mw - P @ SIGN)


def feasibility_certificate(pL: np.ndarray, pR: np.ndarray, nd_mw: np.ndarray,
                            iters: int = 250, soc: bool = True):
    """Does a feasible dispatch EXIST for this net_demand (concern #2)? Project a
    neutral start onto the intersection; a max residual ~0 certifies the scenario is
    within constraints, so any model violation is the model's fault, not the data's.
    Returns (max_balance_resid_mw, feasible_dispatch (N,6))."""
    N = len(nd_mw)
    P0 = np.tile((np.asarray(pL) + np.asarray(pR)) / 2.0, (N, 1))
    P, resid = project_gap(P0, pL, pR, nd_mw, iters=iters, soc=soc)
    return float(resid.max()), P


def _soc_swing_mwh(P):
    dE = (np.clip(P[:, CHG_IDX], 0, None) * ETA - np.clip(P[:, DIS_IDX], 0, None) / ETA) * DT
    cum = np.concatenate([[0.0], np.cumsum(dE)])
    return cum.max() - cum.min()


def _ramp_overshoot_mw(full):
    """Max MW by which any consecutive pair (incl. both seams) breaks its ramp limit.
    full = [pL, P.., pR] stacked on the time axis. ~0 => ramp-feasible (report the
    MAGNITUDE, not a count at an arbitrary tolerance -- convergence leaves ~1e-6 MW)."""
    dd = np.diff(full, axis=-2)
    return float((np.maximum(dd - R_UP, 0) + np.maximum(-dd - R_DN, 0)).max())


if __name__ == "__main__":
    # 1) sanity on ONE real day: project a noisy fill -> ramp-clean on every pair
    #    incl. both seams (2:05 fix), box-clean, SOC-feasible, balance EXACT.
    from gap_data import load_flats, test_gap_windows
    f = load_flats()
    rng = np.random.default_rng(0)
    gws = test_gap_windows(f, context=48)
    gw = gws[100]
    pL, pR, nd = gw.pL_mw, gw.pR_mw, gw.nd_mw
    P, resid = project_gap(gw.truth_mw + rng.normal(0, 120, gw.truth_mw.shape), pL, pR, nd)
    ramp_over = _ramp_overshoot_mw(np.vstack([pL, P, pR]))   # incl. both seams
    print(f"[one day] ramp overshoot {ramp_over:.2e} MW   n_neg {int((P < -1e-6).sum())}   "
          f"box_over {np.maximum(P - CAP, 0).max():.2e} MW   soc_swing {_soc_swing_mwh(P):.0f}/{BATT_CAP_MWH:.0f} MWh   "
          f"balance resid max {resid.max():.2e} MW")

    # 2) feasibility certificate (concern #2): does a feasible dispatch EXIST for the
    #    base and +10% net_demand on EVERY test day? Batched (B=186) -> also exercises
    #    the differentiable batched path the unrolled mode uses. max resid ~0 => within
    #    constraints, so any later model violation is the model's fault, not the data's.
    pL = np.stack([gw.pL_mw for gw in gws]); pR = np.stack([gw.pR_mw for gw in gws])
    nd0 = np.stack([gw.nd_mw for gw in gws])
    dem = np.stack([f.col_mw(f.Xte, 7)[gw.gap_idx] for gw in gws])
    N = nd0.shape[1]
    for g in (0.0, 10.0):
        nd_s = nd0 + (g / 100.0) * dem
        P0 = np.repeat(((pL + pR) / 2.0)[:, None, :], N, axis=1)
        with torch.no_grad():
            P = cyclic_project(torch.tensor(P0, dtype=torch.float64),
                               torch.tensor(pL, dtype=torch.float64),
                               torch.tensor(pR, dtype=torch.float64),
                               torch.tensor(nd_s, dtype=torch.float64), iters=120).numpy()
        resid = np.abs(nd_s - (P * SIGN).sum(-1))
        full = np.concatenate([pL[:, None], P, pR[:, None]], axis=1)
        ramp_over = _ramp_overshoot_mw(full)
        soc_bad = sum(_soc_swing_mwh(P[i]) > BATT_CAP_MWH + 1e-6 for i in range(len(gws)))
        verdict = "FEASIBLE" if (resid.max() < 1e-3 and ramp_over < 1e-3 and soc_bad == 0) else "INFEASIBLE"
        print(f"[certificate +{g:g}%] {len(gws)} days: worst balance resid {resid.max():.2e} MW   "
              f"ramp overshoot {ramp_over:.2e} MW   n_neg {int((P < -1e-6).sum())}   "
              f"soc-infeasible days {int(soc_bad)}   => {verdict}")
