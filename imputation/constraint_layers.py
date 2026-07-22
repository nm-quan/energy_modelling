"""In-graph (differentiable) constraint enforcement for the training loop, selected by
--constraint-mode. All operate on MW tensors and are differentiable, so the model
trains INSIDE the constraints (contrast posthoc, which trains on a soft-balance loss
and projects only at eval).

THE CONSISTENT SCORING RULE (train == val == test == inference):
every mode is scored as  Π(F(x))  where
  F(x) = the mode's DEPLOYED FORWARD: the raw fill for posthoc/unrolled, the RAYEN
         ray-shot for rayen_traj (`deployed_fill` below). For unrolled, F needs no
         extra step at scoring time because Π IS the converged version of its
         in-graph operator (same fixed points) -- applying Π directly is consistent.
  Π    = the shared exact posthoc projection (constraints.project_gap / cyclic_project)
         which guarantees balance/ramp/box/SOC identically for every mode.
So the benchmark stays apples-to-apples (same Π last), while each model is evaluated
through the same map it was trained through -- epoch selection and final numbers
cannot mis-rank a mode by scoring it through an operator it never saw in training.

  unrolled    K rounds of the validated cyclic projection (constraints.cyclic_project):
              balance -> ramp -> box -> SOC, unrolled as differentiable ops. Reuses the
              exact-balance core proven feasible on all 186 days.
  rayen_traj  RAYEN ray-shoot generalized from one step to the whole gap trajectory
              (a single differentiable shot toward the constraint boundary).
"""
from __future__ import annotations

import numpy as np
import torch

import constraints as C


def project_in_graph(fill_mw, pL_mw, pR_mw, nd_mw, mode="unrolled", iters=10):
    """fill_mw (B,N,6), pL_mw/pR_mw (B,6), nd_mw (B,N) MW tensors -> feasible (B,N,6),
    differentiable. `posthoc` never calls this (it projects only at eval)."""
    if mode == "unrolled":
        return C.cyclic_project(fill_mw, pL_mw, pR_mw, nd_mw, iters=iters)
    if mode == "rayen_traj":
        return rayen_traj_project(fill_mw, pL_mw, pR_mw, nd_mw)
    raise ValueError(f"no in-graph projection for constraint-mode={mode!r}")


def deployed_fill(fill_mw: np.ndarray, pL: np.ndarray, pR: np.ndarray,
                  nd: np.ndarray, mode: str) -> np.ndarray:
    """Numpy single-window F(x): the transformation the mode's forward pass applies
    before the shared posthoc projection Π. Identity for posthoc AND unrolled (Π is
    the converged in-graph operator, so Π alone reproduces unrolled's train-time
    map); the ray-shot for rayen_traj. Use at every inference site (final eval,
    scenario_eval, stack_gap) so a checkpoint is scored as it was trained."""
    if mode != "rayen_traj":
        return fill_mw
    with torch.no_grad():
        out = rayen_traj_project(
            torch.tensor(np.asarray(fill_mw)[None], dtype=torch.float64),
            torch.tensor(np.asarray(pL)[None], dtype=torch.float64),
            torch.tensor(np.asarray(pR)[None], dtype=torch.float64),
            torch.tensor(np.asarray(nd)[None], dtype=torch.float64))
    return out[0].numpy()


def rayen_traj_project(fill_mw, pL_mw, pR_mw, nd_mw, anchor_iters: int = 40,
                       soc: bool = False, anchor=None, size_aware: bool = False):
    """Differentiable RAYEN ray-shoot generalized from one step to the whole gap.

    RAYEN's move: from a strictly feasible ANCHOR, travel along a direction that is
    tangent to the equality plane, by a fraction of the distance to the nearest
    inequality wall. Here, for the 216-dim coupled gap polytope:

      anchor A  = the smooth interp between the pinned boundaries, projected onto
                  {balance}∩{ramp}∩{box} (a feasible, ~balance-exact origin). It is
                  model-independent, so it carries no gradient -- a fixed anchor.
      dir   r   = (model fill - A), with its SIGN component removed so moving along r
                  PRESERVES balance exactly (A already satisfies it).
      alpha*    = min over every box + two-sided-ramp wall (incl. both seams) of the
                  distance from A to that wall along r, clamped to [0,1] (never
                  overshoot the model's own fill). ONE alpha for the whole trajectory
                  -- the RAYEN single-shot (conservative but exact & differentiable).
      y = A + alpha* * r    -> balance exact, ramp+box satisfied by construction.

    soc=False (default): SOC is left to the posthoc eval projection -- fine whenever
    the shot is followed by Π, but the bare rayen map (itr C3) is NOT, and its
    balance-forced battery fills violate the swing cap on blackout windows. soc=True
    adds the swing constraint as one more RAYEN wall: E(t) = cumsum((eta*chg -
    dis/eta)*DT) is LINEAR in the dispatch, so along y(a) = A + a*r every ordered
    pair (t,t') gives a linear inequality  dE_A(t,t') + a*dE_r(t,t') <= cap_eff,
    i.e. exactly the box/ramp "distance to the nearest wall" form. The anchor is
    then built with the SOC step ON so a=0 is swing-feasible.
    """
    dev, dt = fill_mw.device, fill_mw.dtype
    sign = C._SIGN_T.to(dev, dt); rup = C._RUP_T.to(dev, dt)
    rdn = C._RDN_T.to(dev, dt); cap = C._CAP_T.to(dev, dt)
    B, N, _ = fill_mw.shape
    ss = (sign * sign).sum()

    tt = (torch.arange(1, N + 1, device=dev, dtype=dt) / (N + 1)).view(1, N, 1)
    interp = pL_mw.unsqueeze(1) + tt * (pR_mw - pL_mw).unsqueeze(1)
    # feasible anchor via the cyclic projection (its box step keeps A >= 0; a raw
    # balance snap would push channels negative). Anchor balance is ~1e-3 MW here;
    # the shot inherits it, and the posthoc EVAL projection makes the scored output
    # machine-exact -- so in-graph balance stays negligible without any negatives.
    # soc=True needs a longer anchor: the SOC damping is the LAST step of each POCS
    # cycle and un-does a little balance, so balance+SOC settle jointly only after
    # ~120 cycles (40: bal max 1.5 MW on blackout; 120: exact to 1e-3). The anchor
    # carries no model gradient either way, so this is a constant-cost knob.
    # `anchor` overrides the interp-based construction with an externally supplied
    # feasible (B,N,6) trajectory -- needed for whole-day counterfactuals, where the
    # shifted nd jumps ~2000 MW in one step at the free-window edges and the POCS
    # anchor can stall from a flat interp start; the Option-A projected reference
    # (scenario_eval.build_shift_scenario's P_ref) is the intended anchor there.
    # It must satisfy balance/ramp(seams to pL,pR)/box(/SOC) at a=0.
    if anchor is not None:
        A = anchor.to(dev, dt)
    else:
        A = C.cyclic_project(interp, pL_mw, pR_mw, nd_mw,
                             iters=max(anchor_iters, 120) if soc else anchor_iters,
                             soc=soc, size_aware=size_aware)

    r = fill_mw - A
    if size_aware:
        # size-aware tangent removal: absorb the balance component of r in
        # proportion to the anchor's channel levels (w from A: fixed, gradient-
        # free) instead of equally -- Σ sign·(r - a·w·sign) = 0 with a = sign·r/Σw.
        w = A.abs() + 1.0                                        # (B,N,6)
        r = r - ((r * sign).sum(-1) / w.sum(-1).clamp_min(1e-6)).unsqueeze(-1) * (w * sign)
    else:
        r = r - ((r * sign).sum(-1) / ss).unsqueeze(-1) * sign    # tangent to balance plane

    big = torch.full_like(A, 1e9)
    a_box_hi = torch.where(r > 1e-9, (cap - A) / r.clamp_min(1e-9), big).amin((1, 2))
    a_box_lo = torch.where(r < -1e-9, (0.0 - A) / r.clamp_max(-1e-9), big).amin((1, 2))
    zb = torch.zeros_like(pL_mw).unsqueeze(1)                     # r=0 at the pinned boundaries
    dA = torch.diff(torch.cat([pL_mw.unsqueeze(1), A, pR_mw.unsqueeze(1)], 1), dim=1)
    dr = torch.diff(torch.cat([zb, r, zb], 1), dim=1)             # step-diffs incl. both seams
    big_r = torch.full_like(dA, 1e9)                              # (B,N+1,6), matches dr
    a_rup = torch.where(dr > 1e-9, (rup - dA) / dr.clamp_min(1e-9), big_r).amin((1, 2))
    a_rdn = torch.where(dr < -1e-9, (-rdn - dA) / dr.clamp_max(-1e-9), big_r).amin((1, 2))
    alpha = torch.stack([a_box_hi, a_box_lo, a_rup, a_rdn], 0).amin(0).clamp(0.0, 1.0)
    if soc:
        # SOC swing wall. alpha <= a_box keeps the whole segment inside the box, so
        # the swing clamps are no-ops and E is exactly linear in alpha there. Same
        # cap_eff (200 MWh headroom) as cyclic_project, so the anchor's swing
        # satisfies every pair at a=0 and (cap_eff - dEA) >= 0.
        cap_eff = C.BATT_CAP_MWH - 2.0 * 100.0
        eA = (A[..., C.CHG_IDX] * C.ETA - A[..., C.DIS_IDX] / C.ETA) * C.DT
        er = (r[..., C.CHG_IDX] * C.ETA - r[..., C.DIS_IDX] / C.ETA) * C.DT
        EA = torch.cat([torch.zeros_like(eA[:, :1]), eA.cumsum(1)], 1)   # (B,N+1), E(0)=0
        Er = torch.cat([torch.zeros_like(er[:, :1]), er.cumsum(1)], 1)
        a_soc = []
        for i in range(0, B, 32):                          # chunk the (b,N+1,N+1) pair grids
            dEA = EA[i:i + 32].unsqueeze(2) - EA[i:i + 32].unsqueeze(1)
            dEr = Er[i:i + 32].unsqueeze(2) - Er[i:i + 32].unsqueeze(1)
            big_s = torch.full_like(dEA, 1e9)
            a_soc.append(torch.where(dEr > 1e-9, (cap_eff - dEA) / dEr.clamp_min(1e-9),
                                     big_s).amin((1, 2)))
        alpha = torch.minimum(alpha, torch.cat(a_soc).clamp(0.0, 1.0))
    return A + alpha.view(B, 1, 1) * r
