"""In-graph (differentiable) constraint enforcement for the training loop, selected by
--constraint-mode. All operate on MW tensors and are differentiable, so the model
trains INSIDE the constraints (contrast posthoc, which trains on a soft-balance loss
and projects only at eval). Every mode is SCORED by the same posthoc eval projection
(constraints.project_gap), so the benchmark is apples-to-apples.

  unrolled    K rounds of the validated cyclic projection (constraints.cyclic_project):
              balance -> ramp -> box -> SOC, unrolled as differentiable ops. Reuses the
              exact-balance core proven feasible on all 186 days.
  rayen_traj  RAYEN ray-shoot generalized from one step to the whole gap trajectory
              (a single differentiable shot toward the constraint boundary).
"""
from __future__ import annotations

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


def rayen_traj_project(fill_mw, pL_mw, pR_mw, nd_mw, anchor_iters: int = 40):
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

    SOC is non-binding on all data and is guaranteed by the posthoc eval projection.
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
    A = C.cyclic_project(interp, pL_mw, pR_mw, nd_mw, iters=anchor_iters, soc=False)

    r = fill_mw - A
    r = r - ((r * sign).sum(-1) / ss).unsqueeze(-1) * sign        # tangent to balance plane

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
    return A + alpha.view(B, 1, 1) * r
