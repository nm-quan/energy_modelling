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

import constraints as C


def project_in_graph(fill_mw, pL_mw, pR_mw, nd_mw, mode="unrolled", iters=10):
    """fill_mw (B,N,6), pL_mw/pR_mw (B,6), nd_mw (B,N) MW tensors -> feasible (B,N,6),
    differentiable. `posthoc` never calls this (it projects only at eval)."""
    if mode == "unrolled":
        return C.cyclic_project(fill_mw, pL_mw, pR_mw, nd_mw, iters=iters)
    if mode == "rayen_traj":
        return rayen_traj_project(fill_mw, pL_mw, pR_mw, nd_mw)
    raise ValueError(f"no in-graph projection for constraint-mode={mode!r}")


def rayen_traj_project(fill_mw, pL_mw, pR_mw, nd_mw):
    raise NotImplementedError(
        "rayen_traj mode is built in the next increment; use --constraint-mode "
        "posthoc or unrolled for now.")
