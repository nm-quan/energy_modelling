"""Recursive RAYEN head for the gap trajectory.

WHY. constraint_layers.rayen_traj_project lifts RAYEN from one step to a 36-step
trajectory by keeping ONE global alpha* over all N*6 cells, anchored on a POCS
projection of the interpolation. Measured, that breaks all three of RAYEN's
preconditions and collapses:
  * the anchor lands ON the box face (POCS's last step is the box clip), so
    alpha_lo = 0/r = 0 whenever any cell wants to go below it -> alpha*=0 in
    30/40 windows;
  * one scalar over 216 cells means one blocked cell caps everything;
  * the step size is |r| itself, so alpha* r = -A at the binding cell REGARDLESS
    of |r| -- the head is invariant to how hard the network pushes, and the
    network cannot learn to assert itself.
Consequence: 0.5% of the counterfactual response is attributable to the network,
and the raw fill is 15.7x worse than plain interpolation because the loss never
reached the weights.

THE FIX. Apply RAYEN RECURSIVELY, once per timestep, on the local feasible set.
The key structural fact is that the per-step feasible set is a BOX INTERSECTED
WITH A HYPERPLANE -- the simplest polytope there is -- because every constraint
that touches step t reduces to an interval once P_{t-1} is fixed:

    lo[t,i] = max( 0,      P[t-1,i] - rdn_i,  pR_i - k*rup_i )
    hi[t,i] = min( cap_i,  P[t-1,i] + rup_i,  pR_i + k*rdn_i )      k = N+1-t
    balance:  s . P_t = nd_t
                                          ^^^^^^^^^^^^^^^^^^ backward-reachability
                                          guard: keeps the greedy sweep from
                                          painting itself into a corner where pR
                                          is no longer reachable (constraints.
                                          ramp_tube's cone, currently dead code).

Per step:
  1. ANCHOR   A_t = glide toward pR at constant rate, then the balance snap with
              DIRECTIONAL room weights (w_i = room in the direction that channel
              must move). Exact and in-box whenever nd_t is achievable at all.
  2. MASK     zero the direction on channels sitting at a wall and pointing out
              (lib/models.py::RayenHead's `at_floor`, generalised per step) so a
              legitimately-zero channel cannot zero the step.
  3. DIRECTION balance-tangent over the FREE channels only, then normalised by
              its inf-norm, so magnitude is separated from direction.
  4. STEP     P_t = A_t + sigmoid(s_t) * alpha_max,t * r_hat_t, where alpha_max,t
              is purely geometric and sigmoid(s_t) is a LEARNED per-step scalar
              (what the single-step RayenHead has and the trajectory port lost).

GUARANTEES, per step and hence for the trajectory: balance exact (tangency), box
exact, ramp exact on every consecutive pair INCLUDING both seams (t=N gives
k=1, i.e. lo/hi within one ramp of pR). SOC is not enforced, same as the
incumbent head. Non-emptiness of [lo,hi] is guaranteed by induction from the
backward guard; balance is achievable inside it under the same condition
constraints.feasibility_certificate already verifies.

    python3 imputation/recursive_head.py        # feasibility self-test
"""
from __future__ import annotations

import torch

import constraints as C


def recursive_rayen(raw, pL, pR, nd, return_debug=False):
    """raw (B,N,7) network output -> P (B,N,6) MW, exactly feasible, differentiable.

    raw[..., :6] is the direction, raw[..., 6] the step logit.
    pL, pR (B,6) MW pinned boundaries; nd (B,N) MW balance target.
    """
    dev, dt = raw.device, raw.dtype
    sign = C._SIGN_T.to(dev, dt)
    rup = C._RUP_T.to(dev, dt)
    rdn = C._RDN_T.to(dev, dt)
    cap = C._CAP_T.to(dev, dt)
    B, N, _ = raw.shape
    dirs, step = raw[..., :6], torch.sigmoid(raw[..., 6])

    P_prev = pL
    outs, alphas, n_masked, bal_slack, anchors, moves = [], [], [], [], [], []
    for j in range(N):
        k = N - j                                   # steps from P_t to pR
        lo = torch.maximum(torch.maximum(torch.zeros_like(P_prev), P_prev - rdn),
                           pR - k * rup)
        hi = torch.minimum(torch.minimum(cap.expand_as(P_prev), P_prev + rup),
                           pR + k * rdn)
        hi = torch.maximum(hi, lo)                  # numerical guard (nonempty by induction)

        # --- 1. anchor: glide to pR at constant rate, then exact balance snap ----
        c = P_prev + (pR - P_prev) / (k + 1)        # within one ramp step by the guard
        c = torch.minimum(torch.maximum(c, lo), hi)
        delta = (nd[:, j] - (c * sign).sum(-1)).unsqueeze(-1)          # (B,1)
        # directional room: how far each channel can move the signed sum the way
        # it needs to. Guarantees |delta| <= sum(room) whenever nd_t is reachable,
        # so the snap is exact AND lands inside the box.
        room = torch.where(delta * sign > 0, hi - c, c - lo).clamp_min(0.0)
        rs = room.sum(-1, keepdim=True)
        A = c + (delta / rs.clamp_min(1e-6)) * (room * sign)
        A = torch.minimum(torch.maximum(A, lo), hi)
        bal_slack.append((delta.abs().squeeze(-1) - rs.squeeze(-1)).clamp_min(0.0))

        # --- 2. mask channels pinned at a wall and pointing out of it -----------
        r = dirs[:, j]
        eps = 1e-6
        blocked = ((A <= lo + eps) & (r < 0)) | ((A >= hi - eps) & (r > 0))
        free = (~blocked).to(dt)
        n_masked.append(blocked.sum(-1))
        r = r * free

        # --- 3. balance-tangent over the free channels, then scale-free ---------
        w = (hi - lo).clamp_min(1e-6) * free
        r = r - ((r * sign).sum(-1) / w.sum(-1).clamp_min(1e-6)).unsqueeze(-1) * (w * sign)
        r = r * free                                # keep blocked channels at exactly 0
        r = r / (r.abs().amax(-1, keepdim=True) + 1e-6)

        # --- 4. geometric wall distance x learned step --------------------------
        big = torch.full_like(A, 1e9)
        a_hi = torch.where(r > 1e-9, (hi - A) / r.clamp_min(1e-9), big).amin(-1)
        a_lo = torch.where(r < -1e-9, (A - lo) / (-r).clamp_min(1e-9), big).amin(-1)
        amax = torch.minimum(a_hi, a_lo).clamp(0.0, 1e6)
        alphas.append(amax)
        move = (step[:, j] * amax).unsqueeze(-1) * r
        P_t = A + move
        outs.append(P_t); anchors.append(A); moves.append(move)
        P_prev = P_t

    P = torch.stack(outs, 1)
    if not return_debug:
        return P
    return P, {"alpha_max": torch.stack(alphas, 1),           # (B,N) geometric room
               "step": step,                                   # (B,N) learned fraction
               "n_masked": torch.stack(n_masked, 1),           # (B,N) blocked channels
               "bal_unreachable": torch.stack(bal_slack, 1),   # (B,N) MW nd was short by
               "anchor": torch.stack(anchors, 1),              # (B,N,6) model-independent part
               "move": torch.stack(moves, 1)}                  # (B,N,6) the network's part


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import numpy as np
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT / "imputation"))
    import gap_data as GD
    GD.NPZ = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
    from gap_data import load_flats, sample_recon_windows, SIGN

    f = load_flats()
    gws = sample_recon_windows(f, "test", n=60, context=48, gap=36, seed=123)
    g = torch.Generator().manual_seed(0)
    t_ = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)

    print("feasibility self-test: RANDOM directions (worst case for the head)\n")
    worst = {"bal": 0.0, "box_lo": 0.0, "box_hi": 0.0, "ramp": 0.0}
    amins, amaxs, masked, unreach = [], [], [], []
    for gw in gws:
        raw = torch.randn(1, 36, 7, generator=g, dtype=torch.float64) * 3.0
        nd = t_(gw.truth_mw @ SIGN)[None]
        P, dbg = recursive_rayen(raw, t_(gw.pL_mw)[None], t_(gw.pR_mw)[None], nd,
                                 return_debug=True)
        p = P[0].numpy()
        full = np.vstack([gw.pL_mw[None], p, gw.pR_mw[None]])
        d = np.diff(full, axis=0)
        worst["bal"] = max(worst["bal"], float(np.abs(p @ SIGN - nd[0].numpy()).max()))
        worst["box_lo"] = max(worst["box_lo"], float(np.maximum(-p, 0).max()))
        worst["box_hi"] = max(worst["box_hi"], float(np.maximum(p - C.CAP, 0).max()))
        worst["ramp"] = max(worst["ramp"], float((np.maximum(d - C.R_UP, 0)
                                                  + np.maximum(-d - C.R_DN, 0)).max()))
        a = dbg["alpha_max"][0].numpy()
        amins.append(a.min()); amaxs.append(a.mean())
        masked.append(float(dbg["n_masked"].double().mean()))
        unreach.append(float(dbg["bal_unreachable"].max()))
    print(f"  worst balance residual      {worst['bal']:.3e} MW")
    print(f"  worst box violation (lo/hi) {worst['box_lo']:.3e} / {worst['box_hi']:.3e} MW")
    print(f"  worst ramp overshoot        {worst['ramp']:.3e} MW   (incl. both seams)")
    print(f"\n  alpha_max per step: mean {np.mean(amaxs):.1f} MW, "
          f"worst single step {np.min(amins):.3f} MW")
    print(f"  steps with alpha_max == 0:  "
          f"{100*np.mean([a == 0 for a in amins]):.1f}% of windows have at least one")
    print(f"  channels masked per step:   {np.mean(masked):.2f} of 6")
    print(f"  nd unreachable in box by:   {max(unreach):.3e} MW (0 => balance always attainable)")
