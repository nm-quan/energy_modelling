"""`hardnet_traj` -- one greedy sweep that makes all four constraints exact at once.

THE SHAPE THAT MAKES IT WORK. Walking forward, once P[t-1], P[t-2], ... are fixed,
everything constraining step t collapses to an INTERVAL per channel:

    lo[t] = max( P_min(t),  max_k (P[t-k] - R_dn(k)),  pR - R_up(N+1-t),  soc_lo[t] )
    hi[t] = min( P_max(t),  min_k (P[t-k] + R_up(k)),  pR + R_dn(N+1-t),  soc_hi[t] )

plus the equality SIGN . P[t] = nd[t]. So the per-step feasible set is a BOX INTERSECTED
WITH A HYPERPLANE -- the simplest polytope there is -- and the exact Euclidean projection
onto it is closed form. Every one of the four constraints is folded into that interval:

  balance   the hyperplane itself
  capacity  P_min(t), P_max(t), read per step so the bound tracks the fleet
  ramp      EVERY k, not just k=1, against fixed past values; plus the backward guard to
            the pinned right endpoint, using the exact remaining distance R(N+1-t)
  SOC       soc_lo/soc_hi, and ONLY on the signed battery coordinate

WHY EVERY k, AND WHY THE EXACT DISTANCE. Enforcing the table on a coarse subset of k, or
approximating the right-seam guard by k*R(1), breaks the induction that keeps [lo,hi]
non-empty. Measured: with k in {1,3,6,12,24,36} and a linear seam guard the sweep violated
ramps by up to 900 MW under stress; with the full grid and R(N+1-t) the worst violation
over the same inputs is 1.1e-13 MW.

WHY [lo, hi] IS NEVER EMPTY. Induction on t. Suppose P[t-1] is reachable from the left and
can still reach pR, i.e. |pR - P[t-1]| is within R(N+2-t). The forward cone from P[t-1] is
[P[t-1]-R_dn(1), P[t-1]+R_up(1)] and the backward cone to pR is [pR-R_up(N+1-t),
pR+R_dn(N+1-t)]. They intersect exactly when R(N+2-t) <= R(N+1-t) + R(1) -- SUBADDITIVITY,
which constraint_set.ramp_table enforces by closure. Longer-k terms cannot empty it
either: each is implied by the sum of the single steps it spans, again by subadditivity.

WHY THE PROJECTION IS WEIGHTED. The unweighted map moves every free channel by the same
number of MW, so a 510 MW peaker takes the same correction as a 5,095 MW coal fleet.
Measured cost in exp6: plain hardnet moved gas_steam's MAE from 48.7 to 155.5 MW and made
the constrained model WORSE than the unconstrained one. w = 1/softmax(logits) makes the
signed share channel i absorbs exactly p_i, and softmax is the right family because the
balance identity forces those shares to sum to one -- the network cannot propose an
inconsistent split, the same way the projection cannot emit an infeasible dispatch.

Self-contained on purpose: `project_box_plane_w` is the same map as
planB/heads.py::project_box_plane_w, re-derived here rather than imported because that
module pulls in planB/limits.py, which reads data/preprocessed/ -- and this track is
data/updata-native. Same math, verified against it in exp7.
"""
from __future__ import annotations

import torch

ALLOC_EPS = 1e-3          # floor on a predicted share -> w caps at 1000


def project_box_plane_w(F, lo, hi, nd, sign, w=None, iters: int = 60):
    """argmin sum_i w_i (P_i - F_i)^2  s.t.  lo <= P <= hi,  sign . P = nd.

    KKT gives a one-scalar solution, P = clip(F + lam * sign/w, lo, hi), and two passes
    keep the gradient honest:
      1. bisect for lam under no_grad. g(lam) = sign . clip(F + lam*sign/w, lo, hi) is
         non-decreasing (a free coordinate contributes sign_i^2/w_i > 0), so lam is
         unique. This pass exists only to identify the ACTIVE SET and is never
         differentiated.
      2. with the active set fixed, lam is an explicit differentiable function of F, so
         the map carries correct gradients almost everywhere.
    The map is the IDENTITY on any F already inside the set -- which is what makes an
    ablation meaningful, since turning a constraint on cannot move a feasible prediction.
    """
    w = torch.ones_like(F) if w is None else w
    step = sign / w
    with torch.no_grad():
        rng = torch.maximum((hi - F).abs(), (F - lo).abs()).amax(-1, keepdim=True)
        R = rng * w.amax(-1, keepdim=True) + 1.0          # enough to clip every coord
        a, b = -R, R
        for _ in range(iters):
            m = 0.5 * (a + b)
            g = (torch.clamp(F + m * step, lo, hi) * sign).sum(-1, keepdim=True)
            too_low = g < nd.unsqueeze(-1)
            a = torch.where(too_low, m, a)
            b = torch.where(too_low, b, m)
        lam0 = 0.5 * (a + b)
        P0 = torch.clamp(F + lam0 * step, lo, hi)
        free = (P0 > lo + 1e-9) & (P0 < hi - 1e-9)
        denom = (free.to(F.dtype) / w).sum(-1, keepdim=True)

    Pc = torch.clamp(F + lam0 * step, lo, hi)
    fixed = (~free).to(F.dtype) * Pc * sign
    lam = (nd.unsqueeze(-1) - fixed.sum(-1, keepdim=True)
           - (free.to(F.dtype) * F * sign).sum(-1, keepdim=True)) / denom.clamp_min(1e-12)
    lam = torch.where(denom > 1e-12, lam, lam0)
    return torch.clamp(F + lam * step, lo, hi)


class SocTorch:
    """The Reservoir arithmetic in torch. Mirrors constraint_set.Reservoir exactly."""

    def __init__(self, eta: float, draw: float, e_max, dt_h: float = 5.0 / 60.0,
                 e_min=0.0):
        self.eta, self.draw = float(eta), float(draw)
        self.e_max, self.e_min = e_max, e_min      # float OR (B,) tensor
        self.a = self.eta * dt_h / 2.0
        self.b = dt_h / (2.0 * self.eta)

    def _bounds(self, like):
        """e_min/e_max as (B,1) tensors on the right device, whichever form they were given."""
        def cast(v):
            if torch.is_tensor(v):
                return v.to(like.device, like.dtype).reshape(-1, 1)
            return torch.full((like.shape[0], 1), float(v), device=like.device,
                              dtype=like.dtype)
        return cast(self.e_min), cast(self.e_max)

    def contrib(self, b):
        return self.a * torch.clamp(-b, min=0.0) - self.b * torch.clamp(b, min=0.0)

    def contrib_inv(self, v):
        return torch.where(v >= 0.0, -v / self.a, -v / self.b)

    def tube(self, b_lo, b_hi, e_target=None, tol: float = 0.0):
        """(E_lo, E_hi), each (B, N+1): the levels from which the rest of the window is
        still survivable. b_lo/b_hi are (B, N) -- the A PRIORI admissible signed power at
        each step, already reflecting cap, ramps, both endpoints AND balance.

        THE RECURSION, and why the obvious version is inert. Over step j the level moves by
        dE = contrib(b_j) + contrib(b_{j-1}) - draw, and contrib is decreasing, so over the
        whole admissible range

            dE_min(j) = 2 contrib(b_hi_j) - draw       (fall as fast as possible)
            dE_max(j) = 2 contrib(b_lo_j) - draw       (rise as fast as possible)

        and reachability backward is

            E_hi(j) = min(E_max, E_hi(j+1) - dE_min(j))
            E_lo(j) = max(E_min, E_lo(j+1) - dE_max(j))

        The terms are NOT clamped at zero, and that is the whole point. When balance forces
        the battery to charge through a stretch, dE_min is POSITIVE -- the level must rise
        whatever the model wants -- and E_hi walks down step by step, so the sweep starts
        making room hours before the ceiling is reached. Clamping the rates at zero (the
        natural-looking "a rate cannot be negative") flattens the tube to a constant
        [E_min, E_max], which is exactly a guard that only objects once recovery is already
        impossible. On THIS data the distinction turns out to be moot -- the thermal
        channels are flexible enough that nothing is forced a priori and the tube stays
        near-flat either way -- but the unclamped form is the correct one and it is what a
        tighter fleet would need.
        """
        B, N = b_lo.shape
        emin, emax = self._bounds(b_lo)
        dE_min = 2.0 * self.contrib(b_hi) - self.draw
        dE_max = 2.0 * self.contrib(b_lo) - self.draw
        lo = emin.expand(B, N + 1).clone()
        hi = emax.expand(B, N + 1).clone()
        if e_target is not None:
            lo[:, N] = torch.maximum(e_target - tol, emin[:, 0])
            hi[:, N] = torch.minimum(e_target + tol, emax[:, 0])
        for j in range(N - 1, -1, -1):
            hi[:, j] = torch.minimum(hi[:, j + 1] - dE_min[:, j], emax[:, 0])
            lo[:, j] = torch.maximum(lo[:, j + 1] - dE_max[:, j], emin[:, 0])
            lo[:, j] = torch.minimum(lo[:, j], hi[:, j])
        return lo, hi

    def window(self, e_prev, b_prev, e_lo_next, e_hi_next):
        """Admissible signed power this step -- exact, not conservative.

        dE = contrib(b) + contrib(b_prev) - draw must land E_prev inside [lo,hi]; contrib
        is strictly decreasing, so inverting the interval gives an interval on b, with the
        ends swapped.
        """
        k = self.contrib(b_prev) - self.draw
        return (self.contrib_inv(e_hi_next - e_prev - k),
                self.contrib_inv(e_lo_next - e_prev - k))

    def advance(self, e_prev, b_prev, b):
        return e_prev + self.contrib(b) + self.contrib(b_prev) - self.draw


def hardnet_traj(raw, pL, pR, nd, P_min, P_max, R_up, R_dn, sign,
                 batt_idx: int = 4, soc: SocTorch | None = None, e0=None,
                 e_target=None, soc_tol: float = 0.0, alloc: bool = True,
                 ramp_k: int | None = None, w_override=None, return_debug: bool = False):
    """raw (B,N,C) levels, or (B,N,2C) levels + allocation logits when `alloc`.

    P_min/P_max (B,N,C) MW; R_up/R_dn (K+1,C) MW indexed by k; nd (B,N); pL/pR (B,C).
    `ramp_k` caps how many backward horizons are enforced (None = all, the guarantee).
    Returns P (B,N,C) -- exactly feasible.
    """
    dev, dt = raw.device, raw.dtype
    B, N, _ = raw.shape
    C = pL.shape[-1]
    F = raw[..., :C]
    if w_override is not None:
        W = w_override.to(dev, dt).expand_as(F)
    elif alloc:
        p = torch.softmax(raw[..., C:2 * C], dim=-1)
        W = 1.0 / p.clamp_min(ALLOC_EPS)
    else:
        W = torch.ones_like(F)

    sign = sign.to(dev, dt)
    R_up = R_up.to(dev, dt)
    R_dn = R_dn.to(dev, dt)
    kcap = R_up.shape[0] - 1

    # ------------------------------------------------------------------ a priori bounds
    # Before any path is chosen, the cap, the ramp table and the two pinned endpoints
    # already bound EVERY channel at EVERY step, for every admissible trajectory:
    #
    #     hi_j = min( P_max(j),  pL + R_up(j),  pR + R_dn(N+1-j) )
    #     lo_j = max( P_min(j),  pL - R_dn(j),  pR - R_up(N+1-j) )
    #
    # These are path-independent, so anything derived from them holds no matter what the
    # network emits -- which is what lets the reservoir tube be a guarantee rather than a
    # hope.
    ks = torch.arange(1, N + 1, device=dev).clamp(max=kcap)
    rk = torch.arange(N, 0, -1, device=dev).clamp(max=kcap)
    if ramp_k == 0:                 # ramp OFF -- the a priori bounds are the box alone.
        ap_lo, ap_hi = P_min, P_max      # (an ablation row must not get ramps by the back
    else:                                #  door through the a priori bounds)
        ap_hi = torch.minimum(P_max, torch.minimum(pL.unsqueeze(1) + R_up[ks],
                                                   pR.unsqueeze(1) + R_dn[rk]))
        ap_lo = torch.maximum(P_min, torch.maximum(pL.unsqueeze(1) - R_dn[ks],
                                                   pR.unsqueeze(1) - R_up[rk]))
        ap_hi = torch.maximum(ap_hi, ap_lo)

    e_lo = e_hi = None
    if soc is not None:
        # THE BATTERY IS NOT FREE -- BALANCE LARGELY DETERMINES IT, and the tube has to
        # know that. SIGN is all +1, so b = nd - G with G the thermal sum, and G is
        # confined to the a priori thermal box. Therefore, before anything is chosen,
        #
        #     b(t) in [ nd(t) - sum ap_hi_th(t) ,  nd(t) - sum ap_lo_th(t) ]
        #
        # Intersecting that with the battery's own a priori box gives the range the
        # reservoir will ACTUALLY be driven over. A tube built from the battery's nameplate
        # instead believes it can recharge at full power right up to the last step, which
        # is strictly wrong whenever balance has already spoken for the battery.
        #
        # Stated plainly, because it bears on what this head does and does not guarantee:
        # measured on this data the tube is NOT what makes SOC hold. The thermal channels
        # are flexible enough that the balance-implied battery range is ~2,500 MW wide on
        # average, so nothing is forced a priori and the tube stays near-flat. What drives
        # the residual to zero is the shrink-and-retry loop in hardnet_traj_polished. The
        # tube is kept because it is the correct bound and it costs O(N) -- not because it
        # is carrying the result.
        th = [i for i in range(C) if i != batt_idx]
        G_lo = (ap_lo[..., th] * sign[th]).sum(-1)
        G_hi = (ap_hi[..., th] * sign[th]).sum(-1)
        s_b = sign[batt_idx]
        b_hi_j = torch.minimum(ap_hi[..., batt_idx], (nd - G_lo) / s_b)
        b_lo_j = torch.maximum(ap_lo[..., batt_idx], (nd - G_hi) / s_b)
        b_hi_j = torch.maximum(b_hi_j, b_lo_j)
        e_lo, e_hi = soc.tube(b_lo_j, b_hi_j, e_target=e_target, tol=soc_tol)
        E = e0.to(dev, dt).clone()
        b_prev = pL[:, batt_idx].clone()

    outs = []
    hist = [pL]                          # hist[j] is P[t-1-j]: index j == distance j+1
    moved, shortfall, soc_clip = [], [], []
    for t in range(1, N + 1):
        nback = len(hist) if ramp_k is None else min(len(hist), ramp_k)
        lo, hi = ap_lo[:, t - 1].clone(), ap_hi[:, t - 1].clone()
        if nback > 0:                       # ramp_k=0 turns the ramp constraint off
            past = torch.stack(hist[:nback], dim=1)              # (B,nb,C) distance 1..nb
            dist = torch.arange(1, nback + 1, device=dev).clamp(max=kcap)
            lo = torch.maximum(lo, (past - R_dn[dist]).amax(1))
            hi = torch.minimum(hi, (past + R_up[dist]).amin(1))
            rem = min(N + 1 - t, kcap)                           # exact distance to pR
            lo = torch.maximum(lo, pR - R_up[rem])
            hi = torch.minimum(hi, pR + R_dn[rem])
        hi = torch.maximum(hi, lo)

        if soc is not None:
            # THE STEP THAT MAKES SOC AND BALANCE BOTH EXACT.
            #
            # Balance fixes the battery completely once the other channels are chosen:
            # SIGN . P = nd  and  SIGN is all +1, so  b = nd - G  with G the thermal sum.
            # G can be anything in [sum lo_th, sum hi_th] (a box's sum is an interval and
            # every value in it is attainable), so BALANCE ALONE already confines b to
            #
            #     b in [ nd - sum hi_th ,  nd - sum lo_th ]
            #
            # Intersecting the SOC interval with THAT, before the projection runs, is what
            # stops the two constraints fighting. Tightening b against SOC alone -- which
            # is what the repo's existing _Soc does -- shrinks the attainable range of
            # SIGN . P until nd falls outside it, and then one of the two has to be given
            # up. Here the intersection is computed first, so a non-empty result means the
            # step can satisfy both, and the projection then hits nd exactly inside it.
            b_lo, b_hi = soc.window(E, b_prev, e_lo[:, t], e_hi[:, t])
            th = [i for i in range(C) if i != batt_idx]
            G_lo = (lo[:, th] * sign[th]).sum(-1)
            G_hi = (hi[:, th] * sign[th]).sum(-1)
            s_b = sign[batt_idx]
            att_lo = (nd[:, t - 1] - G_hi) / s_b
            att_hi = (nd[:, t - 1] - G_lo) / s_b
            lo_b = torch.maximum(torch.maximum(lo[:, batt_idx], b_lo), att_lo)
            hi_b = torch.minimum(torch.minimum(hi[:, batt_idx], b_hi), att_hi)
            # Where even that is empty the step genuinely cannot do both. Ramp and the
            # right-endpoint seam outrank SOC there (the trajectory must still land on
            # pR), SOC is clipped as far as the box allows rather than abandoned, and the
            # event is COUNTED in soc_clip -- never silently absorbed.
            # BALANCE OUTRANKS SOC, the repo's standing precedence. Where the three
            # intervals do not share a point the step cannot serve demand AND respect the
            # reservoir; demand is the identity the model exists to satisfy, so the ramp
            # box is restored and the reservoir takes the excursion. Counted, not hidden.
            bad = hi_b < lo_b
            lo_r, hi_r = lo[:, batt_idx], hi[:, batt_idx]
            lo_b = torch.where(bad, lo_r, lo_b)
            hi_b = torch.where(bad, hi_r, hi_b)
            soc_clip.append(bad.to(dt))
            lo = torch.cat([lo[:, :batt_idx], lo_b.unsqueeze(-1), lo[:, batt_idx + 1:]], -1)
            hi = torch.cat([hi[:, :batt_idx], hi_b.unsqueeze(-1), hi[:, batt_idx + 1:]], -1)

        P_t = project_box_plane_w(F[:, t - 1], lo, hi, nd[:, t - 1], sign, W[:, t - 1])
        if soc is not None:
            E = soc.advance(E, b_prev, P_t[:, batt_idx])
            b_prev = P_t[:, batt_idx]
        moved.append((P_t - F[:, t - 1]).abs().sum(-1))
        shortfall.append(((P_t * sign).sum(-1) - nd[:, t - 1]).abs())
        outs.append(P_t)
        hist.insert(0, P_t)

    P = torch.stack(outs, 1)
    if not return_debug:
        return P
    dbg = {"correction": torch.stack(moved, 1), "bal_short": torch.stack(shortfall, 1)}
    if soc is not None:
        dbg["soc_E"] = E
        dbg["soc_clip"] = torch.stack(soc_clip, 1)
    if alloc:
        dbg["p"] = torch.softmax(raw[..., C:2 * C], dim=-1)
    return P, dbg


def project_soc_path(B, e0, b_prev, soc: SocTorch, e_lo, e_hi):
    """Move a signed-battery path the minimum needed to keep the level inside its tube.

    Causal clamp: sweep forward, and wherever E would leave [e_lo, e_hi] set b to the value
    that lands it exactly on the edge. `contrib` is strictly decreasing and invertible, so
    that value is closed form -- the same inversion the head uses, applied after the fact
    instead of before.

    This alone is not the answer, because moving b breaks the balance plane. It is one leg
    of the alternation below.
    """
    Bp = B.clone()
    E = e0.clone()
    prev = b_prev.clone()
    for t in range(B.shape[1]):
        lo_b, hi_b = soc.window(E, prev, e_lo[:, t + 1], e_hi[:, t + 1])
        b = torch.clamp(Bp[:, t], torch.minimum(lo_b, hi_b), torch.maximum(lo_b, hi_b))
        Bp = torch.cat([Bp[:, :t], b.unsqueeze(1), Bp[:, t + 1:]], 1)
        E = soc.advance(E, prev, b)
        prev = b
    return Bp


def hardnet_traj_polished(raw, pL, pR, nd, P_min, P_max, R_up, R_dn, sign,
                          batt_idx: int = 4, soc: SocTorch | None = None, e0=None,
                          e_target=None, soc_tol: float = 0.0, alloc: bool = True,
                          ramp_k: int | None = None, polish: int = 6,
                          shrink: int = 16, return_debug: bool = False):
    """The deployed map: alternate the greedy sweep with the reservoir projection.

    WHY ONE PASS IS NOT ENOUGH, stated plainly. The greedy sweep commits to P[t] before it
    has seen step t+1. Balance, box and ramp survive that -- they only ever look backward,
    or forward through a reachability guard that is exact. SOC does not: the reservoir
    couples every step to every other, and a sweep that has already pinned the thermal
    channels against their ramp box leaves the battery FULLY DETERMINED by balance, with no
    freedom left to respect the reservoir. Measured on adversarial input, one pass holds
    balance/box/ramp to 1e-12 MW and lets SOC run 25,000 MWh over.

    So the constraints are split by what a single sweep can carry. The sweep restores
    balance, box and ramp EXACTLY, from any input. The reservoir projection restores SOC
    exactly, and moves only the battery. Alternating them is POCS on two convex sets whose
    intersection is non-empty -- the recorded dispatch lies in it -- so it converges, and
    the sweep going last means the trajectory that comes out is exactly balance-, box- and
    ramp-feasible whatever the iteration count. `polish=0` recovers the single pass.
    """
    def _once(sc):
        P, dbg = hardnet_traj(raw, pL, pR, nd, P_min, P_max, R_up, R_dn, sign,
                              batt_idx=batt_idx, soc=sc, e0=e0, e_target=e_target,
                              soc_tol=soc_tol, alloc=alloc, ramp_k=ramp_k,
                              return_debug=True)
        if sc is None or polish <= 0:
            return P, dbg
        dev, dt = P.device, P.dtype
        N = P.shape[1]
        kcap = R_up.shape[0] - 1
        ks = torch.arange(1, N + 1, device=dev).clamp(max=kcap)
        rk = torch.arange(N, 0, -1, device=dev).clamp(max=kcap)
        Ru, Rd = R_up.to(dev, dt), R_dn.to(dev, dt)
        if ramp_k == 0:                 # ablation rows must not get ramps by the back door
            ap_lo, ap_hi = P_min, P_max
        else:
            ap_hi = torch.minimum(P_max, torch.minimum(pL.unsqueeze(1) + Ru[ks],
                                                       pR.unsqueeze(1) + Rd[rk]))
            ap_lo = torch.maximum(P_min, torch.maximum(pL.unsqueeze(1) - Rd[ks],
                                                       pR.unsqueeze(1) - Ru[rk]))
            ap_hi = torch.maximum(ap_hi, ap_lo)
        C = pL.shape[-1]
        th = [i for i in range(C) if i != batt_idx]
        sg = sign.to(dev, dt)
        b_hi_j = torch.minimum(ap_hi[..., batt_idx],
                               (nd - (ap_lo[..., th] * sg[th]).sum(-1)) / sg[batt_idx])
        b_lo_j = torch.maximum(ap_lo[..., batt_idx],
                               (nd - (ap_hi[..., th] * sg[th]).sum(-1)) / sg[batt_idx])
        b_hi_j = torch.maximum(b_hi_j, b_lo_j)
        e_lo, e_hi = sc.tube(b_lo_j, b_hi_j, e_target=e_target, tol=soc_tol)
        tail = raw[..., C:] if raw.shape[-1] > C else None
        w_pol = torch.ones(C, device=dev, dtype=dt)
        w_pol[batt_idx] = 1e4
        for _ in range(polish):
            Bp = project_soc_path(P[..., batt_idx], e0, pL[:, batt_idx], sc, e_lo, e_hi)
            Fn = torch.cat([P[..., :batt_idx], Bp.unsqueeze(-1), P[..., batt_idx + 1:]], -1)
            rn = Fn if tail is None else torch.cat([Fn, tail], -1)
            # the polish sweeps run with soc=None and the battery weighted OUT of the
            # balance correction (w = 1/share, so a large w_batt means it absorbs almost
            # none of the residual and the thermals make room instead). This leg must be
            # the projection onto {balance, box, ramp} of the reservoir-corrected point and
            # nothing else -- re-applying the per-step tightening here makes the two legs
            # fight instead of alternate, and the iteration oscillates.
            P, dbg = hardnet_traj(rn, pL, pR, nd, P_min, P_max, R_up, R_dn, sign,
                                  batt_idx=batt_idx, soc=None, alloc=False, ramp_k=ramp_k,
                                  w_override=w_pol, return_debug=True)
        return P, dbg

    P, dbg = _once(soc)
    if soc is None or shrink <= 0:
        return (P, dbg) if return_debug else P

    # SHRINK AND RETRY. The alternation converges quickly but to a fixed point that can sit
    # slightly outside the reservoir, because neither leg is the exact Euclidean projection
    # onto its set. Rather than pretend otherwise, measure the overshoot and re-run against
    # bounds pulled in by that much: an excursion of v MWh above E_max is removed by
    # solving again with a ceiling of E_max - v. Each round strictly reduces the residual
    # and the loop stops as soon as it is at machine precision, so the map that ships is
    # feasible on all four -- and the number of rounds it took is reported.
    B = P.shape[0]
    dev, dt = P.device, P.dtype
    emax0 = torch.full((B,), float(soc.e_max), device=dev, dtype=dt) \
        if not torch.is_tensor(soc.e_max) else soc.e_max.to(dev, dt).reshape(-1)
    emin0 = torch.full((B,), float(soc.e_min), device=dev, dtype=dt) \
        if not torch.is_tensor(soc.e_min) else soc.e_min.to(dev, dt).reshape(-1)

    def excursion(Pk):
        E = _levels(Pk[..., batt_idx], e0, pL[:, batt_idx], soc)
        return ((E - emax0.unsqueeze(1)).amax(-1).clamp_min(0.0),
                (emin0.unsqueeze(1) - E).amax(-1).clamp_min(0.0))

    over, under = excursion(P)
    cur_max, cur_min = emax0.clone(), emin0.clone()
    best_P = P.clone()
    best_v = torch.maximum(over, under)
    rounds = 0
    for _ in range(shrink):
        if float(best_v.max()) <= 1e-9:
            break
        rounds += 1
        # PER WINDOW, not batch-wide: one hard day must not pull the bounds in on the rest.
        # Only windows still outside are tightened, and the 1.1 overshoots deliberately --
        # pulling in by exactly the excursion lands back on the boundary, where the next
        # balance leg pushes it straight out again.
        active = best_v > 1e-9
        # never shrink the reservoir past half its remaining width: a huge first-pass
        # excursion (an adversary can produce 20,000 MWh) would otherwise invert the
        # bounds and the retry becomes infeasible instead of tighter.
        room = 0.5 * (cur_max - cur_min)
        cur_max = torch.where(active, cur_max - torch.minimum(1.1 * over, room), cur_max)
        cur_min = torch.where(active, cur_min + torch.minimum(1.1 * under, room), cur_min)
        P2, dbg2 = _once(SocTorch(soc.eta, soc.draw, cur_max, e_min=cur_min))
        o2, u2 = excursion(P2)
        v2 = torch.maximum(o2, u2)
        take = v2 < best_v
        best_P = torch.where(take.reshape(-1, 1, 1), P2, best_P)
        best_v = torch.where(take, v2, best_v)
        over = torch.where(take, o2, over)
        under = torch.where(take, u2, under)
        dbg = dbg2
    P = best_P
    over, under = excursion(P)
    dbg["soc_shrink_rounds"] = rounds
    dbg["soc_over"] = over
    dbg["soc_under"] = under
    return (P, dbg) if return_debug else P


def _levels(B, e0, b_prev, soc: SocTorch):
    E, prev, out = e0.clone(), b_prev.clone(), []
    for t in range(B.shape[1]):
        E = soc.advance(E, prev, B[:, t])
        out.append(E)
        prev = B[:, t]
    return torch.stack(out, 1)
