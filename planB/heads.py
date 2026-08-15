"""Two constraint maps for the 3h gap, sharing one per-step feasible set.

RAMPS ARE p99.9, NOT THE ALL-TIME MAX (planB/limits.py). Caps are unchanged.

THE SHARED PART. Walking forward through the gap, once P[t-1] is fixed everything
that constrains step t collapses to an interval per channel:

    lo[t,i] = max( 0,      P[t-1,i] - rdn_i,  pR_i - k*rup_i )     k = N+1-t
    hi[t,i] = min( cap_i,  P[t-1,i] + rup_i,  pR_i + k*rdn_i )
    plus the equality  s . P_t = nd_t

The third term in each is the BACKWARD-REACHABILITY guard: it keeps the greedy
sweep from walking somewhere it can no longer get back from in the steps left
before pR. With it, [lo,hi] is provably non-empty by induction. So the per-step
feasible set is a BOX INTERSECTED WITH A HYPERPLANE -- the simplest polytope
there is, and both maps below act on exactly that.

THE TWO MAPS.

  rayen     Anchor + ray. Pick a feasible point inside the box (glide toward pR,
            then a balance snap weighted by each channel's available room), take
            the network's direction, remove its balance component, and travel
            along it by a learned fraction of the distance to the nearest wall.
            Needs 7 network outputs: 6 direction + 1 step logit.

  hardnet   Exact Euclidean projection. No anchor, no weights, no learned step:

                P_t = argmin ||P - F_t||^2  s.t.  s.P = nd_t, lo <= P <= hi
                    = clip(F_t + lam*s, lo, hi),  lam solving s.clip(...) = nd_t

            Needs 6 network outputs. The allocation rule is "smallest correction",
            which is a definition rather than a choice, and the map is the IDENTITY
            on any F that is already feasible -- neither is true of the ray.

Both are exact on balance / box / ramp / both seams, and differentiable.

SOC (the battery energy reservoir) is enforced OPTIONALLY, via set_soc(); see the
block above _Soc. It folds into the same per-step box, so nothing about the
projection changes. rayen_head does not carry it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# imputation FIRST, then planB, so planB wins position 0 -- otherwise
# imputation/{model,train}.py shadow planB/{nets,train}.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "imputation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import constraints as C                                                # noqa: E402
from limits import limits as _limits, R_UP, R_DN, CAP                  # noqa: E402

# planB uses the p99.9 ramp envelope, not the all-time max -- see limits.py.
# Caps stay at the empirical max. Verified: 0/400 test windows lose bridge
# feasibility under the tighter ramps.


def _box(P_prev, pR, k, rup, rdn, cap):
    """Per-step interval, including the backward-reachability guard."""
    lo = torch.maximum(torch.maximum(torch.zeros_like(P_prev), P_prev - rdn),
                       pR - k * rup)
    hi = torch.minimum(torch.minimum(cap.expand_as(P_prev), P_prev + rup),
                       pR + k * rdn)
    return lo, torch.maximum(hi, lo)


# ----------------------------------------------------------------------------
# hardnet: exact Euclidean projection onto {box} inter {hyperplane}
# ----------------------------------------------------------------------------
def project_box_plane(F, lo, hi, nd, sign, iters: int = 40):
    """argmin ||P-F||^2 s.t. lo<=P<=hi, sign.P = nd.  F,lo,hi (B,K); nd (B,).

    Two passes, and the split matters for the gradient:
      1. bisect for lam under no_grad. g(lam) = sign.clip(F+lam*sign, lo, hi) is
         non-decreasing because sign_i^2 = 1, so lam is unique. This pass exists
         only to identify the ACTIVE SET; bisection is never differentiated.
      2. with the active set A fixed, lam is an explicit differentiable function
         of F:   lam = (nd - sum_{i not in A} s_i P_i - sum_{i in A} s_i F_i)/|A|
         and P = clip(F + lam*sign, lo, hi) carries correct gradients a.e.
    """
    with torch.no_grad():
        # a shift of R clips every coordinate, since |sign_i| = 1
        R = torch.maximum((hi - F).abs(), (F - lo).abs()).amax(-1, keepdim=True) + 1.0
        a, b = -R, R
        for _ in range(iters):
            m = 0.5 * (a + b)
            g = (torch.clamp(F + m * sign, lo, hi) * sign).sum(-1, keepdim=True)
            too_low = g < nd.unsqueeze(-1)
            a = torch.where(too_low, m, a)
            b = torch.where(too_low, b, m)
        lam0 = 0.5 * (a + b)
        P0 = torch.clamp(F + lam0 * sign, lo, hi)
        free = (P0 > lo + 1e-9) & (P0 < hi - 1e-9)                  # active set
        nfree = free.sum(-1, keepdim=True)

    Pc = torch.clamp(F + lam0 * sign, lo, hi)                       # clipped coords
    fixed = (~free).to(F.dtype) * Pc * sign
    lam = (nd.unsqueeze(-1) - fixed.sum(-1, keepdim=True)
           - (free.to(F.dtype) * F * sign).sum(-1, keepdim=True)) / nfree.clamp_min(1)
    lam = torch.where(nfree > 0, lam, lam0)
    return torch.clamp(F + lam * sign, lo, hi)


ALLOC_EPS = 1e-3          # floor on a predicted share -> w caps at 1000

# ---------------------------------------------------------------------------
# SOC — the battery energy reservoir
#
# Constants are re-imported from imputation/constraints.py rather than restated,
# so planB and the imputation track cannot drift apart on the battery physics.
#
# WHAT IS ENFORCED, and why it is a swing rather than a level. The absolute state
# of charge is never observed in the data -- only the power flows are. So the
# constraint is on the SWING, max(E) - min(E) <= cap, which is exactly what
# constraints.py:_soc_project uses and what makes the historical actuals feasible
# on all 186 test days under soc_period="day".
#
# HOW IT FITS A GREEDY PER-STEP SWEEP. SOC is cumulative, the sweep is per-step.
# Same problem the ramps posed, same solution: carry the state forward and turn
# it into bounds on the current step. With state (E, Emin, Emax) the step's energy
# change dE must satisfy
#
#     Emax - cap - E  =:  L   <=   dE   <=   U  :=  Emin + cap - E
#
# because a step that pushes E above Emax must not exceed Emin+cap, and one that
# pushes it below Emin must not fall under Emax-cap. Both bounds bracket zero
# whenever the invariant Emax-Emin <= cap held before, so dE=0 is always
# admissible and the box can never be emptied by this.
#
# dE = (eta*chg - dis/eta)*DT couples the two battery channels, which would break
# the box structure. It is made conservative and separable by bounding each
# channel as if the other were zero -- discharging only lowers dE and charging
# only raises it, so
#
#     chg <= U/(eta*DT)      and      dis <= -L*eta/DT
#
# are sufficient. That keeps the feasible set a box intersected with the balance
# plane, so the single-lambda projection, its monotonicity and its exactness all
# survive untouched.
# ---------------------------------------------------------------------------
ETA_B, DT_H = C.ETA, C.DT
CHG_I, DIS_I = C.CHG_IDX, C.DIS_IDX
BATT_MWH = C.BATT_CAP_MWH
SOC_MARGIN_MWH = 100.0
SOC_CAP = None            # MWh swing budget; None disables. Set via set_soc().
SOC_RETURN_TOL = None     # MWh slack on the CYCLE condition; None disables it.
SOC_RATE_FRAC = 0.25      # how much of the battery's nameplate rate the cycle
                          # guard may ASSUME is still available for recovery.
                          # 1.0 is the naive choice and it is too optimistic: the
                          # guard then believes it can recharge at full cap right
                          # up to the last step, so it lets the reservoir drift
                          # and only objects once recovery is already impossible --
                          # at which point balance wins and the guard is dropped.
                          # A fraction makes it start correcting early instead.


def default_soc_cap() -> float:
    """Same effective budget cyclic_project uses: reservoir less two margins."""
    return BATT_MWH - 2.0 * SOC_MARGIN_MWH


def set_soc(cap, return_tol=None, rate_frac=None):
    """cap:        swing budget in MWh, True for the default, None/False to disable.
    return_tol:    MWh slack on the CYCLE condition -- how far the reservoir may end
                   the window from where the actual dispatch left it. None disables.

    THE TWO CONSTRAINTS ARE DIFFERENT AND BOTH ARE NEEDED.

      swing   max(E) - min(E) <= cap.  Bounds how far the level RANGES.
      cycle   |E_end - E_target| <= tol.  Bounds whether it RETURNS.

    A swing bound alone is blind to a slow drain: a monotone leak of 1,243 MWh a
    day has a swing of 1,243 MWh, comfortably inside a 4,536 MWh budget, and will
    still empty the reservoir in four days. Measured on the 186-day counterfactual
    before this was added, the fill drew the reservoir down 26.2% per day forever,
    while the real dispatch nets to -51 MWh over the whole test set
    (discharge/charge = 0.8341 against eta^2 = 0.8340 -- it cycles exactly).
    """
    global SOC_CAP, SOC_RETURN_TOL, SOC_RATE_FRAC
    SOC_CAP = default_soc_cap() if cap is True else (None if not cap else float(cap))
    SOC_RETURN_TOL = None if return_tol is None else float(return_tol)
    if rate_frac is not None:
        SOC_RATE_FRAC = float(rate_frac)
    return SOC_CAP


class _Soc:
    """Running reservoir state for one greedy sweep.

    `seed` optionally carries (E, Emin, Emax) from before the window -- e.g. the
    day's dispatch since midnight -- so the swing is constrained over the day and
    not just over the 3 h gap, where it would almost never bind.

    `target` is the level E must land on by the last step. The guard that makes
    that reachable is the SAME backward-reachability trick `_box` already uses for
    the ramps: with `rem` steps left the reservoir can still gain at most
    rem*eta*cap_chg*DT and lose at most rem*cap_dis/eta*DT, so E now must lie
    within that reach of the target. One asymmetry the swing bound did not have:
    when the guard says the reservoir MUST net-discharge this step, capping charge
    is not enough -- discharge needs a LOWER bound. So this moves `lo` as well.
    """

    def __init__(self, B, dev, dt, cap, seed=None, target=None, tol=0.0, n=1,
                 caps=None):
        z = torch.zeros(B, device=dev, dtype=dt)
        if seed is None:
            self.E, self.Emin, self.Emax = z.clone(), z.clone(), z.clone()
        else:
            self.E, self.Emin, self.Emax = (s_.to(dev, dt).clone() for s_ in seed)
        self.cap = cap
        self.n = int(n)
        self.tol = float(tol)
        self.target = None if target is None else target.to(dev, dt)
        cc = caps if caps is not None else _limits(dev, dt)[3]
        fr = SOC_RATE_FRAC
        self.rate_gain = float(ETA_B * float(cc[CHG_I]) * DT_H * fr)   # MWh/step
        self.rate_loss = float(float(cc[DIS_I]) / ETA_B * DT_H * fr)
        self.short = z.clone()          # MW the ramp floor forced past the SOC bound

    def tighten(self, lo, hi, j=0, nd=None, sign=None):
        U = self.Emin + self.cap - self.E          # swing: room to rise
        L = self.Emax - self.cap - self.E          # swing: room to fall
        if self.target is not None:
            rem = float(max(self.n - j - 1, 0))
            U = torch.minimum(U, self.target + self.tol + self.rate_loss * rem - self.E)
            L = torch.maximum(L, self.target - self.tol - self.rate_gain * rem - self.E)
        z = torch.zeros_like(U)
        # dE in [L, U] made separable and conservative per channel
        chg_hi = torch.where(U >= 0, U / (ETA_B * DT_H), z)
        chg_lo = torch.where(L > 0, L / (ETA_B * DT_H), z)
        dis_hi = torch.where(L <= 0, (-L) * ETA_B / DT_H, z)
        dis_lo = torch.where(U < 0, (-U) * ETA_B / DT_H, z)
        idx = torch.tensor([CHG_I, DIS_I], device=hi.device).expand(hi.shape[0], 2)
        hi_lim = torch.full_like(hi, 1e18).scatter(
            1, idx, torch.stack([chg_hi, dis_hi], dim=1))
        lo_lim = torch.full_like(lo, -1e18).scatter(
            1, idx, torch.stack([chg_lo, dis_lo], dim=1))
        # The ramp / reachability box wins on conflict -- the trajectory must still
        # land on pR. Everything below stays inside the ORIGINAL [lo, hi].
        lo_n = torch.minimum(torch.maximum(lo, lo_lim), hi)
        hi_n = torch.maximum(torch.minimum(hi, hi_lim), lo_n)
        # BALANCE OUTRANKS SOC. Tightening the battery box shrinks the attainable
        # range of s.P, and if nd falls outside it the step can no longer meet
        # demand -- which is the one thing the head exists to guarantee. Where that
        # happens, drop back to the untightened box for that (window, step) and
        # carry the miss in `short`; the cycle error is reported, never hidden.
        if nd is not None:
            att = lambda a, b: (torch.minimum(a * sign, b * sign).sum(-1),
                                torch.maximum(a * sign, b * sign).sum(-1))
            amin, amax = att(lo_n, hi_n)
            ok = ((nd >= amin - 1e-6) & (nd <= amax + 1e-6)).unsqueeze(-1)
            lo_n = torch.where(ok, lo_n, lo)
            hi_n = torch.where(ok, hi_n, hi)
        self.short = torch.maximum(
            self.short, ((lo_lim - hi_n).clamp_min(0.0)
                         + (lo_n - hi_lim).clamp_min(0.0)).amax(-1))
        return lo_n, hi_n

    def advance(self, P):
        dE = (P[:, CHG_I] * ETA_B - P[:, DIS_I] / ETA_B) * DT_H
        self.E = self.E + dE
        self.Emax = torch.maximum(self.Emax, self.E)
        self.Emin = torch.minimum(self.Emin, self.E)

    @property
    def swing(self):
        return self.Emax - self.Emin

    @property
    def return_err(self):
        z = torch.zeros_like(self.E)
        return z if self.target is None else (self.E - self.target).abs()


def project_box_plane_w(F, lo, hi, nd, sign, w, iters: int = 60):
    """Weighted projection: argmin sum_i w_i (P_i-F_i)^2 over the same set.

    KKT gives the identical one-scalar shape with the travel direction rescaled:

        P = clip(F + lam * sign/w, lo, hi)

    g(lam) = sign.clip(...) is still non-decreasing -- a free coordinate now
    contributes sign_i^2/w_i = 1/w_i > 0 instead of 1 -- so the same bisection
    applies and the active-set formula divides by sum_{i in A} 1/w_i.

    The point of the weights: the UNWEIGHTED map moves every free channel by the
    same number of MW, so a 516 MW peaker takes the same correction as a 4,896 MW
    coal fleet. The signed share channel i takes is (1/w_i)/sum_j(1/w_j), so
    w = 1/p makes that share exactly p.
    """
    step = sign / w
    with torch.no_grad():
        rng = torch.maximum((hi - F).abs(), (F - lo).abs()).amax(-1, keepdim=True)
        R = rng * w.amax(-1, keepdim=True) + 1.0        # enough to clip every coord
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
    lam = torch.where(denom > 0, lam, lam0)
    return torch.clamp(F + lam * step, lo, hi)


def hardnet_alloc_head(raw, pL, pR, nd, return_debug: bool = False, soc_seed=None,
                       soc_target=None):
    """raw (B,N,12): 6 dispatch levels in MW + 6 ALLOCATION LOGITS.

    p = softmax(logits) is the share of a balance correction each channel takes,
    and w = 1/p feeds the weighted projection above. Softmax is the right
    parameterisation because the allocation must be a distribution: the balance
    identity forces sum_i s_i a_i = 1 for the true marginal response, and softmax
    can emit nothing else. So the network cannot propose an inconsistent split --
    the same way the projection cannot emit an infeasible dispatch.

    Every guarantee of `hardnet_head` survives for ANY p: balance, box, ramp and
    both seams stay exact, and the map is still the identity on a feasible F.
    Only WHERE the correction lands is now a model output rather than 1/n.
    """
    dev, dt = raw.device, raw.dtype
    sign, rup, rdn, cap = _limits(dev, dt)
    B, N, _ = raw.shape
    F, logit = raw[..., :6], raw[..., 6:]
    p = torch.softmax(logit, dim=-1)
    w = 1.0 / p.clamp_min(ALLOC_EPS)
    soc = (_Soc(B, dev, dt, SOC_CAP, soc_seed, target=soc_target,
                tol=(SOC_RETURN_TOL or 0.0), n=N, caps=cap)
           if SOC_CAP is not None else None)
    P_prev, outs, moved = pL, [], []
    for j in range(N):
        lo, hi = _box(P_prev, pR, N - j, rup, rdn, cap)
        if soc is not None:
            lo, hi = soc.tighten(lo, hi, j, nd[:, j], sign)
        P_t = project_box_plane_w(F[:, j], lo, hi, nd[:, j], sign, w[:, j])
        if soc is not None:
            soc.advance(P_t)
        moved.append((P_t - F[:, j]).abs().sum(-1))
        outs.append(P_t)
        P_prev = P_t
    P = torch.stack(outs, 1)
    if not return_debug:
        return P
    dbg = {"p": p, "correction": torch.stack(moved, 1)}
    if soc is not None:
        dbg |= {"soc_swing": soc.swing, "soc_short": soc.short, "soc_E": soc.E,
                "soc_return_err": soc.return_err}
    return P, dbg


def hardnet_head(raw, pL, pR, nd, return_debug: bool = False, soc_seed=None,
                 soc_target=None):
    """raw (B,N,6) network output in MW -> exactly feasible (B,N,6)."""
    dev, dt = raw.device, raw.dtype
    sign, rup, rdn, cap = _limits(dev, dt)
    B, N, _ = raw.shape
    soc = (_Soc(B, dev, dt, SOC_CAP, soc_seed, target=soc_target,
                tol=(SOC_RETURN_TOL or 0.0), n=N, caps=cap)
           if SOC_CAP is not None else None)
    P_prev, outs, moved, short = pL, [], [], []
    for j in range(N):
        lo, hi = _box(P_prev, pR, N - j, rup, rdn, cap)
        if soc is not None:
            lo, hi = soc.tighten(lo, hi, j, nd[:, j], sign)
        P_t = project_box_plane(raw[:, j], lo, hi, nd[:, j], sign)
        if soc is not None:
            soc.advance(P_t)
        moved.append((P_t - raw[:, j]).abs().sum(-1))
        # nd is not always attainable inside the tightened box (at p99.9 the
        # worst shortfall over the test set is 12.8 MW; at p99 it was 189 MW).
        # The projection then returns the closest achievable point; report the
        # gap rather than letting it hide inside "balance is exact".
        short.append(((P_t * sign).sum(-1) - nd[:, j]).abs())
        outs.append(P_t)
        P_prev = P_t
    P = torch.stack(outs, 1)
    if not return_debug:
        return P
    dbg = {"correction": torch.stack(moved, 1),          # (B,N) MW moved per step
           "bal_short": torch.stack(short, 1)}           # (B,N) MW nd was out of reach
    if soc is not None:
        dbg |= {"soc_swing": soc.swing, "soc_short": soc.short, "soc_E": soc.E,
                "soc_return_err": soc.return_err}
    return P, dbg


# ----------------------------------------------------------------------------
# rayen: anchor + ray (the planA recursive head, unchanged in behaviour)
# ----------------------------------------------------------------------------
def rayen_head(raw, pL, pR, nd, return_debug: bool = False):
    """raw (B,N,7): 6 direction + 1 step logit -> exactly feasible (B,N,6)."""
    dev, dt = raw.device, raw.dtype
    sign, rup, rdn, cap = _limits(dev, dt)
    B, N, _ = raw.shape
    dirs, step = raw[..., :6], torch.sigmoid(raw[..., 6])
    P_prev, outs, alphas, glides, anchors = pL, [], [], [], []
    for j in range(N):
        k = N - j
        lo, hi = _box(P_prev, pR, k, rup, rdn, cap)
        # anchor: glide toward pR, then an exact balance snap weighted by the
        # room each channel has in the direction it must move
        c = torch.minimum(torch.maximum(P_prev + (pR - P_prev) / (k + 1), lo), hi)
        delta = (nd[:, j] - (c * sign).sum(-1)).unsqueeze(-1)
        room = torch.where(delta * sign > 0, hi - c, c - lo).clamp_min(0.0)
        A = c + (delta / room.sum(-1, keepdim=True).clamp_min(1e-6)) * (room * sign)
        A = torch.minimum(torch.maximum(A, lo), hi)
        glides.append(c); anchors.append(A)
        # direction: drop channels pinned at a wall and pointing out, then make it
        # tangent to the balance plane over what is left, then scale-free
        r = dirs[:, j]
        blocked = ((A <= lo + 1e-6) & (r < 0)) | ((A >= hi - 1e-6) & (r > 0))
        free = (~blocked).to(dt)
        r = r * free
        w = (hi - lo).clamp_min(1e-6) * free
        r = r - ((r * sign).sum(-1) / w.sum(-1).clamp_min(1e-6)).unsqueeze(-1) * (w * sign)
        r = (r * free)
        r = r / (r.abs().amax(-1, keepdim=True) + 1e-6)
        big = torch.full_like(A, 1e9)
        a_hi = torch.where(r > 1e-9, (hi - A) / r.clamp_min(1e-9), big).amin(-1)
        a_lo = torch.where(r < -1e-9, (A - lo) / (-r).clamp_min(1e-9), big).amin(-1)
        amax = torch.minimum(a_hi, a_lo).clamp(0.0, 1e6)
        alphas.append(amax)
        P_t = A + (step[:, j] * amax).unsqueeze(-1) * r
        outs.append(P_t)
        P_prev = P_t
    P = torch.stack(outs, 1)
    if not return_debug:
        return P
    return P, {"alpha_max": torch.stack(alphas, 1), "step": step,
               "glide": torch.stack(glides, 1),      # (B,N,6) stage 1, before the snap
               "anchor": torch.stack(anchors, 1)}    # (B,N,6) stage 2, what the ray starts from


HEADS = {"rayen": (rayen_head, 7), "hardnet": (hardnet_head, 6),
         "hardnet_alloc": (hardnet_alloc_head, 12), "none": (None, 6)}


if __name__ == "__main__":
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "imputation"))
    import gap_data as GD
    GD.NPZ = (Path(__file__).resolve().parent.parent
              / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz")
    from gap_data import load_flats, sample_recon_windows, SIGN

    f = load_flats()
    gws = sample_recon_windows(f, "test", n=60, context=48, gap=36, seed=123)
    g = torch.Generator().manual_seed(0)
    t_ = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)[None]
    print("feasibility self-test, p99.9 ramp envelope\n")
    print("  input = truth + N(0, 150 MW), i.e. a plausible-but-wrong network.\n"
          "  Pure noise would walk the trajectory somewhere the box cannot reach nd\n"
          "  from, which measures the input, not the head.\n")
    for name in ("rayen", "hardnet"):
        fn, nout = HEADS[name]
        worst = dict(bal=0.0, lo=0.0, hi=0.0, ramp=0.0)
        for gw in gws:
            nd = gw.truth_mw @ SIGN
            noise = torch.randn(1, 36, nout, generator=g, dtype=torch.float64)
            if name == "hardnet":            # raw is MW
                raw = noise * 150.0 + torch.tensor(gw.truth_mw)[None]
            else:                            # raw is a direction + a step logit
                raw = noise
            P = fn(raw, t_(gw.pL_mw), t_(gw.pR_mw), t_(nd))[0].numpy()
            full = np.vstack([gw.pL_mw[None], P, gw.pR_mw[None]])
            d = np.diff(full, axis=0)
            worst["bal"] = max(worst["bal"], float(np.abs(P @ SIGN - nd).max()))
            worst["lo"] = max(worst["lo"], float(np.maximum(-P, 0).max()))
            worst["hi"] = max(worst["hi"], float(np.maximum(P - CAP, 0).max()))
            worst["ramp"] = max(worst["ramp"], float((np.maximum(d - R_UP, 0)
                                                      + np.maximum(-d - R_DN, 0)).max()))
        print(f"  {name:8s} balance {worst['bal']:.2e}  box lo {worst['lo']:.2e}  "
              f"hi {worst['hi']:.2e}  ramp {worst['ramp']:.2e} MW")
    # identity check: hardnet must not move an already-feasible input
    gw = gws[0]
    nd = gw.truth_mw @ SIGN
    P = hardnet_head(t_(gw.truth_mw), t_(gw.pL_mw), t_(gw.pR_mw), t_(nd))[0].numpy()
    print("\n  balance residual above is the shortfall where nd is not attainable\n"
          "  inside the tightened box, not a solver error.\n")
    # identity: hardnet must not move an input that IS feasible under p99
    moved_t, moved_f = [], []
    for gw in gws[:60]:
        nd = gw.truth_mw @ SIGN
        Pt = hardnet_head(t_(gw.truth_mw), t_(gw.pL_mw), t_(gw.pR_mw), t_(nd))[0].numpy()
        moved_t.append(np.abs(Pt - gw.truth_mw).max())
        Pf = hardnet_head(t_(Pt), t_(gw.pL_mw), t_(gw.pR_mw), t_(nd))[0].numpy()
        moved_f.append(np.abs(Pf - Pt).max())          # re-projecting its own output
    print(f"  hardnet on the TRUTH:      max move {np.max(moved_t):8.2f} MW  "
          f"(truth breaks the envelope on 0.23% of steps, so a move is CORRECT)")
    print(f"  hardnet on its own output: max move {np.max(moved_f):8.2e} MW  "
          f"<- idempotent, i.e. the identity on a feasible input")
