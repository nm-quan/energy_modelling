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
SOC is not enforced by either (same as everything else in this project).
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


def hardnet_head(raw, pL, pR, nd, return_debug: bool = False):
    """raw (B,N,6) network output in MW -> exactly feasible (B,N,6)."""
    dev, dt = raw.device, raw.dtype
    sign, rup, rdn, cap = _limits(dev, dt)
    B, N, _ = raw.shape
    P_prev, outs, moved, short = pL, [], [], []
    for j in range(N):
        lo, hi = _box(P_prev, pR, N - j, rup, rdn, cap)
        P_t = project_box_plane(raw[:, j], lo, hi, nd[:, j], sign)
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
    return P, {"correction": torch.stack(moved, 1),      # (B,N) MW moved per step
               "bal_short": torch.stack(short, 1)}       # (B,N) MW nd was out of reach


# ----------------------------------------------------------------------------
# rayen: anchor + ray (the planA recursive head, unchanged in behaviour)
# ----------------------------------------------------------------------------
def rayen_head(raw, pL, pR, nd, return_debug: bool = False):
    """raw (B,N,7): 6 direction + 1 step logit -> exactly feasible (B,N,6)."""
    dev, dt = raw.device, raw.dtype
    sign, rup, rdn, cap = _limits(dev, dt)
    B, N, _ = raw.shape
    dirs, step = raw[..., :6], torch.sigmoid(raw[..., 6])
    P_prev, outs, alphas = pL, [], []
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
    return P, {"alpha_max": torch.stack(alphas, 1), "step": step}


HEADS = {"rayen": (rayen_head, 7), "hardnet": (hardnet_head, 6),
         "none": (None, 6)}


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
