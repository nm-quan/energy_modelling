"""SELF-CONTAINED reference for the constraint machinery of the `size_aware`
BiLSTM gap imputer. Written to be READ, not imported by the pipeline.

Everything the deployed path does is re-derived here from scratch, in one file,
with no repo imports for the constraint math. Companion document:
imputation/sizeaware_constraints.tex (equation numbers referenced below).

WHAT THE REAL CODE IS
    constants ................ ml/check_caps.py            (CAPS, RAMPS, BATT_CAP_MWH)
    channel order + signs .... imputation/gap_data.py      (TARGETS, SIGN)
    projection operators ..... imputation/constraints.py   (_balance_project, _ramp_project,
                                                            _soc_project, cyclic_project)
    RAYEN ray-shot ........... imputation/constraint_layers.py  (rayen_traj_project)
    network + loss ........... imputation/model.py         (BiLSTMImputer, masked_loss)
    training driver .......... imputation/plan1_train.py   (--arm size_aware)
    eval harness ............. imputation/plan1_bench.py

THE PIPELINE IN THREE LINES
    F = (interp(pL, pR) + BiLSTM(x, mask)) * y_scale + y_mean     # raw fill, MW, infeasible
    P = rayen_shot(F, pL, pR, nd, size_aware=True)                # <-- the constraint layer
    loss = MSE(P, truth)                                          # trained THROUGH the layer

Run this file to (a) verify the reference reproduces the repo implementation
bit-for-bit on real test windows, and (b) print the alpha* diagnostics.

    python3 imputation/sizeaware_constraints_reference.py
"""
from __future__ import annotations

import numpy as np
import torch

# =============================================================================
# 0. CONSTANTS  (mirrors of ml/check_caps.py + gap_data.py, inlined on purpose)
# =============================================================================
# Channel order is fixed everywhere in the repo by gap_data.TARGETS:
TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]

# Balance signs: battery CHARGING is a load, so it enters the sum negatively.
# Note s_i^2 == 1 for every channel -- the whole size-aware algebra leans on this.
SIGN = np.array([1.0, 1.0, 1.0, 1.0, -1.0, 1.0])

# Per-channel cap (MW) and asymmetric 5-min ramp limits (MW/step), both stored
# POSITIVE. Full-history empirical maxima, rounded outward -> 0 historical
# violations. Canon lives in ml/check_caps.py.
CAP   = np.array([2168.0, 4895.8,  516.4, 1748.6, 1611.6, 1687.5])
R_UP  = np.array([ 956.5,  333.4,   82.5,  400.3,  795.0,  715.1])
R_DN  = np.array([ 735.4, 1553.6,  499.9,  363.9,  863.1,  670.9])   # magnitudes

CHG_IDX, DIS_IDX = 4, 5                  # battery channels within TARGETS
BATT_CAP_MWH = 4735.75                   # nameplate VIC fleet storage
ETA = float(np.sqrt(0.834))              # per-side efficiency (round trip 0.834)
DT = 5.0 / 60.0                          # hours per 5-min step
SOC_MARGIN_MWH = 100.0                   # headroom taken off BOTH ends
CAP_EFF = BATT_CAP_MWH - 2.0 * SOC_MARGIN_MWH      # 4535.75 MWh usable swing

# --- float32 quirk of the deployed code, reproduced here on purpose ----------
# constraints.py caches the limits as MODULE-LEVEL float32 tensors:
#     _CAP_T = torch.tensor(CAP, dtype=torch.float32)     # likewise _RUP_T/_RDN_T
# and every consumer does `.to(dev, dt)`. Upcasting f32 -> f64 does NOT recover the
# decimals, so even the "float64 posthoc" path runs with caps/ramps that differ
# from the declared values by up to 1.95e-4 MW (worst: coal_brown cap
# 4895.8 -> 4895.7998046875). Physically irrelevant on a 4.9 GW cap, but it is the
# entire reason a clean f64 reimplementation disagrees with the repo at ~1e-4 MW.
# Set MATCH_REPO_F32 = False to use the true float64 constants instead.
MATCH_REPO_F32 = True
if MATCH_REPO_F32:
    _f32 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float64)
    CAP, R_UP, R_DN = _f32(CAP), _f32(R_UP), _f32(R_DN)


# =============================================================================
# 1. THE FEASIBLE SET  (tex eq. 1)
# -----------------------------------------------------------------------------
# For a gap of N steps with pinned boundaries pL (step 0) and pR (step N+1):
#
#   balance   SIGN . P_t == nd_t                       for t = 1..N
#   box       0 <= P_ti <= CAP_i
#   ramp      -R_DN_i <= P_ti - P_{t-1,i} <= R_UP_i    for t = 1..N+1, i.e. INCLUDING
#             the entry seam pL -> P_1 and the exit seam P_N -> pR
#   SOC       max_t E_t - min_t E_t <= CAP_EFF,  E = cumsum((eta*chg - dis/eta)*DT)
#             (swing only: absolute state of charge at gap open is unknown)
#
# All four are convex; all four are polyhedral (SOC expands to the finite family
# of pairwise inequalities E_t - E_t' <= CAP_EFF).
#
# WHICH nd?  Two different quantities are called "net demand" in this project:
#   - the MODEL INPUT is demand-side:  demand - wind - solar - curtailments
#     (feature col 6, rebuilt by plan1_data.py)
#   - the CONSTRAINT TARGET is supply-side: nd_t = SIGN . P_true_t
#     (plan1_train.supply_nd; plan1_bench uses gw.truth_mw @ SIGN)
# They differ by the interconnector, mean +-500 MW, up to 4.9 GW. Balancing six
# fuels to the demand-side value is infeasible on many windows, so the map uses
# the supply-side sum, which is attainable by construction.
# =============================================================================


def check_feasible(P, pL, pR, nd, soc=True):
    """Report the violation magnitude of every constraint for one (N,6) MW window."""
    P = np.asarray(P, dtype=np.float64)
    full = np.vstack([pL[None], P, pR[None]])
    d = np.diff(full, axis=0)
    E = np.concatenate([[0.0], np.cumsum(
        (np.clip(P[:, CHG_IDX], 0, None) * ETA
         - np.clip(P[:, DIS_IDX], 0, None) / ETA) * DT)])
    out = {
        "balance_max_mw": float(np.abs(P @ SIGN - nd).max()),
        "box_lo_max_mw":  float(np.maximum(-P, 0).max()),
        "box_hi_max_mw":  float(np.maximum(P - CAP, 0).max()),
        "ramp_max_mw":    float((np.maximum(d - R_UP, 0)
                                 + np.maximum(-d - R_DN, 0)).max()),
    }
    if soc:
        out["soc_swing_mwh"] = float(E.max() - E.min())
        out["soc_over_mwh"] = float(max(0.0, E.max() - E.min() - CAP_EFF))
    return out


# =============================================================================
# 2. THE ANCHOR: cyclic projection (POCS)                       (tex eq. 6-13)
# -----------------------------------------------------------------------------
# The RAYEN shot needs a FEASIBLE ORIGIN. It is made by projecting the straight
# interpolation between the pinned boundaries onto the feasible set, by cycling
#
#       A <- soc( box( ramp( balance(A) ) ) )       M times
#
# The anchor depends only on pL, pR, nd -- NOT on the network -- so it carries no
# gradient. Repo: constraints.cyclic_project.
# =============================================================================


def interp_skeleton(pL, pR, N):
    """(N,6) straight line between the pinned boundaries; tex eq. 5.
    tau_t = t/(N+1), so the line lands ON pL at t=0 and pR at t=N+1."""
    tau = (torch.arange(1, N + 1, device=pL.device, dtype=pL.dtype) / (N + 1)).view(N, 1)
    return pL[None, :] + tau * (pR - pL)[None, :]


def balance_project(P, nd, size_aware):
    """EXACT per-step balance snap; tex eq. 8-10. THIS is the size-aware operator.

    Residual  delta_t = nd_t - SIGN . P_t  is absorbed by moving along w * SIGN:

        P_t <- P_t + (delta_t / sum_i w_ti) * (w_t * SIGN)

    Exact for ANY strictly-positive w, because sum_i w_ti * SIGN_i^2 = sum_i w_ti:

        SIGN . P_new = SIGN . P + delta_t * (sum w * SIGN^2)/(sum w) = nd_t

    The two arms differ ONLY in w:

        size_aware=False   w_ti = 1              -> P_t + (delta_t/6)*SIGN
                           every channel moves the SAME MW. A 600 MW residual
                           dumps 100 MW onto gas_steam, whose typical output is
                           single-digit MW. This is the failure being fixed.

        size_aware=True    w_ti = |P_ti| + 1     -> correction proportional to level
                           coal_brown (thousands of MW) absorbs the bulk;
                           gas_steam moves a proportionate sliver. The +1 MW floor
                           keeps w > 0 so the identity above survives P_ti == 0.

    Geometrically this is the OBLIQUE projection onto {SIGN.x = nd_t} along
    w*SIGN. Only w == 1 is the orthogonal / minimum-norm projection; the
    size-aware variant trades Euclidean minimality for keeping the mix ratios.
    """
    sign = torch.as_tensor(SIGN, device=P.device, dtype=P.dtype)
    resid = nd - (P * sign).sum(-1)                        # (N,)  delta_t
    if size_aware:
        w = P.abs() + 1.0                                  # (N,6)  level-proportional
        return P + (resid / w.sum(-1).clamp_min(1e-6)).unsqueeze(-1) * (w * sign)
    return P + (resid / (sign * sign).sum()).unsqueeze(-1) * sign      # /6


def ramp_project(P, pL, pR):
    """One forward sweep from pL, then one backward sweep from pR; tex eq. 11-12.

    forward :  P_t <- clip(P_t,  P_{t-1} - R_DN,  P_{t-1} + R_UP)     t = 1..N
    backward:  P_t <- clip(P_t,  P_{t+1} - R_UP,  P_{t+1} + R_DN)     t = N..1

    The R_UP/R_DN swap in the backward pass is deliberate: the constraint
    P_{t+1} - P_t <= R_UP read backwards is P_t >= P_{t+1} - R_UP.

    ONE fwd+bwd pair is not enough on its own -- the backward sweep can re-break a
    forward bound -- which is why this sits inside the outer POCS cycle. Built
    functionally (list + stack, no in-place writes) so autograd can traverse it.
    """
    rup = torch.as_tensor(R_UP, device=P.device, dtype=P.dtype)
    rdn = torch.as_tensor(R_DN, device=P.device, dtype=P.dtype)
    N = P.shape[0]
    prev, fwd = pL, []
    for t in range(N):
        x = torch.maximum(torch.minimum(P[t], prev + rup), prev - rdn)
        fwd.append(x); prev = x
    nxt, back = pR, [None] * N
    for t in range(N - 1, -1, -1):
        x = torch.maximum(torch.minimum(fwd[t], nxt + rdn), nxt - rup)
        back[t] = x; nxt = x
    return torch.stack(back, dim=0)


def box_project(P):
    """tex eq. 14."""
    cap = torch.as_tensor(CAP, device=P.device, dtype=P.dtype)
    return torch.minimum(P.clamp_min(0.0), cap)


def soc_project(P):
    """Scale BOTH battery channels by one factor so the swing fits; tex eq. 15.
    E is linear in the battery channels, so scaling them by sigma scales the swing
    by sigma. Identity whenever SOC is slack. NOT USED on the plan1 path (see
    rayen_shot's soc flag)."""
    chg = P[:, CHG_IDX].clamp_min(0.0)
    dis = P[:, DIS_IDX].clamp_min(0.0)
    E = torch.cat([torch.zeros(1, device=P.device, dtype=P.dtype),
                   torch.cumsum((chg * ETA - dis / ETA) * DT, 0)])
    swing = E.max() - E.min()
    sigma = torch.clamp(CAP_EFF / (swing + 1e-6), max=1.0)
    return torch.cat([P[:, :CHG_IDX], P[:, CHG_IDX:DIS_IDX + 1] * sigma], dim=-1)


def build_anchor(pL, pR, nd, iters, size_aware, soc=False):
    """A = POCS(interp) -> feasible, model-independent, gradient-free; tex eq. 6."""
    A = interp_skeleton(pL, pR, nd.shape[0])
    for _ in range(iters):
        A = balance_project(A, nd, size_aware)
        A = ramp_project(A, pL, pR)
        A = box_project(A)
        if soc:
            A = soc_project(A)
    return A


# =============================================================================
# 3. THE RAYEN RAY-SHOT -- the actual constraint layer          (tex eq. 16-22)
# -----------------------------------------------------------------------------
# From the feasible anchor A, travel toward the network's raw fill F along a
# direction that PRESERVES balance, and stop at the first inequality wall.
#
#   r  = F - A,  with its balance component removed (size-aware weighting)
#   a* = min over every box/ramp[/SOC] wall of the distance from A along r,
#        clipped to [0, 1]        <- ONE global scalar for all N*6 cells
#   P  = A + a* r
#
# Balance holds for ANY a (r is tangent); box/ramp hold because a* is the exact
# distance to the nearest wall and every wall is affine along the ray.
# Repo: constraint_layers.rayen_traj_project.
# =============================================================================


def rayen_shot(F, pL, pR, nd, size_aware=True, anchor_iters=160, soc=False,
               anchor=None, return_debug=False):
    """The deployed constraint layer for one (N,6) MW window. Differentiable in F.

    size_aware=True is the `size_aware` arm; it changes TWO things at once:
      (a) the anchor's balance operator (build_anchor -> balance_project), and
      (b) the tangent removal below.
    Both use w = |.| + 1 instead of w = 1.

    soc defaults to False because that is what plan1_train.py and plan1_bench.py
    actually run (rayen_traj_project's own default). SOC is therefore NOT enforced
    for this arm -- only measured afterwards, where it turns out to be slack on
    every 3h test window.
    """
    sign = torch.as_tensor(SIGN, device=F.device, dtype=F.dtype)
    rup = torch.as_tensor(R_UP, device=F.device, dtype=F.dtype)
    rdn = torch.as_tensor(R_DN, device=F.device, dtype=F.dtype)
    cap = torch.as_tensor(CAP, device=F.device, dtype=F.dtype)
    N = F.shape[0]

    # ---- 3a. feasible anchor (no gradient) ----------------------------------
    A = build_anchor(pL, pR, nd, anchor_iters, size_aware, soc) if anchor is None else anchor

    # ---- 3b. balance-tangent direction; tex eq. 16 --------------------------
    # Same weighting rule as balance_project, but w is built from the ANCHOR, so
    # it is a CONSTANT of the graph: the removal is a fixed linear operator on r,
    # and the |.| kink never reaches the gradient.
    r = F - A
    if size_aware:
        w = A.abs() + 1.0
        r = r - ((r * sign).sum(-1) / w.sum(-1).clamp_min(1e-6)).unsqueeze(-1) * (w * sign)
    else:
        r = r - ((r * sign).sum(-1) / (sign * sign).sum()).unsqueeze(-1) * sign
    # now SIGN . r_t == 0 exactly  =>  SIGN . (A + a r)_t == SIGN . A_t for all a

    # ---- 3c. distance to every wall; tex eq. 18-20 --------------------------
    # Each wall is affine in a, so "largest admissible a" = slack / approach rate.
    BIG = torch.full_like(A, 1e9)
    a_box_hi = torch.where(r > 1e-9, (cap - A) / r.clamp_min(1e-9), BIG).amin()
    a_box_lo = torch.where(r < -1e-9, (0.0 - A) / r.clamp_max(-1e-9), BIG).amin()

    # Ramp walls on the N+1 consecutive differences. Padding r with ZERO rows at
    # both ends encodes "the shot does not move the pinned boundaries", which is
    # what makes both seams enforced rather than just interior steps.
    z = torch.zeros_like(pL)[None]
    dA = torch.diff(torch.cat([pL[None], A, pR[None]], 0), dim=0)      # (N+1,6)
    dr = torch.diff(torch.cat([z, r, z], 0), dim=0)                    # (N+1,6)
    BIG_R = torch.full_like(dA, 1e9)
    a_ramp_up = torch.where(dr > 1e-9, (rup - dA) / dr.clamp_min(1e-9), BIG_R).amin()
    a_ramp_dn = torch.where(dr < -1e-9, (-rdn - dA) / dr.clamp_max(-1e-9), BIG_R).amin()

    walls = {"box_hi": a_box_hi, "box_lo": a_box_lo,
             "ramp_up": a_ramp_up, "ramp_dn": a_ramp_dn}

    if soc:
        # SOC wall; tex eq. 21. E is LINEAR in the dispatch, so along the ray
        # E_t(a) = E^A_t + a E^r_t and every ordered pair gives one affine
        # inequality  dE^A + a dE^r <= CAP_EFF -- same distance-to-wall form.
        eA = (A[:, CHG_IDX] * ETA - A[:, DIS_IDX] / ETA) * DT
        er = (r[:, CHG_IDX] * ETA - r[:, DIS_IDX] / ETA) * DT
        EA = torch.cat([torch.zeros(1, device=F.device, dtype=F.dtype), eA.cumsum(0)])
        Er = torch.cat([torch.zeros(1, device=F.device, dtype=F.dtype), er.cumsum(0)])
        dEA = EA[:, None] - EA[None, :]
        dEr = Er[:, None] - Er[None, :]
        BIG_S = torch.full_like(dEA, 1e9)
        walls["soc"] = torch.where(dEr > 1e-9, (CAP_EFF - dEA) / dEr.clamp_min(1e-9),
                                   BIG_S).amin()

    # ---- 3d. the shot; tex eq. 22 -------------------------------------------
    a_raw = torch.stack(list(walls.values())).amin()
    alpha = a_raw.clamp(0.0, 1.0)          # never overshoot past the model's own fill
    P = A + alpha * r

    if not return_debug:
        return P
    return P, {
        "alpha": float(alpha),
        "alpha_raw": float(a_raw),
        "binding_wall": min(walls, key=lambda k: float(walls[k])),
        "walls": {k: float(v) for k, v in walls.items()},
        "anchor": A,
        "direction": r,
        # THE DEGENERACY (tex sec. 8.3): the anchor's last inequality step is the
        # box clamp, so A routinely sits EXACTLY on the lower face (A_ti == 0).
        # Any such cell with r_ti < 0 makes a_box_lo = 0/r = 0, hence alpha* = 0,
        # hence P = A and the network's entire contribution is discarded. One cell
        # out of N*6 = 216 is enough, because alpha* is a single global scalar.
        "anchor_zero_cells": int((A.abs() < 1e-6).sum()),
        "mean_abs_deviation_mw": float((F - A).abs().mean()),
        "mean_applied_move_mw": float(alpha) * float(r.abs().mean()),
    }


# =============================================================================
# 4. TRAINING OBJECTIVE                                          (tex eq. 23)
# -----------------------------------------------------------------------------
# The loss is computed on P (the MAPPED output), so the network trains THROUGH
# the constraints rather than being penalised for breaking them. Repo:
# model.masked_loss + plan1_train.main.
#
#   L = MSE(P_z, Y_z)  +  0.1 * balance_mse  [+ 0.1 * cost_term, `cost` arm only]
#
# NOTE the soft balance term is numerically inert here: the map already enforces
# balance to the anchor's residual. Measured over the real training runs,
# (train_loss - train_mse) is 0.0 for `baseline` and 4e-6 for `size_aware` at
# both first and last epoch. It survives only because masked_loss is shared with
# the posthoc models, where it does real work.
# =============================================================================


# =============================================================================
# 5. VERIFICATION  --  this reference vs the repo implementation, on real data
# =============================================================================
def _verify():
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT / "imputation"))
    import gap_data as GD
    GD.NPZ = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
    from gap_data import load_flats, sample_recon_windows
    from gap_data import SIGN as REPO_SIGN, TARGETS as REPO_TARGETS
    import constraints as C
    from constraint_layers import rayen_traj_project
    from model import BiLSTMImputer
    import plan1_bench as B

    # constants must match the canon exactly
    assert REPO_TARGETS == TARGETS and np.allclose(REPO_SIGN, SIGN)
    assert np.allclose(C.CAP, CAP, atol=2e-4) and np.allclose(C.R_UP, R_UP, atol=2e-4)
    assert np.allclose(C.R_DN, R_DN, atol=2e-4)
    assert np.isclose(C.ETA, ETA) and np.isclose(C.BATT_CAP_MWH, BATT_CAP_MWH)
    # the tensors the deployed code actually uses (f32-cached, see MATCH_REPO_F32)
    assert np.array_equal(C._CAP_T.to(torch.float64).numpy(), CAP) == MATCH_REPO_F32
    print(f"constants match ml/check_caps.py canon: OK  (MATCH_REPO_F32={MATCH_REPO_F32})")

    f = load_flats()
    gws = sample_recon_windows(f, "test", n=40, context=48, gap=36, seed=123)
    ckpt = ROOT / "imputation/results/plan1/size_aware.pt"
    model = BiLSTMImputer(n_features=len(f.feat_cols))
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    print(f"loaded {ckpt.name} | {len(gws)} test 3h gaps (seed 123), 21 features\n")

    worst, alphas, walls, zeros, devs, moves = 0.0, [], {}, [], [], []
    feas = {"balance_max_mw": 0.0, "box_lo_max_mw": 0.0,
            "box_hi_max_mw": 0.0, "ramp_max_mw": 0.0, "soc_over_mwh": 0.0}
    for gw in gws:
        F_np = B.fill_window(model, f, gw, "cpu").astype(np.float64)   # raw fill, MW
        nd_np = gw.truth_mw @ SIGN                                     # SUPPLY-side target
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)

        P_ref, dbg = rayen_shot(t(F_np), t(gw.pL_mw), t(gw.pR_mw), t(nd_np),
                                size_aware=True, anchor_iters=160, return_debug=True)
        with torch.no_grad():                                          # repo implementation
            P_repo = rayen_traj_project(t(F_np)[None], t(gw.pL_mw)[None], t(gw.pR_mw)[None],
                                        t(nd_np)[None], anchor_iters=160,
                                        size_aware=True)[0]
        worst = max(worst, float((P_ref - P_repo).abs().max()))

        alphas.append(dbg["alpha"])
        walls[dbg["binding_wall"]] = walls.get(dbg["binding_wall"], 0) + 1
        zeros.append(dbg["anchor_zero_cells"])
        devs.append(dbg["mean_abs_deviation_mw"]); moves.append(dbg["mean_applied_move_mw"])
        v = check_feasible(P_ref.numpy(), gw.pL_mw, gw.pR_mw, nd_np)
        for k in feas:
            feas[k] = max(feas[k], v[k])

    a = np.array(alphas)
    print(f"reference vs repo rayen_traj_project: max |diff| = {worst:.3e} MW  "
          f"({'IDENTICAL' if worst < 1e-9 else 'MISMATCH'})\n")
    print("feasibility of the mapped output, worst over all windows")
    print(f"  balance residual      {feas['balance_max_mw']:.3e} MW   "
          "(only as exact as the anchor -- see anchor_iters)")
    print(f"  box lower violation   {feas['box_lo_max_mw']:.3e} MW   (exact by construction)")
    print(f"  box upper violation   {feas['box_hi_max_mw']:.3e} MW   (exact by construction)")
    print(f"  ramp overshoot        {feas['ramp_max_mw']:.3e} MW   (exact by construction)")
    print(f"  SOC over cap          {feas['soc_over_mwh']:.3e} MWh  (NOT enforced -- slack anyway)\n")
    print("alpha* -- the fraction of the network's deviation that survives")
    print(f"  mean {a.mean():.4f}  median {np.median(a):.4f}  min {a.min():.4f}  max {a.max():.4f}")
    print(f"  alpha* == 0 (output collapses to the anchor) in {(a < 1e-9).sum()}/{len(a)} windows")
    print(f"  binding wall counts: {walls}")
    print(f"  anchor cells sitting exactly at 0: {np.mean(zeros):.1f} of {36*6} per window")
    print(f"  mean |F - A| = {np.mean(devs):.1f} MW  ->  mean applied move = {np.mean(moves):.1f} MW")


if __name__ == "__main__":
    _verify()
