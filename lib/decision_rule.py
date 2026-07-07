"""Decision-rules hard-constraint framework (Constante Flores, Chen & Li 2025,
arXiv:2505.13858 -- hardlinearconstraints.pdf; see plan.md, approach 2).

Two subnetworks blended per prediction:

  safe network   f_SN(x) = F x, a fixed linear rule found ONCE by the LP (14)
                 (jointly-linear case: A input-independent, b(x) = B x). F is
                 guaranteed to satisfy the balance equality and every ramp /
                 floor inequality for every x in the input polytope X.
  task network   any 7-output backbone (D_t, P_1..P_6) trained for accuracy
                 as usual; its raw output is snapped onto the balance plane
                 with the closed-form orthogonal projection (appendix A,
                 eq. 16), then blended:

  y = (1 - alpha) y_TN + alpha y_SN,   alpha = max_{i: s_TN_i < 0}
      -s_TN_i / (s_SN_i - s_TN_i)      (eq. 4; alpha = 0 if nothing violated)

Constraint set (same as RayenHead so the two approaches are apples-to-apples):
  balance   SIGN . P - D_t = 0                    (a . y = 0, a = (-1, SIGN))
  ramps     -R_dn_i <= P_i - P_{i,t-1} <= R_up_i  (asymmetric, per generator)
  floors    P_i >= 0                              (include_floor=True)

The safe-net input is x = [1, P_{1,t-1}..P_{6,t-1}, nd_{t-1}] (k = 8), read
off the last step of the model's input window, so b(x) = B x is exactly the
ramp right-hand side the blend must respect.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SIGN = np.array([1., 1., 1., 1., -1., 1.])
A_PLANE = np.concatenate([[-1.], SIGN])          # normal of SIGN.P - D = 0
N_OUT, K_IN = 7, 8


def build_constraints(ramp_up, ramp_dn, include_floor: bool = True):
    """Inequality rows A y <= b(x) = B x, standard form s(x) = Bx - Ay >= 0.

    Rows: 6 ramp-up, 6 ramp-down, then (optionally) 6 floors.
    A: (m, 7) on y = (D, P1..P6);  B: (m, 8) on x = [1, P_prev(6), nd_prev].
    """
    ramp_up, ramp_dn = np.asarray(ramp_up, float), np.asarray(ramp_dn, float)
    rows_a, rows_b = [], []
    for i in range(6):
        up = np.zeros(N_OUT); up[1 + i] = 1.0                 # P_i <= prev_i + R_up
        bu = np.zeros(K_IN); bu[0] = ramp_up[i]; bu[1 + i] = 1.0
        rows_a.append(up); rows_b.append(bu)
    for i in range(6):
        dn = np.zeros(N_OUT); dn[1 + i] = -1.0                # -P_i <= R_dn - prev_i
        bd = np.zeros(K_IN); bd[0] = ramp_dn[i]; bd[1 + i] = -1.0
        rows_a.append(dn); rows_b.append(bd)
    if include_floor:
        for i in range(6):
            fl = np.zeros(N_OUT); fl[1 + i] = -1.0            # -P_i <= 0
            rows_a.append(fl); rows_b.append(np.zeros(K_IN))
    return np.array(rows_a), np.array(rows_b)


def input_polytope(p_hi, nd_lo: float, nd_hi: float):
    """X = {x : P x >= p}: x_0 == 1 (two rows), 0 <= P_prev_i <= p_hi_i,
    nd_lo <= nd_prev <= nd_hi. Returns (P, p)."""
    p_hi = np.asarray(p_hi, float)
    P, p = [], []
    e = np.eye(K_IN)
    P += [e[0], -e[0]]; p += [1.0, -1.0]
    for i in range(6):
        P += [e[1 + i], -e[1 + i]]; p += [0.0, -p_hi[i]]
    P += [e[7], -e[7]]; p += [nd_lo, -nd_hi]
    return np.array(P), np.array(p)


def fit_safe_F(ramp_up, ramp_dn, p_hi, nd_lo, nd_hi,
               include_floor: bool = True, mc_samples: int = 20000,
               seed: int = 0) -> dict:
    """Solve LP (14) for the safe rule F (robust-feasible over all of X).

    Per inequality row l, Proposition 2 dualises "s_l(x) = (B_l - A_l F) x >= t
    for all x in X" into: exists lambda_l >= 0 with P^T lambda_l = (B_l - A_l F)^T
    and p^T lambda_l >= t. The balance equality holds identically via a^T F = 0
    (B_eq = 0). Maximises t = the worst-case slack; t* > 0 means F is strictly
    interior everywhere, so blend denominators are bounded away from 0.
    """
    from scipy.optimize import linprog

    A, B = build_constraints(ramp_up, ramp_dn, include_floor)
    P, p = input_polytope(p_hi, nd_lo, nd_hi)
    m, l = A.shape[0], P.shape[0]
    nF, nL = N_OUT * K_IN, m * l                      # F row-major, Lambda row-major
    nv = nF + nL + 1                                  # + t (last)

    def fidx(i, j): return i * K_IN + j
    def lidx(ln, r): return nF + ln * l + r

    eq_rows, eq_rhs = [], []
    for j in range(K_IN):                             # a^T F = 0, column j
        row = np.zeros(nv)
        for i in range(N_OUT):
            row[fidx(i, j)] = A_PLANE[i]
        eq_rows.append(row); eq_rhs.append(0.0)
    for ln in range(m):                               # A_l F + lambda_l^T P = B_l
        for j in range(K_IN):
            row = np.zeros(nv)
            for i in range(N_OUT):
                row[fidx(i, j)] = A[ln, i]
            for r in range(l):
                row[lidx(ln, r)] = P[r, j]
            eq_rows.append(row); eq_rhs.append(B[ln, j])

    ub_rows, ub_rhs = [], []
    for ln in range(m):                               # t - p^T lambda_l <= 0
        row = np.zeros(nv); row[-1] = 1.0
        for r in range(l):
            row[lidx(ln, r)] = -p[r]
        ub_rows.append(row); ub_rhs.append(0.0)

    c = np.zeros(nv); c[-1] = -1.0                    # maximise t
    bounds = [(None, None)] * nF + [(0, None)] * nL + [(None, None)]
    res = linprog(c, A_ub=np.array(ub_rows), b_ub=np.array(ub_rhs),
                  A_eq=np.array(eq_rows), b_eq=np.array(eq_rhs),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"safe-network LP failed: {res.message}")
    F = res.x[:nF].reshape(N_OUT, K_IN)
    t_star = float(-res.fun)

    # Monte-Carlo audit: worst slack / equality residual of Fx over sampled X
    rng = np.random.default_rng(seed)
    xs = np.empty((mc_samples, K_IN)); xs[:, 0] = 1.0
    xs[:, 1:7] = rng.uniform(0.0, np.asarray(p_hi, float), size=(mc_samples, 6))
    xs[:, 7] = rng.uniform(nd_lo, nd_hi, size=mc_samples)
    y = xs @ F.T
    slack = xs @ B.T - y @ A.T
    return {"F": F, "t": t_star,
            "mc_min_slack": float(slack.min()),
            "mc_eq_max": float(np.abs(y @ A_PLANE).max()),
            "include_floor": include_floor,
            "p_hi": list(map(float, p_hi)), "nd_lo": float(nd_lo),
            "nd_hi": float(nd_hi)}


class DecisionRuleHead(nn.Module):
    """Wraps a trained 7-output task net; applies projection + safe blend.

    base(x) -> (B, 7) scaled: [:, 0] = D_t in x-scaler net_demand units,
    [:, 1:] = dispatch in y-scaler units (the y7 supervision layout in
    ml/train_hist.py). Output uses the same layout, so it drops into the
    RayenHead evaluation path unchanged. Inference-only by design (post-hoc,
    task net "trained for accuracy as usual"), but the ops are differentiable
    a.e. if a trained-through arm is ever wanted.

    Per forward pass, `last_alpha` (B,) and `last_sn_min` (B,) are stashed:
    alpha = blend weight actually used; sn_min = the safe point's worst slack
    (< 0 flags an input outside the certified polytope X -- count these).
    """

    def __init__(self, base: nn.Module, F: np.ndarray, x_mean, x_scale,
                 y_mean, y_scale, ramp_up, ramp_dn, nd_feat_idx: int = 6,
                 include_floor: bool = True, target_feat_idx=None):
        super().__init__()
        self.base = base
        self.nd_feat_idx = nd_feat_idx
        if target_feat_idx is None:
            from models import TARGET_FEAT_IDX as target_feat_idx
        A, B = build_constraints(ramp_up, ramp_dn, include_floor)
        self.register_buffer("_A", torch.tensor(A, dtype=torch.float32))
        self.register_buffer("_B", torch.tensor(B, dtype=torch.float32))
        self.register_buffer("_F", torch.tensor(np.asarray(F), dtype=torch.float32))
        self.register_buffer("_a", torch.tensor(A_PLANE, dtype=torch.float32))
        self.register_buffer("_feat_idx", torch.tensor(list(target_feat_idx), dtype=torch.long))
        self.register_buffer("_x_mean", torch.tensor(np.asarray(x_mean)[list(target_feat_idx)], dtype=torch.float32))
        self.register_buffer("_x_scale", torch.tensor(np.asarray(x_scale)[list(target_feat_idx)], dtype=torch.float32))
        self.register_buffer("_nd_mean", torch.tensor(float(np.asarray(x_mean)[nd_feat_idx])))
        self.register_buffer("_nd_scale", torch.tensor(float(np.asarray(x_scale)[nd_feat_idx])))
        self.register_buffer("_y_mean", torch.tensor(np.asarray(y_mean), dtype=torch.float32))
        self.register_buffer("_y_scale", torch.tensor(np.asarray(y_scale), dtype=torch.float32))
        self.last_alpha: torch.Tensor | None = None
        self.last_sn_min: torch.Tensor | None = None

    def forward(self, x):
        raw = self.base(x)                                       # (B, 7) scaled
        d_mw = raw[:, :1] * self._nd_scale + self._nd_mean
        p_mw = raw[:, 1:] * self._y_scale + self._y_mean
        y = torch.cat([d_mw, p_mw], dim=1)                       # (B, 7) MW
        y = y - (y @ self._a).unsqueeze(-1) * self._a / (self._a @ self._a)

        prev_mw = x[:, -1, :].index_select(1, self._feat_idx) * self._x_scale + self._x_mean
        nd_mw = (x[:, -1, self.nd_feat_idx] * self._nd_scale + self._nd_mean).unsqueeze(-1)
        xdr = torch.cat([torch.ones_like(nd_mw), prev_mw, nd_mw], dim=1)  # (B, 8)

        b = xdr @ self._B.T                                      # (B, m)
        s_tn = b - y @ self._A.T
        safe = xdr @ self._F.T                                   # (B, 7)
        s_sn = b - safe @ self._A.T
        ratio = (-s_tn) / (s_sn - s_tn).clamp_min(1e-9)
        alpha = torch.where(s_tn < 0, ratio, torch.zeros_like(ratio)) \
            .amax(dim=1).clamp(0.0, 1.0).unsqueeze(-1)           # (B, 1)
        yb = (1 - alpha) * y + alpha * safe

        self.last_alpha = alpha.detach().squeeze(-1)
        self.last_sn_min = s_sn.detach().amin(dim=1)
        d_scaled = (yb[:, :1] - self._nd_mean) / self._nd_scale
        gen_scaled = (yb[:, 1:] - self._y_mean) / self._y_scale
        return torch.cat([d_scaled, gen_scaled], dim=1)          # (B, 7)
