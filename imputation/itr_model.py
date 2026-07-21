"""Bidirectional iTransformer imputer (fixed architecture for the ablation).

SAITS-style ingredients: the input is (values with the gap dispatch zeroed) ⊕
(observed-mask channel), and training can add the SAITS aux reconstruction term
(reconstruct the OBSERVED dispatch too, weight 0.1) next to the masked-imputation
term. iTransformer ingredient: channels are the tokens -- each of the 18 input
channels embeds its whole T=288 series with one shared Linear(T -> d), a channel-
identity embedding is added, full (non-causal => bidirectional) self-attention
mixes the channels, and the 6 dispatch tokens decode back to T steps.

Also here: the numpy-side constraint maps used at eval/deploy time
(raw / posthoc Π / RAYEN ray-shoot / Option-A free-endpoint Π).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import constraints as C                                                       # noqa: E402
from constraint_layers import rayen_traj_project                              # noqa: E402
from gap_data import TARGET_FEAT_IDX                                          # noqa: E402
from shift_feasibility import free_boundary_project                           # noqa: E402

N_TOKENS = 18                     # 17 features + 1 observed-mask channel


class ITransformerImputer(nn.Module):
    def __init__(self, T=288, d_model=128, n_layers=3, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.T = T
        self.embed = nn.Linear(T, d_model)                       # shared per-channel series embed
        self.tok = nn.Parameter(torch.zeros(N_TOKENS, d_model))  # channel identity
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, T)                        # per dispatch token -> series
        self.tgt_tok = list(TARGET_FEAT_IDX)                     # dispatch tokens, TARGETS order

    def forward(self, x, mask):
        """x (B,T,17) z-scored (gap dispatch zeroed), mask (B,T,1) observed=1.
        Returns (B,T,6) y-scaled reconstruction of the 6 dispatch channels."""
        tokens = torch.cat([x, mask], dim=-1).transpose(1, 2)    # (B,18,T)
        h = self.encoder(self.embed(tokens) + self.tok)          # (B,18,d)
        return self.head(h[:, self.tgt_tok]).transpose(1, 2)    # (B,T,6)


def gap_slice(t, g0, glen):
    """Extract the (B,glen,6) gap trajectory from a (B,T,6) tensor/array with
    per-window gap starts g0 (B,)."""
    B = t.shape[0]
    if torch.is_tensor(t):
        pos = torch.as_tensor(g0, device=t.device)[:, None] + torch.arange(glen, device=t.device)
        return t[torch.arange(B, device=t.device)[:, None], pos]
    pos = np.asarray(g0)[:, None] + np.arange(glen)[None, :]
    return t[np.arange(B)[:, None], pos]


def apply_gap_map(name, fill_mw, g0, glen, pL, pR, nd):
    """Numpy eval-time constraint map on the gap portion of a (B,T,6) MW fill.
      raw    identity (C0/C1/causal rows -- violations are the point)
      proj   posthoc Π: cyclic projection pinned to the known boundaries (C2)
      rayen  the RAYEN ray-shoot, the map C3 trained through
      free   Option-A Π: no pinned boundaries (whole-day counterfactual deploy)
    Returns the fill with the gap replaced by the mapped trajectory."""
    if name == "raw":
        return fill_mw
    out = fill_mw.copy()
    G = gap_slice(fill_mw, g0, glen)
    if name == "proj":
        mapped = np.stack([C.project_gap(G[k], pL[k], pR[k], nd[k])[0] for k in range(len(G))])
    elif name == "rayen":
        with torch.no_grad():
            mapped = rayen_traj_project(torch.tensor(G), torch.tensor(pL),
                                        torch.tensor(pR), torch.tensor(nd)).numpy()
    elif name == "free":
        with torch.no_grad():
            mapped = free_boundary_project(torch.tensor(G), torch.tensor(nd)).numpy()
    else:
        raise ValueError(name)
    pos = np.asarray(g0)[:, None] + np.arange(glen)[None, :]
    out[np.arange(len(G))[:, None], pos] = mapped
    return out


def soft_penalty(P_gap_mw, nd, pL, pR):
    """Differentiable feasibility penalty for the C1 arm, on the (B,N,6) MW gap
    trajectory: balance + negativity + two-sided ramp overshoot (incl. both
    seams), each normalized to comparable O(1) units. One lambda scales the sum."""
    sign = C._SIGN_T.to(P_gap_mw.device, P_gap_mw.dtype)
    rup = C._RUP_T.to(P_gap_mw.device, P_gap_mw.dtype)
    rdn = C._RDN_T.to(P_gap_mw.device, P_gap_mw.dtype)
    bal = ((P_gap_mw * sign).sum(-1) - nd).abs().mean() / 1000.0
    neg = torch.relu(-P_gap_mw).mean() / 100.0
    full = torch.cat([pL[:, None], P_gap_mw, pR[:, None]], dim=1)
    d = torch.diff(full, dim=1)
    ramp = (torch.relu(d - rup) + torch.relu(-d - rdn)).mean() / 100.0
    return bal + neg + ramp
