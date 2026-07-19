"""Bidirectional LSTM gap imputer (professor's reframe; GRIN-family, simplest member).

Reads the whole window [ctxL | gap | ctxR] in BOTH directions, so the 14:00
boundary flows into the gap and the fill lands on it — no drift, no 2:05 seam. The
raw (B,G,6) fill (y-scaled, TARGETS order) is projected to hard feasibility by
constraints.project_gap at inference (kept OUT of the training graph — the tube is
a non-differentiable POCS; we train the raw fill with a masked reconstruction loss
and a soft balance term, then project at eval, mirroring the study's
inference-projection finding that projection is ~WAPE-free).

Input channels: the 17 features (6 source cols zeroed in the gap) + 1 mask flag.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gap_data import TARGET_FEAT_IDX


class BiLSTMImputer(nn.Module):
    def __init__(self, n_features: int = 17, n_targets: int = 6,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_features + 1, hidden, num_layers=layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden * 2, n_targets)
        self.register_buffer("_tfi", torch.tensor(TARGET_FEAT_IDX, dtype=torch.long))

    def forward(self, x, mask):
        """x (B,W,17), mask (B,W,1). Returns (B,W,6) y-scaled fill; caller slices
        the gap steps. Source cols are assumed already zeroed where mask==0."""
        h, _ = self.rnn(torch.cat([x, mask], dim=-1))
        return self.head(h)


def masked_loss(pred_gap, y_gap, x_gap, nd_mean, nd_scale, y_mean, y_scale,
                sign, lam_bal: float = 0.1):
    """Reconstruction MSE on the gap + soft balance term (SIGN.P == net_demand).
    Balance is soft here (a training nudge); the hard guarantee is the eval-time
    projection. All tensors y-scaled except the balance term, done in MW."""
    rec = ((pred_gap - y_gap) ** 2).mean()
    p_mw = pred_gap * y_scale + y_mean                       # (B,G,6) MW
    nd_mw = x_gap * nd_scale + nd_mean                       # (B,G) MW  (net_demand col, pre-sliced)
    bal = (((p_mw * sign).sum(-1) - nd_mw) ** 2).mean() / (y_scale.mean() ** 2)
    return rec + lam_bal * bal, rec.detach(), bal.detach()
