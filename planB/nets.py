"""planB backbones (file is `nets.py`, not `model.py`, so it cannot be shadowed
 by imputation/model.py on the shared sys.path).
 Two architectures, both emitting 8 imputed channels
(6 dispatch + wind/solar curtailment) plus, for the rayen head only, a step logit.

BiLSTMImputer   the planA backbone with fix #5 applied: SEPARATE output heads for
                dispatch and curtailment. They previously shared one Linear, and
                that is what produced the softplus scale bug -- one group needed to
                reach ~1 (z-scores) and the other ~102 (raw MW) through the same
                weights, the same init and the same learning rate. Separate heads
                make the scaling explicit per group and remove the class of bug.

BRITSImputer    Cao et al. 2018, Bidirectional Recurrent Imputation for Time
                Series. The parts that matter here, per direction:

                  delta_t  steps since that channel was last observed. Inside a
                           36-step gap it climbs 1..36, so the model knows how
                           stale its own history is -- the BiLSTM has no such
                           signal, it only gets a binary mask.
                  gamma    exp(-relu(W delta)) decays the hidden state as the gap
                           widens, so a deep-gap estimate leans less on history.
                  x_hat    history estimate from h_{t-1}.
                  x_c      complement: observed value where known, x_hat where not.
                           This is what gets fed forward, so the recurrence never
                           sees a zero-filled hole.
                  z_hat    feature estimate, W_z x_c with a ZEROED DIAGONAL so a
                           channel cannot predict itself -- it must be inferred
                           from net demand, wind, solar and the other fuels.
                  beta     learned mix of the history and feature estimates.

                Three deviations from the paper, all for "one MSE / simplest":
                  * BRITS sums losses on x_hat, z_hat, c_hat plus a forward/backward
                    consistency term. Here the output is the MEAN of the two
                    directions' c_hat and only that is scored. Averaging is BRITS's
                    own combination rule, so consistency is implicit rather than
                    penalised.
                  * one layer per direction, as in the paper.
                  * the constraint head runs AFTER, on the gap slice, exactly as
                    for the BiLSTM arms -- putting it inside the recurrence would
                    make the per-step box direction-dependent and break the A/B.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BiLSTMImputer(nn.Module):
    def __init__(self, n_features: int, n_dispatch: int = 6, hidden: int = 192,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_features + 1, hidden, num_layers=layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head_dispatch = nn.Linear(hidden * 2, n_dispatch)
        self.head_curt = nn.Linear(hidden * 2, 2)

    def forward(self, x, mask, delta=None):
        h, _ = self.rnn(torch.cat([x, mask], dim=-1))
        return self.head_dispatch(h), self.head_curt(h)


class _RITS(nn.Module):
    """One direction of BRITS."""

    def __init__(self, D: int, hidden: int):
        super().__init__()
        self.D, self.H = D, hidden
        self.cell = nn.LSTMCell(D * 2, hidden)
        self.gamma_h = nn.Linear(D, hidden)          # hidden-state decay
        self.gamma_x = nn.Linear(D, D)               # input decay, feeds beta
        self.hist = nn.Linear(hidden, D)             # x_hat
        self.feat = nn.Linear(D, D)                  # z_hat (diagonal masked out)
        self.beta = nn.Linear(D * 2, D)
        self.register_buffer("_eye", torch.eye(D))

    def forward(self, x, m, delta):
        B, T, D = x.shape
        h = x.new_zeros(B, self.H); c = x.new_zeros(B, self.H)
        outs = []
        for t in range(T):
            g_h = torch.exp(-torch.relu(self.gamma_h(delta[:, t])))
            g_x = torch.exp(-torch.relu(self.gamma_x(delta[:, t])))
            h = h * g_h
            x_hat = self.hist(h)
            x_c = m[:, t] * x[:, t] + (1 - m[:, t]) * x_hat
            # zero the diagonal so a channel is never predicted from itself
            z_hat = torch.nn.functional.linear(
                x_c, self.feat.weight * (1 - self._eye), self.feat.bias)
            b = torch.sigmoid(self.beta(torch.cat([g_x, m[:, t]], -1)))
            c_hat = b * z_hat + (1 - b) * x_hat
            outs.append(c_hat)
            fed = m[:, t] * x[:, t] + (1 - m[:, t]) * c_hat
            h, c = self.cell(torch.cat([fed, m[:, t]], -1), (h, c))
        return torch.stack(outs, 1)


class BRITSImputer(nn.Module):
    """Bidirectional RITS. Emits (dispatch, curtailment) like BiLSTMImputer.

    `disp_idx` are the feature columns holding the dispatch channels IN TARGETS
    ORDER, `curt_idx` the two curtailment columns. `extra` adds the rayen head's
    step logit as a small separate output, since it is not a feature.
    """

    def __init__(self, n_features: int, disp_idx, curt_idx, n_dispatch: int = 6,
                 hidden: int = 192):
        super().__init__()
        self.fwd = _RITS(n_features, hidden)
        self.bwd = _RITS(n_features, hidden)
        self.register_buffer("_di", torch.tensor(list(disp_idx), dtype=torch.long))
        self.register_buffer("_ci", torch.tensor(list(curt_idx), dtype=torch.long))
        self.extra = n_dispatch - len(disp_idx)      # 1 for rayen, 0 for hardnet
        if self.extra:
            self.head_extra = nn.Linear(n_features, self.extra)

    def forward(self, x, mask, delta):
        m = mask.expand_as(x)
        f = self.fwd(x, m, delta)
        rev = lambda z: torch.flip(z, dims=[1])
        b = rev(self.bwd(rev(x), rev(m), rev(delta)))
        c = 0.5 * (f + b)                             # BRITS's own combination
        disp = c.index_select(-1, self._di)
        curt = c.index_select(-1, self._ci)
        if self.extra:
            disp = torch.cat([disp, self.head_extra(c)], -1)
        return disp, curt


def build_delta(mask, n_features: int):
    """(B,T,1) mask -> (B,T,D) steps since last observed, forward direction.

    The mask is shared across the imputed channels here (the whole gap is blanked
    at once), so delta is the same for every column; it is still emitted per
    column because BRITS's decay weights are per-feature.
    """
    B, T, _ = mask.shape
    d = mask.new_ones(B, T, 1)
    prev = mask.new_ones(B, 1)
    outs = []
    for t in range(T):
        cur = torch.ones_like(prev) if t == 0 else torch.where(
            mask[:, t - 1] > 0.5, torch.ones_like(prev), prev + 1)
        outs.append(cur); prev = cur
    return torch.stack(outs, 1).expand(B, T, n_features).contiguous()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# SAITS — Du, Cote & Liu 2023, "SAITS: Self-Attention-based Imputation for Time
# Series", Expert Systems with Applications 219:119619.
#
# WHY THIS ONE, given what planB measured. Every arm trails linear interpolation
# (50.6 MW) and the counterfactual allocation is decided by the constraint head
# rather than learned. Three properties of SAITS attack exactly that:
#
#   1. DIAGONALLY-MASKED SELF-ATTENTION. A timestep cannot attend to itself, so
#      its estimate must be assembled from other timesteps. This is BRITS's
#      zeroed-diagonal feature regression generalised to the time axis. Mid-gap
#      steps reach BOTH pinned boundaries in one hop; BRITS has to propagate 18
#      recurrent steps to do the same, which is why it needs a decay term at all.
#   2. TWO CASCADED BLOCKS + LEARNED COMBINATION. The second block re-imputes
#      given the first block's completion, and the two estimates are mixed by a
#      weight derived from the attention map and the mask. A single pass has to
#      commit; this does not.
#   3. MIT (masked imputation task). Randomly mask observed cells during training
#      and require reconstructing them. Our windows always blank the SAME centred
#      steps, so each window supplies one supervision pattern; MIT turns 40k
#      windows into far more. That directly addresses the data-starvation showing
#      up as "worse than interpolation".
#
# MIT is implemented in fit.py by extending the mask, so the extra cells fall into
# the SAME single MSE rather than a second weighted term.
#
# One deviation: the paper derives the combining weight from the second block's
# attention map through an unspecified reduction. Here the reduction is the
# attention mass each timestep RECEIVES, concatenated with the mask. Same inputs,
# an explicit reduction.
# ---------------------------------------------------------------------------
class _DMSA(nn.Module):
    """Self-attention with the diagonal masked out."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                nn.Linear(d_ff, d_model))
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        sh = lambda z: z.view(B, T, self.h, self.dk).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)
        eye = torch.eye(T, device=x.device, dtype=torch.bool)
        att = att.masked_fill(eye, float("-inf"))        # <- the diagonal mask
        a = att.softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(B, T, D)
        x = self.n1(x + self.drop(self.proj(o)))
        x = self.n2(x + self.drop(self.ff(x)))
        return x, a


class _Block(nn.Module):
    def __init__(self, D, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.emb = nn.Linear(D * 2, d_model)
        self.layers = nn.ModuleList(
            [_DMSA(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, D)

    def forward(self, x, m, pos):
        h = self.emb(torch.cat([x, m], -1)) + pos
        a = None
        for l in self.layers:
            h, a = l(h)
        return self.out(h), a


class SAITSImputer(nn.Module):
    def __init__(self, n_features, disp_idx, curt_idx, n_dispatch=6, hidden=192,
                 n_heads=4, n_layers=2, dropout=0.1, max_len=512):
        super().__init__()
        D = n_features
        d_model = hidden
        self.b1 = _Block(D, d_model, n_heads, n_layers, d_model * 2, dropout)
        self.b2 = _Block(D, d_model, n_heads, n_layers, d_model * 2, dropout)
        self.eta = nn.Linear(D + 1, D)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.register_buffer("_di", torch.tensor(list(disp_idx), dtype=torch.long))
        self.register_buffer("_ci", torch.tensor(list(curt_idx), dtype=torch.long))
        self.extra = n_dispatch - len(disp_idx)
        if self.extra:
            self.head_extra = nn.Linear(D, self.extra)

    def forward(self, x, mask, delta=None):
        m = mask.expand_as(x)
        pos = self.pe[:, :x.shape[1]]
        x1, _ = self.b1(x, m, pos)
        xc = m * x + (1 - m) * x1                     # completion after block 1
        x2, a2 = self.b2(xc, m, pos)
        recv = a2.mean(1).sum(1, keepdim=True).transpose(1, 2)   # attention received
        eta = torch.sigmoid(self.eta(torch.cat([recv, m], -1)))
        x3 = eta * x1 + (1 - eta) * x2
        disp = x3.index_select(-1, self._di)
        curt = x3.index_select(-1, self._ci)
        if self.extra:
            disp = torch.cat([disp, self.head_extra(x3)], -1)
        return disp, curt
