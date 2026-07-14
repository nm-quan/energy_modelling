"""Model architectures for the 5-min multi-target forecasting benchmark.

Input:  (B, T=288, C=16)     T=24h of 5-min steps, C from pipeline net_dispatch.
Output: (B, 6)               1-step-ahead forecast of the 6 dispatchable targets.

Neural models live as nn.Module subclasses; Linear and XGBoost are sklearn-style
wrappers with fit / predict so the experiment script can treat them uniformly.

target_feat_idx maps each of the 6 targets to its position in the 16-feature
input vector. The order is:
    TARGETS         = [hydro, coal_brown, gas_steam, gas_ocgt, batt_c, batt_d]
    features[0..5]  = [hydro, gas_steam, gas_ocgt, coal_brown, batt_c, batt_d]
    => target_feat_idx = [0, 3, 1, 2, 4, 5]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

TARGET_FEAT_IDX = [0, 3, 1, 2, 4, 5]

# Demand-driver channels in the `net_dispatch_totdem` feature layout:
#   [6] net_demand, [7] demand_mw. SelectiveRevIN passes these through on the
# frozen global scaler (absolute level preserved) instead of per-window RevIN,
# so the model can learn a demand-level -> dispatch mapping.
DEMAND_FEAT_IDX = [6, 7]


# -------------------------------- recurrent --------------------------------

class LSTMForecaster(nn.Module):
    def __init__(self, n_features=16, n_targets=6, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_targets)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class GRUForecaster(nn.Module):
    def __init__(self, n_features=16, n_targets=6, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_targets)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class BiLSTMForecaster(nn.Module):
    def __init__(self, n_features=16, n_targets=6, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0, bidirectional=True)
        self.head = nn.Linear(hidden * 2, n_targets)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class CNNLSTMForecaster(nn.Module):
    def __init__(self, n_features=16, n_targets=6, hidden=128, layers=2, dropout=0.2,
                 conv_channels=64, kernel=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel, padding=kernel // 2),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(conv_channels, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_targets)

    def forward(self, x):
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(c)
        return self.head(out[:, -1, :])


# -------------------------------- transformers --------------------------------

def _revin(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-window per-channel normalisation. x: (B, T, C) -> (xn, mean, std)."""
    mean = x.mean(dim=1, keepdim=True).detach()
    std = x.std(dim=1, keepdim=True).detach() + 1e-5
    return (x - mean) / std, mean, std


class RevIN(nn.Module):
    """Wrap a (B, T, C) -> (B, K) forecaster so input is per-window normalised
    on every channel, and output is denormalised using the target channels'
    own per-window mean/std. K = len(target_feat_idx)."""

    def __init__(self, base: nn.Module, target_feat_idx):
        super().__init__()
        self.base = base
        self.register_buffer("_idx", torch.tensor(list(target_feat_idx), dtype=torch.long))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()           # (B, 1, C)
        std = x.std(dim=1, keepdim=True).detach() + 1e-5      # (B, 1, C)
        out = self.base((x - mean) / std)                     # (B, K) normed
        tgt_mean = mean.squeeze(1).index_select(1, self._idx) # (B, K)
        tgt_std = std.squeeze(1).index_select(1, self._idx)   # (B, K)
        return out * tgt_std + tgt_mean


class SelectiveRevIN(nn.Module):
    """RevIN on every channel EXCEPT the listed `passthrough_idx`.

    Standard RevIN normalises all channels per-window, which erases the
    absolute level of the demand drivers and kills the demand response. Here
    the passthrough channels (the demand drivers) keep their globally-scaled
    raw values, so the model still sees "today is a high-demand day", while the
    dispatch-history and other channels keep the per-window normalisation that
    drives accuracy. Output is denormalised with the target channels' stats, as
    in plain RevIN.
    """

    def __init__(self, base: nn.Module, target_feat_idx, passthrough_idx, n_features: int):
        super().__init__()
        self.base = base
        self.register_buffer("_idx", torch.tensor(list(target_feat_idx), dtype=torch.long))
        # (1, 1, C) float mask: 1 where the channel is RevIN-normalised, 0 where it
        # passes through raw. Elementwise blend below avoids index_copy (unimplemented
        # on MPS) while giving identical results.
        norm_mask = torch.ones(1, 1, n_features)
        norm_mask[0, 0, list(passthrough_idx)] = 0.0
        self.register_buffer("_norm_mask", norm_mask)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()           # (B, 1, C)
        std = x.std(dim=1, keepdim=True).detach() + 1e-5      # (B, 1, C)
        x_n = (x - mean) / std
        # keep passthrough (demand) channels at their raw globally-scaled level
        x_n = x_n * self._norm_mask + x * (1.0 - self._norm_mask)
        out = self.base(x_n)                                  # (B, K) normed
        tgt_mean = mean.squeeze(1).index_select(1, self._idx) # (B, K)
        tgt_std = std.squeeze(1).index_select(1, self._idx)   # (B, K)
        return out * tgt_std + tgt_mean


class DishTS(nn.Module):
    """Dish-TS-style reversible normalisation (Fan et al., AAAI 2023).

    RevIN *copies* the lookback mean onto the output, so the predicted level is
    always anchored to dispatch history and cannot follow a demand shift. Dish-TS
    instead *predicts* the horizon level (xi) with a learned coefficient net. The
    `xi_mode` selects what that level-predictor is allowed to see:

      'own'   : per-target learned weighted-mean of that target's OWN history
                (channel-independent, faithful Dish-TS) -> still level-blind to demand.
      'cross' : linear net over ALL channels' recent mean -> FREE to use demand.
      'exo'   : linear net over NON-target channels' recent mean (demand/price/
                calendar) -> level is FORCED through demand.

    Input is per-window RevIN-normalised in every mode; only the output level path
    differs. At init 'own'/'cross' reduce to ~RevIN (output = recent target mean).
    """

    def __init__(self, base: nn.Module, target_feat_idx, n_features: int,
                 xi_mode: str = "own", seq_len: int = 288, recent: int = 12):
        super().__init__()
        self.base = base
        self.xi_mode = xi_mode
        self.recent = recent
        idx = list(target_feat_idx)
        self.register_buffer("_idx", torch.tensor(idx, dtype=torch.long))
        self.w_phi = nn.Parameter(torch.full((seq_len,), 1.0 / seq_len))   # input level weights
        if xi_mode == "own":
            self.w_xi = nn.Parameter(torch.full((seq_len,), 1.0 / seq_len))
        elif xi_mode == "cross":
            self.level_net = nn.Linear(n_features, len(idx))
            nn.init.zeros_(self.level_net.weight); nn.init.zeros_(self.level_net.bias)
            with torch.no_grad():                       # start = each target's own recent mean
                for k, c in enumerate(idx):
                    self.level_net.weight[k, c] = 1.0
        elif xi_mode == "exo":
            exo = [i for i in range(n_features) if i not in idx]
            self.register_buffer("_exo", torch.tensor(exo, dtype=torch.long))
            self.level_net = nn.Linear(len(exo), len(idx))
            nn.init.zeros_(self.level_net.weight); nn.init.zeros_(self.level_net.bias)
        else:
            raise ValueError(f"unknown xi_mode {xi_mode!r}")

    def forward(self, x):                                                  # (B, T, C)
        phi_mu = (x * self.w_phi.view(1, -1, 1)).sum(dim=1, keepdim=True)   # (B, 1, C)
        sigma = x.std(dim=1, keepdim=True).detach() + 1e-5                  # (B, 1, C)
        out = self.base((x - phi_mu) / sigma)                              # (B, K) normed
        xt = x.index_select(2, self._idx)                                  # (B, T, K)
        xi_sigma = xt.std(dim=1).detach() + 1e-5                            # (B, K)
        if self.xi_mode == "own":
            xi_mu = (xt * self.w_xi.view(1, -1, 1)).sum(dim=1)             # (B, K)
        else:
            summ = x[:, -self.recent:, :].mean(dim=1)                       # (B, C) recent raw mean
            if self.xi_mode == "exo":
                summ = summ.index_select(1, self._exo)                      # (B, n_exo)
            xi_mu = self.level_net(summ)                                    # (B, K)
        return out * xi_sigma + xi_mu


# net_demand = hydro + coal_brown + gas_steam + gas_ocgt + battery_discharging
#              - battery_charging   (pipeline.build_table, exact identity by
# construction). Split TARGETS by their sign in that identity:
TARGET_GEN_IDX = [0, 1, 2, 3, 5]   # hydro, coal_brown, gas_steam, gas_ocgt, batt_dis (+1)
TARGET_CHG_IDX = 4                  # battery_charging (-1)


class DemandAnchoredHead(nn.Module):
    """Rescales the base model's output so it satisfies the net-demand identity
    EXACTLY: sum(generation) + batt_dis - batt_chg == net_demand, read straight
    off the input window (in MW). RevIN, SelectiveRevIN, Dish-TS and a soft
    energy-balance loss all left the model free to ignore demand because nothing
    forced it to -- RevIN's output denorm re-anchors the level to dispatch
    HISTORY, so MSE is minimised without ever reading demand (saliency ~0%, see
    demand_simulation/findings.md). This wrapper removes that escape hatch
    architecturally: the base network only has to learn the fuel MIX (the
    relative shares of 5 generation channels, plus the charging level); the
    TOTAL is corrected after the fact to match net_demand exactly, so the
    response is structural and cannot decay with more training.

    base:  any (B,T,C) -> (B,6) forecaster (TARGETS order). Its raw output is
           interpreted in y_scaler-scaled units like every other model here, so
           it drops into the existing train/eval loop unchanged.
    x_nd_mean/x_nd_scale, y_mean/y_scale: the GLOBAL x_scaler/y_scaler stats
           (fit once on train data) needed to round-trip into MW, where the
           SIGN identity is physically meaningful, and back.
    nd_feat_idx: position of net_demand within the C input feature channels.
    rescale_idx: which generation-sign targets absorb the aggregate correction
           (default: all 5). Targets left out are softplus passthroughs whose
           predicted MW is subtracted from `need` instead -- the identity still
           holds exactly. Motivation: the proportional rescale's error is worst
           on tiny inflexible units (gas_steam) whose on-periods coincide with
           ramps, so excluding them keeps their bare accuracy while the big
           flexible channels carry the correction.
    pos_fn: 'softplus' (default; smooth, needed to train through the head) or
           'relu' (exact zeros; use for inference-only anchoring -- softplus's
           ~+0.7 MW floor on near-zero predictions is pure error on units that
           are off most of the time, e.g. gas_steam).
    """

    def __init__(self, base: nn.Module, x_nd_mean: float, x_nd_scale: float,
                 y_mean, y_scale, nd_feat_idx: int = 6, eps: float = 1.0,
                 rescale_idx=None, pos_fn: str = "softplus"):
        super().__init__()
        assert pos_fn in ("softplus", "relu")
        self.base = base
        self.nd_feat_idx = nd_feat_idx
        self.eps = eps
        self.pos_fn = pos_fn
        rescale_idx = TARGET_GEN_IDX if rescale_idx is None else list(rescale_idx)
        assert set(rescale_idx) <= set(TARGET_GEN_IDX), "rescale_idx must be generation-sign targets"
        pass_idx = [i for i in TARGET_GEN_IDX if i not in rescale_idx]
        self.register_buffer("_gen_idx", torch.tensor(rescale_idx, dtype=torch.long))
        # derived layout buffers are persistent=False so state_dicts saved by the
        # original all-5 head still load. _scatter places the R rescaled values
        # back into TARGETS order via matmul (index_copy is unimplemented on MPS).
        self.register_buffer("_pass_idx", torch.tensor(pass_idx, dtype=torch.long), persistent=False)
        scatter = torch.zeros(len(rescale_idx), 6)
        for r, t in enumerate(rescale_idx):
            scatter[r, t] = 1.0
        keep = torch.ones(6)
        keep[rescale_idx] = 0.0
        self.register_buffer("_scatter", scatter, persistent=False)
        self.register_buffer("_keep_mask", keep, persistent=False)
        self.register_buffer("_x_nd_mean", torch.tensor(float(x_nd_mean)))
        self.register_buffer("_x_nd_scale", torch.tensor(float(x_nd_scale)))
        self.register_buffer("_y_mean", torch.tensor(np.asarray(y_mean), dtype=torch.float32))
        self.register_buffer("_y_scale", torch.tensor(np.asarray(y_scale), dtype=torch.float32))

    def forward(self, x):
        raw = self.base(x)                                              # (B,6) y_scaler-scaled
        raw_mw = raw * self._y_scale + self._y_mean                      # (B,6) MW, may be < 0
        pos_mw = (nn.functional.softplus(raw_mw) if self.pos_fn == "softplus"
                  else nn.functional.relu(raw_mw))                       # (B,6) MW, >= 0 (targets are >=0)

        nd_mw = x[:, -1, self.nd_feat_idx] * self._x_nd_scale + self._x_nd_mean  # (B,) MW
        chg = pos_mw[:, TARGET_CHG_IDX:TARGET_CHG_IDX + 1]                # (B,1) MW
        need = nd_mw.unsqueeze(-1) + chg                                  # (B,1) MW, generation+discharge required
        if len(self._pass_idx):                                           # passthrough units serve their share as-is
            need = need - pos_mw.index_select(1, self._pass_idx).sum(dim=-1, keepdim=True)
        need = need.clamp_min(self.eps)

        gen = pos_mw.index_select(1, self._gen_idx)                       # (B,R) MW
        gen_safe = gen + self.eps / len(TARGET_GEN_IDX)                   # > 0, so sum is always > 0
        scale = need / gen_safe.sum(dim=-1, keepdim=True)
        gen_scaled = gen_safe * scale                                     # (B,R) MW, sums EXACTLY to `need`

        out_mw = pos_mw * self._keep_mask + gen_scaled @ self._scatter    # (B,6) TARGETS order
        return (out_mw - self._y_mean) / self._y_scale                    # back to y_scaler-scaled units


def make_demand_anchored(base_arch: str, x_scaler, y_scaler, nd_feat_idx: int = 6,
                         n_features: int = 17, n_targets: int = 6) -> nn.Module:
    """Build `base_arch` (e.g. 'lstm', 'lstm_revin', 'itransformer') and wrap it
    with DemandAnchoredHead using the pipeline's fitted scalers. Needs the
    scalers at construction time (unlike the other wrappers, which only use
    per-window local stats), so it is not routed through make_neural/make_model.
    """
    base = make_neural(base_arch, n_features=n_features, n_targets=n_targets)
    return DemandAnchoredHead(base, x_scaler.mean_[nd_feat_idx], x_scaler.scale_[nd_feat_idx],
                              y_scaler.mean_, y_scaler.scale_, nd_feat_idx=nd_feat_idx)


class LearnableRevIN(nn.Module):
    """RevIN with per-channel learnable affine on the normalised representation.

    After (x - mu)/sigma, apply a learnable per-channel (gamma, beta). The
    inverse of the affine is applied to the target-dim outputs before adding
    back the per-window mean/std.
    """

    def __init__(self, base: nn.Module, target_feat_idx, n_features: int):
        super().__init__()
        self.base = base
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.register_buffer("_idx", torch.tensor(list(target_feat_idx), dtype=torch.long))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(dim=1, keepdim=True).detach() + 1e-5
        x_n = (x - mean) / std
        x_n = x_n * self.gamma + self.beta                    # (B, T, C) * (C,) + (C,)
        out = self.base(x_n)                                  # (B, K) in post-affine norm
        gamma_t = self.gamma.index_select(0, self._idx)
        beta_t = self.beta.index_select(0, self._idx)
        out = (out - beta_t) / (gamma_t + 1e-8)               # undo affine on targets
        tgt_mean = mean.squeeze(1).index_select(1, self._idx)
        tgt_std = std.squeeze(1).index_select(1, self._idx)
        return out * tgt_std + tgt_mean


class NLinearShift(nn.Module):
    """NLinear-style normalisation: subtract the window's last value before
    the model, add the target channels' last value back after. Pegs the
    zero point at "where we are now" instead of "window mean" (RevIN).
    """

    def __init__(self, base: nn.Module, target_feat_idx):
        super().__init__()
        self.base = base
        self.register_buffer("_idx", torch.tensor(list(target_feat_idx), dtype=torch.long))

    def forward(self, x):
        last = x[:, -1:, :]                                   # (B, 1, C)
        out = self.base(x - last)                             # (B, K) in shifted space
        tgt_last = last.squeeze(1).index_select(1, self._idx) # (B, K)
        return out + tgt_last


class SeriesDecompGRU(nn.Module):
    """Autoformer-style series decomposition with a GRU backbone.

    Input window is split into a smooth trend (moving average, kernel=25 = ~2h
    of 5-min steps) and a seasonal residual (input minus trend). Two parallel
    GRUs process trend and seasonal; their final hidden states are concatenated
    and projected to the n_targets outputs.
    """

    def __init__(self, n_features=16, n_targets=6, kernel=25,
                 hidden=128, layers=2, dropout=0.2):
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd to keep length identical"
        self.kernel = kernel
        self.pad = kernel // 2
        self.gru_trend = nn.GRU(n_features, hidden, num_layers=layers,
                                batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
        self.gru_season = nn.GRU(n_features, hidden, num_layers=layers,
                                 batch_first=True,
                                 dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, n_targets)

    def _trend(self, x):
        # replicate-pad the edges so trend has the same length as input
        xc = x.transpose(1, 2)                                # (B, C, T)
        xp = nn.functional.pad(xc, (self.pad, self.pad), mode="replicate")
        return nn.functional.avg_pool1d(xp, kernel_size=self.kernel, stride=1).transpose(1, 2)

    def forward(self, x):
        trend = self._trend(x)                                # (B, T, C)
        seasonal = x - trend
        h_t, _ = self.gru_trend(trend)
        h_s, _ = self.gru_season(seasonal)
        return self.head(torch.cat([h_t[:, -1, :], h_s[:, -1, :]], dim=1))


class TimeXer(nn.Module):
    """TimeXer adapted to 6 endogenous + 10 exogenous channels.

    - Per-window RevIN on every channel (denormalised target prediction at output).
    - Each endogenous channel is patched into temporal tokens, prefixed with a
      learned global token, and processed by self-attention; the global token
      then cross-attends to every exogenous variate token.
    - Per-target head: takes the patch tokens of that target's endogenous
      channel (after self/cross attention) and projects to a single MW value.
    """

    def __init__(self, seq_len=288, n_features=16, n_targets=6, target_feat_idx=None,
                 patch_len=24, d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.n_targets = n_targets
        self.target_idx = list(range(n_targets)) if target_feat_idx is None else list(target_feat_idx)
        # exogenous = every feature index not in target_idx, in input order
        self.exo_idx = [i for i in range(n_features) if i not in self.target_idx]
        self.n_exo = len(self.exo_idx)

        self.endo_embed = nn.Linear(patch_len, d_model)
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.endo_pos = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        self.exo_embed = nn.Linear(seq_len, d_model)
        self.exo_pos = nn.Parameter(torch.randn(1, self.n_exo, d_model) * 0.02)

        self.self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                       dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.cross_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.n_patches * d_model, 1)  # per-target

        self.register_buffer("_target_idx_t", torch.tensor(self.target_idx, dtype=torch.long))
        self.register_buffer("_exo_idx_t", torch.tensor(self.exo_idx, dtype=torch.long))

    def forward(self, x):
        B, T, _ = x.shape
        endo_raw = x.index_select(2, self._target_idx_t)                       # (B, T, K)
        exo_raw = x.index_select(2, self._exo_idx_t)                           # (B, T, n_exo)

        endo_n, endo_mean, endo_std = _revin(endo_raw)                          # (B, T, K)
        exo_n, _, _ = _revin(exo_raw)                                          # (B, T, n_exo)

        # Patch each endo channel: (B, K, P, patch_len) -> (B, K, P, d)
        K = self.n_targets
        endo_p = endo_n.permute(0, 2, 1).reshape(B * K, self.n_patches, self.patch_len)
        endo_tok = self.endo_embed(endo_p)                                      # (B*K, P, d)
        gtok = self.global_token.expand(B * K, -1, -1)                          # (B*K, 1, d)
        endo_tok = torch.cat([gtok, endo_tok], dim=1)                           # (B*K, P+1, d)
        endo_tok = self.dropout(endo_tok + self.endo_pos)

        # Exo variate tokens, shared across endo channels: (B, n_exo, d)
        exo_tok = self.exo_embed(exo_n.permute(0, 2, 1)) + self.exo_pos          # (B, n_exo, d)
        exo_tok_rep = exo_tok.repeat_interleave(K, dim=0)                       # (B*K, n_exo, d)

        for self_attn, cross_attn, norm in zip(self.self_layers, self.cross_layers, self.cross_norms):
            endo_tok = self_attn(endo_tok)
            ca, _ = cross_attn(endo_tok, exo_tok_rep, exo_tok_rep, need_weights=False)
            endo_tok = norm(endo_tok + ca)

        patch_repr = endo_tok[:, 1:].flatten(1)                                 # (B*K, P*d)
        out = self.head(patch_repr).reshape(B, K)                               # (B, K) in normed space

        # Denormalise each target back to its own channel mean/std
        endo_mean_k = endo_mean.squeeze(1)                                      # (B, K)
        endo_std_k = endo_std.squeeze(1)                                        # (B, K)
        return out * endo_std_k + endo_mean_k


class iTransformer(nn.Module):
    """iTransformer: each variable is one token, self-attention runs across
    variables. Head reads the K target tokens and outputs 1 value per target.
    """

    def __init__(self, seq_len=288, n_features=16, n_targets=6, target_feat_idx=None,
                 d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1,
                 use_revin=True, revin_out=True):
        super().__init__()
        self.use_revin = use_revin
        # revin_out=False: normalise inputs but return RAW head outputs (no
        # per-window denorm) -- required when outputs are direction vectors
        # (RayenHead), where adding window means would corrupt the direction.
        self.revin_out = revin_out
        self.target_idx = list(range(n_targets)) if target_feat_idx is None else list(target_feat_idx)
        self.var_embed = nn.Linear(seq_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

        self.register_buffer("_target_idx_t", torch.tensor(self.target_idx, dtype=torch.long))

    def forward(self, x):
        if self.use_revin:
            x_in, mean, std = _revin(x)                                         # (B, T, C)
        else:
            x_in = x                                       # raw (globally-scaled) input
        tokens = self.var_embed(x_in.permute(0, 2, 1))                          # (B, C, d)
        tokens = self.dropout(tokens + self.pos)
        tokens = self.encoder(tokens)                                           # (B, C, d)
        tgt_tok = tokens.index_select(1, self._target_idx_t)                    # (B, K, d)
        out = self.head(tgt_tok).squeeze(-1)                                    # (B, K)
        if not self.use_revin or not self.revin_out:
            return out                                     # raw (no per-window denorm)
        # denormalise per target using each target's own channel stats
        tgt_mean = mean.squeeze(1).index_select(1, self._target_idx_t)
        tgt_std = std.squeeze(1).index_select(1, self._target_idx_t)
        return out * tgt_std + tgt_mean


class PatchTST(nn.Module):
    """PatchTST (Nie et al. 2023), channel-independent, multi-target output.

    Each of the C input channels is independently patched and passed through a
    shared transformer encoder. The head produces one prediction per channel;
    the 6 target channels are selected as the final output.
    """

    def __init__(self, seq_len=288, n_features=16, n_targets=6, target_feat_idx=None,
                 patch_len=16, stride=8, d_model=128, n_heads=4, n_layers=3,
                 d_ff=256, dropout=0.1):
        super().__init__()
        self.target_idx = list(range(n_targets)) if target_feat_idx is None else list(target_feat_idx)
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, 1)

        self.register_buffer("_target_idx_t", torch.tensor(self.target_idx, dtype=torch.long))

    def forward(self, x):
        x_n, mean, std = _revin(x)                                              # (B, T, C)
        B, T, C = x_n.shape
        # Unfold into patches per channel: (B, C, n_patches, patch_len)
        x_c = x_n.permute(0, 2, 1)                                              # (B, C, T)
        patches = x_c.unfold(dimension=2, size=self.patch_len, step=self.stride)  # (B, C, P, patch_len)
        z = self.patch_embed(patches)                                           # (B, C, P, d)
        z = z.reshape(B * C, self.n_patches, -1)
        z = self.dropout(z + self.pos)
        z = self.encoder(z)                                                     # (B*C, P, d)
        z = z.flatten(1)                                                        # (B*C, P*d)
        out = self.head(z).reshape(B, C)                                        # (B, C) normed

        # take target channels and denormalise per target
        tgt = out.index_select(1, self._target_idx_t)                           # (B, K)
        tgt_mean = mean.squeeze(1).index_select(1, self._target_idx_t)
        tgt_std = std.squeeze(1).index_select(1, self._target_idx_t)
        return tgt * tgt_std + tgt_mean


class RayenHead(nn.Module):
    """Konstantinov/RAYEN-style hard-constraint output layer (see plan.md).

    Output y = (D_t, P_1..P_6) in MW satisfies, by construction:
      - balance:  SIGN . P - D_t = 0   (exactly, at any weights)
      - ramps:    |P_i - P_{i,t-1}| within the asymmetric per-step limits
      - floors:   P_i >= 0 (optional include_floor, on by default)

    Mechanism: anchor p = previous step's dispatch (its D entry = SIGN.p, so p
    is on the balance plane and trivially ramp-feasible; zero movement =
    persistence). The backbone emits a raw direction r (7) + scalar s; r is
    projected onto the balance plane, alpha* is the distance to the nearest
    ramp/floor wall along r, and y = p + sigmoid(s) * alpha* * r. Exact
    gradients, no divisions by predicted sums (cf. the DemandAnchoredHead
    trained-through failure).

    base outputs (B, 8) RAW (no output denorm -- RevIN denorm would add
    per-window means onto direction components). Returned prediction is (B, 7)
    scaled: [:, 0] = D_t in x-scaler nd units, [:, 1:] = dispatch in y-scaler
    units (so [:, 1:] drops into the existing eval loop).
    """

    def __init__(self, base: nn.Module, x_mean, x_scale, y_mean, y_scale,
                 ramp_up, ramp_dn, nd_feat_idx: int = 6,
                 include_floor: bool = True, eps_mw: float = 0.5):
        super().__init__()
        self.base = base
        self.nd_feat_idx = nd_feat_idx
        self.include_floor = include_floor
        self.eps_mw = eps_mw
        sign = torch.tensor([1., 1., 1., 1., -1., 1.])
        self.register_buffer("_sign", sign)
        a = torch.cat([torch.tensor([-1.]), sign])                    # (7,) plane normal
        self.register_buffer("_a", a)
        self.register_buffer("_feat_idx", torch.tensor(TARGET_FEAT_IDX, dtype=torch.long))
        self.register_buffer("_x_mean", torch.tensor(np.asarray(x_mean)[TARGET_FEAT_IDX], dtype=torch.float32))
        self.register_buffer("_x_scale", torch.tensor(np.asarray(x_scale)[TARGET_FEAT_IDX], dtype=torch.float32))
        self.register_buffer("_nd_mean", torch.tensor(float(np.asarray(x_mean)[nd_feat_idx])))
        self.register_buffer("_nd_scale", torch.tensor(float(np.asarray(x_scale)[nd_feat_idx])))
        self.register_buffer("_y_mean", torch.tensor(np.asarray(y_mean), dtype=torch.float32))
        self.register_buffer("_y_scale", torch.tensor(np.asarray(y_scale), dtype=torch.float32))
        self.register_buffer("_r_up", torch.tensor(np.asarray(ramp_up), dtype=torch.float32))
        self.register_buffer("_r_dn", torch.tensor(np.asarray(ramp_dn), dtype=torch.float32))
        # start AT the anchor (= persistence, the strong baseline): damped raw
        # directions + strongly negative step bias => initial output ~ p, and
        # training pushes away from persistence only where the loss says so.
        self.r_gain = nn.Parameter(torch.tensor(0.05))
        self.s_bias = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x):
        out = self.base(x)                                            # (B, 8) raw
        r, s = out[:, :7] * self.r_gain, out[:, 7] + self.s_bias

        prev = x[:, -1, :].index_select(1, self._feat_idx)            # (B,6) x-scaled
        prev_mw = prev * self._x_scale + self._x_mean
        p_gen = prev_mw.clamp_min(self.eps_mw)                        # eps-interior anchor
        p = torch.cat([(p_gen * self._sign).sum(-1, keepdim=True), p_gen], dim=1)

        if self.include_floor:                                        # kill descent on floored channels
            at_floor = (p_gen <= self.eps_mw + 1e-6) & (r[:, 1:] < 0)
            r = torch.cat([r[:, :1], r[:, 1:].masked_fill(at_floor, 0.0)], dim=1)
        r = r - (r @ self._a).unsqueeze(-1) * self._a / (self._a @ self._a)

        rg = r[:, 1:]                                                 # (B,6) generator components
        big = torch.full_like(rg, 1e9)
        wall_up = torch.where(rg > 1e-9, self._r_up / rg.clamp_min(1e-9), big)
        dn_lim = self._r_dn if not self.include_floor else torch.minimum(
            self._r_dn.expand_as(rg), p_gen)                          # floor wall: travel < p_i
        wall_dn = torch.where(rg < -1e-9, dn_lim / (-rg).clamp_min(1e-9), big)
        alpha = torch.minimum(wall_up, wall_dn).amin(dim=1)           # (B,)
        alpha = torch.where(alpha >= 1e9, torch.zeros_like(alpha), alpha)  # r_gen ~ 0 -> persistence

        y = p + torch.sigmoid(s).unsqueeze(-1) * alpha.unsqueeze(-1) * r
        d_scaled = (y[:, :1] - self._nd_mean) / self._nd_scale
        gen_scaled = (y[:, 1:] - self._y_mean) / self._y_scale
        return torch.cat([d_scaled, gen_scaled], dim=1)               # (B, 7)


class RayenHeadFixedD(nn.Module):
    """RAYEN hard-constraint layer with the demand plane PINNED to the exogenous
    net demand read off the input window (nd(t-1)), instead of a co-predicted
    D_t output (contrast RayenHead).

    Why. In RayenHead the balance plane SIGN.P = D_t has D_t as a FREE network
    output (the 7-dim direction moves it along r[0]), so the net co-predicts
    demand and dispatch: balance holds "vs itself" (@own=0) but a demand-side
    input shift never *forces* the fleet to move -- the collapse-frozen backbone
    just re-forecasts its own D_t (plan.md note 4; findings.md non-response).
    Here D is not predicted: d* = nd(t-1) is read straight from the window (like
    DemandAnchoredHead / anchor(persistence)), so the plane MOVES with the
    exogenous demand and the dispatch is forced onto it. Response is structural
    and cannot decay with training, while the RAYEN ramp + floor guarantees are
    retained. Backbone emits 7 RAW outputs: 6 direction + 1 scalar s.

    Construction (per step, MW):
      p     = prev dispatch (eps-interior)                    # persistence anchor, (B,6)
      d*    = nd(t-1) read from the window                    # (B,) plane RHS
      delta = d* - SIGN.p                                     # (B,) residual to the plane
      p_on  = p + m,  m = ramp/floor-limited move closing delta  # FORCED normal move = the response
      r     = SIGN-tangential backbone direction (sum-preserving)
      y     = p_on + sigmoid(s) * alpha* * r                  # FREE mix reallocation

    delta == 0 in-distribution teacher-forced (data identity nd == SIGN.P), so
    p_on == p and this reduces to a ramp/floor-feasible mix step exactly like the
    stationary RAYEN head. In the closed-loop free window delta is the exogenous
    5-min demand *change* (the ~197 MW supply/demand-side offset cancels in the
    delta); it is distributed across channels by ramp headroom and clipped to the
    ramps. When a demand jump exceeds the fleet's one-step ramp capacity the
    reprojection is ramp-bound and a residual imbalance remains (physically
    unavoidable, reported) -- ramps win over balance, the correct priority for a
    hard physical wall. The free step then uses only the *remaining* ramp budget,
    so the total move p -> y stays within ramps.
    """

    def __init__(self, base: nn.Module, x_mean, x_scale, y_mean, y_scale,
                 ramp_up, ramp_dn, nd_feat_idx: int = 6,
                 include_floor: bool = True, eps_mw: float = 0.5,
                 passthrough_idx=None, alloc_weights=None):
        super().__init__()
        self.base = base
        self.nd_feat_idx = nd_feat_idx
        self.include_floor = include_floor
        self.eps_mw = eps_mw
        # SOC-fix 1 (study): allocation preference for the forced reprojection.
        # None/ones = legacy behaviour (proportional to ramp headroom alone,
        # which hands the batteries ~54% of every delta). MinT-style weights
        # (inverse error variance / energy share) shift corrections onto the
        # channels that actually absorb them in reality. Non-persistent so
        # existing checkpoints load unchanged.
        w = torch.ones(6) if alloc_weights is None else \
            torch.as_tensor(alloc_weights, dtype=torch.float32)
        self.register_buffer("_alloc_w", w, persistent=False)
        # SOC-fix 2 (study): stateful shield hook. A closed-loop driver may set
        # this to a (B, 2) tensor of battery LEVEL caps [chg_max_mw, dis_max_mw]
        # derived from its tracked state of charge before each forward; the
        # head then respects them as extra walls (None = off).
        self.batt_room: torch.Tensor | None = None
        self.register_buffer("_sign", torch.tensor([1., 1., 1., 1., -1., 1.]))
        # passthrough channels are frozen at persistence: excluded from BOTH the
        # forced reprojection and the tangential mix step, so the flexible
        # channels carry balance + response. MSE barely penalises a small
        # inflexible unit (gas_steam), so leaving it free lets the tangential
        # step shove it around (WAPE 0.036 -> 0.26). Same fix as the DA-head's
        # rescale_idx=nosteam (see da-head / inference-anchor findings).
        free = torch.ones(6)
        if passthrough_idx is not None:
            free[list(passthrough_idx)] = 0.0
        self.register_buffer("_free", free)                          # 1 flexible, 0 passthrough
        self.register_buffer("_sign_free", torch.tensor([1., 1., 1., 1., -1., 1.]) * free)
        self.register_buffer("_feat_idx", torch.tensor(TARGET_FEAT_IDX, dtype=torch.long))
        self.register_buffer("_x_mean", torch.tensor(np.asarray(x_mean)[TARGET_FEAT_IDX], dtype=torch.float32))
        self.register_buffer("_x_scale", torch.tensor(np.asarray(x_scale)[TARGET_FEAT_IDX], dtype=torch.float32))
        self.register_buffer("_nd_mean", torch.tensor(float(np.asarray(x_mean)[nd_feat_idx])))
        self.register_buffer("_nd_scale", torch.tensor(float(np.asarray(x_scale)[nd_feat_idx])))
        self.register_buffer("_y_mean", torch.tensor(np.asarray(y_mean), dtype=torch.float32))
        self.register_buffer("_y_scale", torch.tensor(np.asarray(y_scale), dtype=torch.float32))
        self.register_buffer("_r_up", torch.tensor(np.asarray(ramp_up), dtype=torch.float32))
        self.register_buffer("_r_dn", torch.tensor(np.asarray(ramp_dn), dtype=torch.float32))
        self.r_gain = nn.Parameter(torch.tensor(0.05))
        self.s_bias = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x):
        out = self.base(x)                                            # (B,7) raw
        r = out[:, :6] * self.r_gain                                  # (B,6) tangential direction
        s = out[:, 6] + self.s_bias                                   # (B,)

        prev = x[:, -1, :].index_select(1, self._feat_idx)            # (B,6) x-scaled
        prev_mw = prev * self._x_scale + self._x_mean
        # free channels get an eps-interior anchor (needed for the ramp-down floor
        # wall); passthrough channels take no walls, so anchor them at raw
        # persistence -- the 0.5 MW eps is pure error on an often-idle peaker.
        p = torch.where(self._free.bool(), prev_mw.clamp_min(self.eps_mw), prev_mw.clamp_min(0.0))
        d_star = x[:, -1, self.nd_feat_idx] * self._nd_scale + self._nd_mean   # (B,) MW

        # SOC shield (if armed): battery LEVEL caps [chg, dis] from the driver's
        # tracked state of charge. Two effects, both spending ramp budget:
        # (1) forced ramp-limited descent when the current level already exceeds
        #     its room (the overshoot is then bounded by the ramp-down envelope,
        #     ~1-2 steps -- drivers add a small margin to the room for strictness);
        # (2) up-walls so the level can never grow past the room.
        r_up_eff = self._r_up.expand_as(p)
        r_dn_eff = self._r_dn.expand_as(p)
        if self.batt_room is not None:
            room = self.batt_room.to(p.dtype)
            p = p.clone(); r_up_eff = r_up_eff.clone(); r_dn_eff = r_dn_eff.clone()
            for j, ch in enumerate((4, 5)):
                down = torch.minimum((p[:, ch] - room[:, j]).clamp_min(0.0), r_dn_eff[:, ch])
                p[:, ch] = p[:, ch] - down
                r_dn_eff[:, ch] = r_dn_eff[:, ch] - down
                r_up_eff[:, ch] = torch.minimum(r_up_eff[:, ch],
                                                (room[:, j] - p[:, ch]).clamp_min(0.0))

        # --- forced ramp/floor-limited move onto the plane SIGN.P = d_star ---
        delta = d_star - (p * self._sign).sum(-1)                     # (B,) residual to close
        pos = (delta > 0).unsqueeze(-1)                               # (B,1) raising the signed sum?
        floor_cap = torch.minimum(r_dn_eff, p)                       # down-move bounded by ramp AND floor
        # per-channel capacity to raise / lower the SIGNED sum within ramps+floor
        up_cap = torch.where(self._sign > 0, r_up_eff, floor_cap)
        dn_cap = torch.where(self._sign > 0, floor_cap, r_up_eff)
        cap = torch.where(pos, up_cap, dn_cap) * self._free           # (B,6) capacity toward delta, >=0
        # preference-weighted allocation of |delta| across channels, iterated so
        # saturated channels hand their remainder on. With uniform weights round
        # 1 reduces to the legacy proportional-to-capacity split exactly.
        dir_move = torch.where(pos, self._sign, -self._sign)         # (B,6) P-space sign of the move
        need = delta.abs()
        m_mag = torch.zeros_like(cap)
        for _ in range(3):
            avail = (cap - m_mag).clamp_min(0.0)
            w_avail = avail * self._alloc_w
            tot = w_avail.sum(-1, keepdim=True)                       # (B,1)
            rem = (need - m_mag.sum(-1)).clamp_min(0.0).unsqueeze(-1)
            add = torch.minimum(avail, torch.where(tot > 1e-9, w_avail / tot.clamp_min(1e-9),
                                                   torch.zeros_like(w_avail)) * rem)
            m_mag = m_mag + add
        m = m_mag * dir_move                                          # (B,6) reprojection move
        p_on = p + m

        # remaining ramp/floor budget for the free step (keeps total move within ramps)
        rem_up = (r_up_eff - m.clamp_min(0.0)).clamp_min(0.0)
        rem_dn = torch.minimum(r_dn_eff - (-m).clamp_min(0.0), p_on).clamp_min(0.0)

        # --- free tangential mix step (sum-preserving over flexible channels) ---
        r = r * self._free                                            # freeze passthrough at persistence
        if self.include_floor:
            r = r.masked_fill((p_on <= self.eps_mw + 1e-6) & (r < 0), 0.0)
        # project onto the reduced plane (orthogonal to SIGN over free channels);
        # sign_free is 0 on passthrough so r stays 0 there and Sum_free is preserved
        r = r - (r * self._sign_free).sum(-1, keepdim=True) / (self._sign_free * self._sign_free).sum() * self._sign_free
        big = torch.full_like(r, 1e9)
        wall_up = torch.where(r > 1e-9, rem_up / r.clamp_min(1e-9), big)
        wall_dn = torch.where(r < -1e-9, rem_dn / (-r).clamp_min(1e-9), big)
        alpha = torch.minimum(wall_up, wall_dn).amin(dim=1)
        alpha = torch.where(alpha >= 1e9, torch.zeros_like(alpha), alpha)

        y = p_on + torch.sigmoid(s).unsqueeze(-1) * alpha.unsqueeze(-1) * r
        return (y - self._y_mean) / self._y_scale                     # (B,6) y-scaled


def make_rayen(base_arch: str, x_scaler, y_scaler, ramp_up, ramp_dn,
               nd_feat_idx: int = 6, n_features: int = 17,
               include_floor: bool = True, fix_demand: bool = False,
               passthrough_idx=(2,), alloc_weights=None) -> nn.Module:
    """lstm_rayen: plain LSTM backbone. itransformer_rayen: input-RevIN
    iTransformer with output denorm off. fix_demand=False emits 8 raw outputs
    (D_t + 6 dir + s, RayenHead); fix_demand=True emits 7 (6 dir + s,
    RayenHeadFixedD -- D is pinned to nd(t-1), not predicted).
    passthrough_idx (fix_demand only): TARGETS indices frozen at persistence;
    default (2,) = gas_steam, the inflexible peaker WAPE regresses on."""
    n_out = 7 if fix_demand else 8
    if base_arch == "lstm":
        base = LSTMForecaster(n_features=n_features, n_targets=n_out, hidden=128, layers=2, dropout=0.2)
    elif base_arch == "itransformer":
        tfi = [0, 3, 1, 2, 4, 5, 6] if fix_demand else [0, 3, 1, 2, 4, 5, 6, 7]
        base = iTransformer(seq_len=288, n_features=n_features, n_targets=n_out,
                            target_feat_idx=tfi,
                            d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1,
                            use_revin=True, revin_out=False)
    else:
        raise ValueError(f"unsupported rayen base {base_arch!r}")
    if fix_demand:
        return RayenHeadFixedD(base, x_scaler.mean_, x_scaler.scale_, y_scaler.mean_, y_scaler.scale_,
                               ramp_up, ramp_dn, nd_feat_idx=nd_feat_idx, include_floor=include_floor,
                               passthrough_idx=passthrough_idx, alloc_weights=alloc_weights)
    return RayenHead(base, x_scaler.mean_, x_scaler.scale_, y_scaler.mean_, y_scaler.scale_,
                     ramp_up, ramp_dn, nd_feat_idx=nd_feat_idx, include_floor=include_floor)


def make_task7(base_arch: str, n_features: int = 17, nd_feat_idx: int = 6) -> nn.Module:
    """7-output task nets (D_t, P_1..P_6) for the decision-rules approach
    (lib/decision_rule.py): trained for accuracy as usual on the y7 targets
    ([scaled nd_t, 6 scaled dispatch], same supervision as the rayen arm).
    lstm_task7 = plain LSTM; itransformer_task7 = full-RevIN iTransformer
    (outputs are levels here, not directions, so output denorm stays ON)."""
    if base_arch == "lstm":
        return LSTMForecaster(n_features=n_features, n_targets=7, hidden=128, layers=2, dropout=0.2)
    if base_arch == "itransformer":
        return iTransformer(seq_len=288, n_features=n_features, n_targets=7,
                            target_feat_idx=[nd_feat_idx] + TARGET_FEAT_IDX,
                            d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1,
                            use_revin=True)
    raise ValueError(f"unsupported task7 base {base_arch!r}")


class PersistenceForecaster(nn.Module):
    """Naive 5-min persistence as a drop-in (B,T,C) -> (B,6) module: returns the
    six dispatch-history channels at the window's LAST step, re-expressed in
    y_scaler space. Stage-0 finding (constraint_research.md): at h=1 this beats
    every learned model (WAPE 0.0956) and its signed sum equals nd(t-1) exactly,
    so anchor(persistence) is the zero-parameter fully-responsive baseline. In
    closed loop it feeds back its own output and freezes the mix -- the neural
    backbones must beat it there to justify themselves.
    """

    def __init__(self, x_mean, x_scale, y_mean, y_scale):
        super().__init__()
        idx = torch.tensor(TARGET_FEAT_IDX, dtype=torch.long)
        self.register_buffer("_idx", idx)
        self.register_buffer("_x_mean", torch.tensor(np.asarray(x_mean)[TARGET_FEAT_IDX], dtype=torch.float32))
        self.register_buffer("_x_scale", torch.tensor(np.asarray(x_scale)[TARGET_FEAT_IDX], dtype=torch.float32))
        self.register_buffer("_y_mean", torch.tensor(np.asarray(y_mean), dtype=torch.float32))
        self.register_buffer("_y_scale", torch.tensor(np.asarray(y_scale), dtype=torch.float32))

    def forward(self, x):
        last = x[:, -1, :].index_select(1, self._idx)          # (B,6) x-scaled, TARGETS order
        mw = last * self._x_scale + self._x_mean
        return (mw - self._y_mean) / self._y_scale


# -------------------------------- non-neural --------------------------------

class LinearForecaster:
    """Closed-form linear regression on flattened lookback windows."""

    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()

    def fit(self, Xtr, Ytr, **_):
        self.model.fit(Xtr.reshape(len(Xtr), -1), Ytr)
        return self

    def predict(self, X):
        return self.model.predict(X.reshape(len(X), -1))


class XGBoostForecaster:
    """One xgb.XGBRegressor per target, early-stopped on validation."""

    def __init__(self, n_estimators=1000, max_depth=8, learning_rate=0.05,
                 subsample=0.9, colsample_bytree=0.9, early_stopping_rounds=20):
        self.kwargs = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
            tree_method="hist", n_jobs=-1, verbosity=0,
        )
        self.models = []
        self.val_rmse_per_target = []

    def fit(self, Xtr, Ytr, Xva=None, Yva=None):
        import xgboost as xgb
        Xtrf = Xtr.reshape(len(Xtr), -1)
        Xvaf = Xva.reshape(len(Xva), -1) if Xva is not None else None
        self.models = []
        self.val_rmse_per_target = []
        for i in range(Ytr.shape[1]):
            m = xgb.XGBRegressor(**self.kwargs)
            eval_set = [(Xvaf, Yva[:, i])] if Xvaf is not None else None
            m.fit(Xtrf, Ytr[:, i], eval_set=eval_set, verbose=False)
            self.models.append(m)
            if eval_set is not None:
                self.val_rmse_per_target.append(m.evals_result()["validation_0"]["rmse"])
        return self

    def predict(self, X):
        Xf = X.reshape(len(X), -1)
        return np.stack([m.predict(Xf) for m in self.models], axis=1)


# -------------------------------- registry --------------------------------

@dataclass
class TrainSpec:
    epochs: int = 200
    patience: int = 20
    batch: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-5


NEURAL_ARCHES = {
    "lstm": LSTMForecaster,
    "bi_lstm": BiLSTMForecaster,
    "gru": GRUForecaster,
    "cnn_lstm": CNNLSTMForecaster,
    "timexer": TimeXer,
    "itransformer": iTransformer,
    "itransformer_norevin": iTransformer,   # iTransformer with internal RevIN disabled
    "patchtst": PatchTST,
    # RevIN-wrapped recurrent variants
    "lstm_selrevin": LSTMForecaster,      # selective RevIN: demand channels passthrough
    "lstm_dishts": LSTMForecaster,        # Dish-TS: learned own-history horizon level
    "lstm_dishtsx": LSTMForecaster,       # Dish-TS: cross-channel level (may use demand)
    "lstm_dishtsd": LSTMForecaster,       # Dish-TS: exogenous-only level (forced demand)
    "lstm_revin": LSTMForecaster,
    "bi_lstm_revin": BiLSTMForecaster,
    "gru_revin": GRUForecaster,
    # architectural-trick GRU variants
    "gru_decomp": SeriesDecompGRU,        # RevIN + Autoformer decomposition
    "gru_nlin": GRUForecaster,            # NLinear shift (no RevIN)
    "gru_revin_affine": GRUForecaster,    # Learnable RevIN (Dish-TS style)
}

NON_NEURAL = {"linear": LinearForecaster, "xgboost": XGBoostForecaster}

ALL_ARCHES = list(NEURAL_ARCHES.keys()) + list(NON_NEURAL.keys())


def make_neural(arch: str, n_features=16, n_targets=6) -> nn.Module:
    """Build a neural forecaster by short name with benchmark defaults."""
    # RevIN + Autoformer decomposition GRU
    if arch == "gru_decomp":
        base = SeriesDecompGRU(n_features=n_features, n_targets=n_targets,
                               kernel=25, hidden=128, layers=2, dropout=0.2)
        return RevIN(base, TARGET_FEAT_IDX)
    # NLinear-style shift wrapping a plain GRU
    if arch == "gru_nlin":
        base = make_neural("gru", n_features=n_features, n_targets=n_targets)
        return NLinearShift(base, TARGET_FEAT_IDX)
    # Learnable affine RevIN wrapping a plain GRU
    if arch == "gru_revin_affine":
        base = make_neural("gru", n_features=n_features, n_targets=n_targets)
        return LearnableRevIN(base, TARGET_FEAT_IDX, n_features=n_features)
    # Selective RevIN: demand drivers pass through, everything else RevIN-normed
    if arch == "lstm_selrevin":
        base = make_neural("lstm", n_features=n_features, n_targets=n_targets)
        return SelectiveRevIN(base, TARGET_FEAT_IDX, DEMAND_FEAT_IDX, n_features)
    # Dish-TS: learned (predicted) horizon level instead of RevIN's copied mean
    if arch in ("lstm_dishts", "lstm_dishtsx", "lstm_dishtsd"):
        base = make_neural("lstm", n_features=n_features, n_targets=n_targets)
        mode = {"lstm_dishts": "own", "lstm_dishtsx": "cross", "lstm_dishtsd": "exo"}[arch]
        return DishTS(base, TARGET_FEAT_IDX, n_features, xi_mode=mode)
    if arch.endswith("_revin"):
        base_arch = arch[: -len("_revin")]
        base = make_neural(base_arch, n_features=n_features, n_targets=n_targets)
        return RevIN(base, TARGET_FEAT_IDX)
    if arch in ("lstm", "bi_lstm", "gru"):
        return NEURAL_ARCHES[arch](n_features=n_features, n_targets=n_targets,
                                   hidden=128, layers=2, dropout=0.2)
    if arch == "cnn_lstm":
        return CNNLSTMForecaster(n_features=n_features, n_targets=n_targets,
                                 hidden=128, layers=2, dropout=0.2,
                                 conv_channels=64, kernel=3)
    if arch == "timexer":
        return TimeXer(seq_len=288, n_features=n_features, n_targets=n_targets,
                       target_feat_idx=TARGET_FEAT_IDX,
                       patch_len=24, d_model=128, n_heads=4, n_layers=2,
                       d_ff=256, dropout=0.1)
    if arch in ("itransformer", "itransformer_norevin"):
        return iTransformer(seq_len=288, n_features=n_features, n_targets=n_targets,
                            target_feat_idx=TARGET_FEAT_IDX,
                            d_model=128, n_heads=4, n_layers=3,
                            d_ff=256, dropout=0.1,
                            use_revin=(arch == "itransformer"))
    if arch == "patchtst":
        return PatchTST(seq_len=288, n_features=n_features, n_targets=n_targets,
                        target_feat_idx=TARGET_FEAT_IDX,
                        patch_len=16, stride=8, d_model=128, n_heads=4,
                        n_layers=3, d_ff=256, dropout=0.1)
    raise ValueError(f"unknown neural arch {arch!r}")


def make_model(arch: str):
    """Build any model (neural or not) by short name with benchmark defaults."""
    if arch in NEURAL_ARCHES:
        return make_neural(arch)
    if arch == "linear":
        return LinearForecaster()
    if arch == "xgboost":
        return XGBoostForecaster()
    raise ValueError(f"unknown arch {arch!r}")
