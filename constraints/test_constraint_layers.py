"""Feasibility unit tests for the hard-constraint output layers (plan.md).

The guarantees are architectural, so they must hold at ANY weights -- every
test runs on randomly initialised backbones over real input windows:

  RayenHead          balance exact, ramps within eps-anchor tol, floors >= 0
  DecisionRuleHead   same set via projection + safe blend; alpha in [0,1];
                     survives adversarial task outputs; no-op when feasible
  safe-network LP    t* > 0, Monte-Carlo feasibility of Fx over the polytope

Run directly (python3 constraints/test_constraint_layers.py) or via pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "ml"))
import pipeline                  # noqa: E402
import models as M               # noqa: E402
import decision_rule as dr       # noqa: E402
from check_caps import RAMPS, CAPS  # noqa: E402

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]
SIGN = np.array([1, 1, 1, 1, -1, 1])
BAL_TOL = 0.05                   # MW, float32 arithmetic on ~GW magnitudes
RAMP_TOL = 0.6                   # eps-anchor slack + noise (constraint metric tol)
NEG_TOL = 0.1
N_WIN = 512

_cache = {}


def _ctx():
    if not _cache:
        data = pipeline.load_prepared("5min", 24, 1, "net_dispatch_totdem", "hist")
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(len(data["Xte"]), N_WIN, replace=False))
        X = np.ascontiguousarray(data["Xte"][pick])
        xs, ys = data["x_scaler"], data["y_scaler"]
        _cache.update(
            X=torch.from_numpy(X), xs=xs, ys=ys,
            nd_idx=data["feat_cols"].index("net_demand"),
            n_features=len(data["feat_cols"]),
            ramp_up=np.array([RAMPS[t][1] for t in TARGETS]),
            ramp_dn=np.array([abs(RAMPS[t][0]) for t in TARGETS]),
            prev=(X[:, -1, M.TARGET_FEAT_IDX] * xs.scale_[M.TARGET_FEAT_IDX]
                  + xs.mean_[M.TARGET_FEAT_IDX]).astype(np.float64),
        )
        c = _cache
        span = 12000.0
        c["fit"] = dr.fit_safe_F(c["ramp_up"], c["ramp_dn"],
                                 [1.3 * CAPS[t] for t in TARGETS], -2000.0, span)
    return _cache


def _split(out, c):
    d_mw = out[:, 0].numpy().astype(np.float64) * c["xs"].scale_[c["nd_idx"]] \
        + c["xs"].mean_[c["nd_idx"]]
    p_mw = c["ys"].inverse_transform(out[:, 1:].numpy()).astype(np.float64)
    return d_mw, p_mw


def _assert_feasible(d_mw, p_mw, c, tag):
    bal = np.abs(p_mw @ SIGN - d_mw).max()
    assert bal <= BAL_TOL, f"{tag}: balance residual {bal:.4f} MW"
    delta = p_mw - c["prev"]
    up = (delta - (c["ramp_up"] + RAMP_TOL)).max()
    dn = (-delta - (c["ramp_dn"] + RAMP_TOL)).max()
    assert up <= 0 and dn <= 0, f"{tag}: ramp excess up={up:.3f} dn={dn:.3f} MW"
    neg = p_mw.min()
    assert neg >= -NEG_TOL, f"{tag}: negative dispatch {neg:.3f} MW"
    print(f"  {tag}: bal_max={bal:.4f} MW, ramp/floor OK (min P={neg:.4f})")


def _rayen(base_arch, c, seed):
    torch.manual_seed(seed)
    return M.make_rayen(base_arch, c["xs"], c["ys"], c["ramp_up"], c["ramp_dn"],
                        nd_feat_idx=c["nd_idx"], n_features=c["n_features"]).eval()


def _dr_head(base, c):
    return dr.DecisionRuleHead(base, c["fit"]["F"], c["xs"].mean_, c["xs"].scale_,
                               c["ys"].mean_, c["ys"].scale_, c["ramp_up"], c["ramp_dn"],
                               nd_feat_idx=c["nd_idx"]).eval()


class _ConstBase(nn.Module):
    """Task net emitting one fixed (adversarial) scaled output row."""

    def __init__(self, row):
        super().__init__()
        self.register_buffer("row", torch.tensor(row, dtype=torch.float32))

    def forward(self, x):
        return self.row.expand(len(x), -1)


class _PersistBase(nn.Module):
    """Task net whose output IS the feasible persistence point (alpha must be 0)."""

    def __init__(self, c):
        super().__init__()
        xs, ys = c["xs"], c["ys"]
        tfi = list(M.TARGET_FEAT_IDX)
        self.register_buffer("xm", torch.tensor(xs.mean_[tfi], dtype=torch.float32))
        self.register_buffer("xsd", torch.tensor(xs.scale_[tfi], dtype=torch.float32))
        self.register_buffer("ym", torch.tensor(ys.mean_, dtype=torch.float32))
        self.register_buffer("ysd", torch.tensor(ys.scale_, dtype=torch.float32))
        self.register_buffer("sign", torch.tensor(SIGN, dtype=torch.float32))
        self.register_buffer("idx", torch.tensor(tfi, dtype=torch.long))
        self.ndm, self.nds = float(xs.mean_[c["nd_idx"]]), float(xs.scale_[c["nd_idx"]])

    def forward(self, x):
        prev = x[:, -1, :].index_select(1, self.idx) * self.xsd + self.xm
        d = (prev * self.sign).sum(-1, keepdim=True)
        return torch.cat([(d - self.ndm) / self.nds, (prev - self.ym) / self.ysd], dim=1)


def test_lp_certificate():
    c = _ctx()
    fit = c["fit"]
    assert fit["t"] > 1.0, f"safe LP t*={fit['t']} not strictly interior"
    assert fit["mc_min_slack"] >= fit["t"] - 1e-6, "MC found slack below t*"
    assert fit["mc_eq_max"] < 1e-8, "safe net off the balance plane"
    print(f"  LP: t*={fit['t']:.2f} MW, mc_min_slack={fit['mc_min_slack']:.2f}, "
          f"mc_eq_max={fit['mc_eq_max']:.1e}")


def test_rayen_random_weights():
    c = _ctx()
    for arch in ("lstm", "itransformer"):
        for seed in (0, 1):
            with torch.no_grad():
                out = _rayen(arch, c, seed)(c["X"])
            _assert_feasible(*_split(out, c), c, f"rayen/{arch}/s{seed}")


def test_rayen_gradients():
    c = _ctx()
    model = _rayen("lstm", c, 0).train()
    out = model(c["X"][:32])
    out.pow(2).mean().backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert sum(g > 0 for g in grads) > 2, "no gradients flow through RayenHead"
    print(f"  rayen gradients: {sum(g > 0 for g in grads)}/{len(grads)} params nonzero")


def test_dr_random_weights():
    c = _ctx()
    for arch in ("lstm", "itransformer"):
        torch.manual_seed(0)
        base = M.make_task7(arch, n_features=c["n_features"], nd_feat_idx=c["nd_idx"])
        head = _dr_head(base, c)
        with torch.no_grad():
            out = head(c["X"])
        a = head.last_alpha.numpy()
        assert (a >= 0).all() and (a <= 1).all(), "alpha outside [0,1]"
        assert (head.last_sn_min.numpy() > 0).all(), "input escaped the certified polytope"
        _assert_feasible(*_split(out, c), c, f"DR/{arch} (alpha act {(a > 1e-6).mean():.0%})")


def test_dr_adversarial():
    c = _ctx()
    for k, row in enumerate(([1e4] * 7, [-1e4] * 7,
                             [0, 1e5, -1e5, 1e5, -1e5, 1e5, -1e5])):
        head = _dr_head(_ConstBase(row), c)
        with torch.no_grad():
            out = head(c["X"])
        _assert_feasible(*_split(out, c), c, f"DR/adversarial{k} "
                         f"(alpha max {head.last_alpha.max():.3f})")


def test_dr_noop_when_feasible():
    c = _ctx()
    base = _PersistBase(c)
    head = _dr_head(base, c)
    with torch.no_grad():
        raw = base(c["X"])
        out = head(c["X"])
    # the persistence point sits ON the floor wall (idle gas_steam), so float32
    # scaler round trips can push it ~1e-4 MW outside; alpha may fire at the
    # matching epsilon scale but must stay negligible and the output must not move
    a_max = float(head.last_alpha.max())
    assert a_max < 1e-3, f"alpha {a_max} fired materially on a feasible point"
    drift = (out - raw).abs().max().item()
    assert drift < 1e-3, f"feasible point moved by {drift} (projection should be identity)"
    print(f"  DR no-op: alpha_max={a_max:.1e}, max drift {drift:.2e}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print(f"\nALL {len(tests)} TESTS PASSED")
