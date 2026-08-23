"""EXPERIMENT 6 -- exp5 repeated with a ONE-DAY gap instead of 3 hours.

A whole missing day cannot be posed inside a single 288-step tensor: there are no gap
edges. So the unit here is a 3-DAY window (864 steps) and the middle day's 10 target
channels are masked. demand, price and calendar stay visible throughout, as in exp5.

This is the regime where interpolation should stop being unbeatable: a straight line
across 24 hours cannot reproduce a diurnal cycle, whereas the optimizer's balance target
is built from OBSERVED demand and therefore still carries the day's shape.

Arms (one fold, one seed -- this is a probe, not the full matrix):
    interp     linear interpolation across the 288-step gap
    persist    copy the previous day  (the honest baseline for a daily gap)
    ren        renewables-only BiLSTM: wind, solar_utility and both curtailment
               channels. Weather-driven, no economics -- a separable stage-1 model.
    opt        TWO-STAGE: stage-1 `ren` for the 4 weather channels + the constrained
               dispatch QP for the 6 dispatchables. No interpolation anywhere in it.
    opt_iren   the old single-stage version, QP dispatchables + INTERPOLATED
               renewables, kept only so the fix can be scored against it
    base       BiLSTM imputer, UNCONSTRAINED -- nothing enforces box/ramp/balance
    opt_in     BiLSTM imputer + the QP solution as 6 extra input channels
    hardnet    base, projected onto the feasible set by planB/heads.py hardnet_head
               (exact Euclidean projection: box, ramp, both gap seams, balance plane)

Ramps are the R(k) DURATION envelope from exp4, not a single per-step box. planB's own
limits are overridden with the same envelope's k=1 row so hardnet and the QP are held to
one standard rather than two.

    python colab/capacity_experiment/exp6_oneday_gap.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (CAL, OBS, STEPS_PER_DAY, TARGETS,  # noqa: E402
                      load_table, make_days, metrics)

HERE = Path(__file__).resolve().parent
DISP = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
        "battery_charging", "battery_discharging"]
SIGN = np.array([1., 1., 1., 1., -1., 1.])
SLOPE = np.array([0.224, 0.256, 0.042, 0.100, 0.150, 0.150])
PMED = np.array([91., 65., 171., 148., 11., 97.])

GAPD = STEPS_PER_DAY                      # 288 = one day
W = 3 * STEPS_PER_DAY                     # 864 = three days
G0, G1 = STEPS_PER_DAY, 2 * STEPS_PER_DAY
VAL_DAYS, TEST_DAYS = 45, 60
HIDDEN, LAYERS, DROPOUT = 128, 2, 0.2
LR, WEIGHT_DECAY, BATCH = 1e-3, 1e-4, 32
EPOCHS, PATIENCE = 60, 8
SEED = 0
QP_OUTER, QP_INNER = 30, 80

_mps = getattr(torch.backends, "mps", None)
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if (_mps is not None and _mps.is_available()) else "cpu")

cols = TARGETS + OBS + CAL
X, days = make_days(load_table(), cols)
ti = {c: i for i, c in enumerate(cols)}
di = [ti[t] for t in DISP]
tgt_all = [ti[t] for t in TARGETS]
dem_i = ti["demand"]

# consecutive-day triples only
ok = [i for i in range(len(days) - 2)
      if (days[i + 1] - days[i]).days == 1 and (days[i + 2] - days[i + 1]).days == 1]
ok = np.array(ok)
Xw = np.concatenate([X[ok], X[ok + 1], X[ok + 2]], axis=1)      # (N, 864, C)
mid = days[ok + 1]
N = len(Xw)
print(f"3-day windows: {N} (middle days {mid[0].date()}..{mid[-1].date()})  device={DEVICE}")

o = N - VAL_DAYS - TEST_DAYS
tr = np.arange(0, o)
va = np.arange(o, o + VAL_DAYS)
te = np.arange(o + VAL_DAYS, o + VAL_DAYS + TEST_DAYS)
print(f"train {len(tr)}  val {len(va)}  test {len(te)}  "
      f"({mid[te[0]].date()}..{mid[te[-1]].date()})")

# ---------------------------------------------------------------- cost parameters
flat = Xw[tr][:, :, di].reshape(-1, len(DISP))
cap, mu = flat.max(0), flat.mean(0)
B_ = 1.0 / SLOPE
A_ = PMED - B_ * mu
A_[DISP.index("battery_charging")] *= -1
d1 = np.diff(Xw[tr][:, :, di], axis=1).reshape(-1, len(DISP))
R_UP = np.percentile(d1, 99.9, axis=0)
R_DN = -np.percentile(d1, 0.1, axis=0)

# the RAMP-DURATION envelope R(k) from exp4, replacing the single per-step box.
# A per-step limit is 3-12x too loose over multi-step horizons, and a 288-step solve
# exploits that far more than a 36-step one does.
_env = json.loads((HERE / "exp4_ramp_envelope_results.json").read_text())
KS = [k for k in _env["enforce"]]
RK_UP = np.array([[_env["envelope"][t][str(k)]["p999_up"] for t in DISP] for k in KS])
RK_DN = np.array([[-_env["envelope"][t][str(k)]["p999_dn"] for t in DISP] for k in KS])
print(f"R(k) envelope, k={KS} steps ({[k*5 for k in KS]} min)")
for j, k in enumerate(KS):
    print(f"  k={k:2d}  up " + " ".join(f"{x:7.1f}" for x in RK_UP[j]))

ND = (Xw[:, :, di] * SIGN).sum(-1)
w_lin = np.arange(1, GAPD + 1) / (GAPD + 1)


def gap_interp(v):
    L, R = v[:, G0 - 1], v[:, G1]
    if v.ndim == 3:
        return L[:, None, :] * (1 - w_lin[None, :, None]) + R[:, None, :] * w_lin[None, :, None]
    return L[:, None] * (1 - w_lin[None, :]) + R[:, None] * w_lin[None, :]


nd_dev = (ND[tr][:, G0:G1] - gap_interp(ND[tr])).ravel()
dm_dev = (Xw[tr][:, G0:G1, dem_i] - gap_interp(Xw[tr][:, :, dem_i])).ravel()
GAMMA = float(dm_dev @ nd_dev / (dm_dev @ dm_dev))
print(f"gamma = {GAMMA:.3f}")


def solve_qp(pL, pR, nd_t):
    t_ = lambda z: torch.tensor(np.asarray(z), dtype=torch.float64)
    a, b, capT = t_(A_), t_(B_), t_(cap)
    ru, rd, s = t_(R_UP), t_(R_DN), t_(SIGN)
    rku, rkd = t_(RK_UP), t_(RK_DN)
    ndT, L, R = t_(nd_t), t_(pL), t_(pR)
    ww = torch.tensor(w_lin, dtype=torch.float64)
    p = (L[:, None, :] * (1 - ww[None, :, None]) + R[:, None, :] * ww[None, :, None]).clone()
    p.requires_grad_(True)
    lam = torch.zeros_like(ndT)
    opt = torch.optim.Adam([p], lr=2.0)
    rho = 1.0
    for _ in range(QP_OUTER):
        for _ in range(QP_INNER):
            opt.zero_grad()
            cost = (a * p + 0.5 * b * p ** 2).sum(-1).sum(-1)
            bal = (p * s).sum(-1) - ndT
            traj = torch.cat([L[:, None, :], p, R[:, None, :]], 1)
            vr = 0.0
            for j_, k_ in enumerate(KS):
                dk = traj[:, k_:, :] - traj[:, :-k_, :]
                vr = vr + (torch.relu(dk - rku[j_]) ** 2
                           + torch.relu(-dk - rkd[j_]) ** 2).sum(-1).sum(-1)
            vb = (torch.relu(-p) ** 2 + torch.relu(p - capT) ** 2).sum(-1).sum(-1)
            (cost + (lam * bal).sum(-1) + 0.5 * rho * (bal ** 2).sum(-1)
             + 0.5 * rho * 1e3 * (vr + vb)).mean().backward()
            opt.step()
        with torch.no_grad():
            lam += rho * ((p * s).sum(-1) - ndT)
            rho *= 1.3
    P = p.detach().numpy()
    bal = np.abs((P * SIGN).sum(-1) - nd_t)
    traj = np.concatenate([pL[:, None, :], P, pR[:, None, :]], 1)
    over = 0.0
    for j_, k_ in enumerate(KS):
        dk = traj[:, k_:, :] - traj[:, :-k_, :]
        over = max(over, float((np.maximum(dk - RK_UP[j_], 0)
                                + np.maximum(-dk - RK_DN[j_], 0)).max()))
    print(f"  QP: balance mean {bal.mean():.3f} max {bal.max():.2f} MW | "
          f"ramp over {over:.3f} | box {max(float(np.maximum(-P,0).max()), float(np.maximum(P-cap,0).max())):.3f}",
          flush=True)
    return P.astype(np.float32)


print("\nsolving the day-long QP for every window ...", flush=True)
t0 = time.time()
nd_hat = gap_interp(ND) + GAMMA * (Xw[:, G0:G1, dem_i] - gap_interp(Xw[:, :, dem_i]))
OPT = solve_qp(Xw[:, G0 - 1][:, di], Xw[:, G1][:, di], nd_hat)
print(f"  done in {time.time() - t0:.0f}s", flush=True)


class BiLSTMImputer(nn.Module):
    def __init__(self, n_in, n_tgt):
        super().__init__()
        self.rnn = nn.LSTM(n_in + 1, HIDDEN, LAYERS, batch_first=True,
                           bidirectional=True, dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN * 2, n_tgt)

    def forward(self, x, mask):
        h, _ = self.rnn(torch.cat([x, mask], dim=-1))
        return self.head(h)


flat_tr = Xw[tr].reshape(-1, Xw.shape[2]).astype(np.float64)
mean, std = flat_tr.mean(0), flat_tr.std(0)
std_safe = np.where(std <= 1e-9 * (np.abs(mean) + 1.0), 1.0, std)
Xs = Xw.astype(np.float32).copy()
for cn in TARGETS + OBS:
    j = ti[cn]
    Xs[:, :, j] = (Xw[:, :, j] - mean[j]) / std_safe[j]
y_mean = mean[tgt_all].astype(np.float32)
y_std = std_safe[tgt_all].astype(np.float32)
omean, ostd = mean[di].astype(np.float32), std_safe[di].astype(np.float32)


def batch(idx, use_opt):
    xin = Xs[idx].copy()
    msk = np.ones((len(idx), W, 1), np.float32)
    xin[:, G0:G1][:, :, tgt_all] = 0.0
    msk[:, G0:G1, 0] = 0.0
    if not use_opt:
        return xin.astype(np.float32), msk
    ext = Xw[idx][:, :, di].astype(np.float32).copy()
    ext[:, G0:G1] = OPT[idx]
    ext = (ext - omean) / ostd
    return np.concatenate([xin, ext], -1).astype(np.float32), msk


def run(use_opt, tag, out_cols=None):
    """out_cols: names of the channels this model predicts (default: all TARGETS)."""
    out_cols = list(TARGETS) if out_cols is None else list(out_cols)
    oi = [TARGETS.index(c) for c in out_cols]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = BiLSTMImputer(Xw.shape[2] + (len(DISP) if use_opt else 0),
                          len(out_cols)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    vin, vmk = batch(va, use_opt)
    vin_t, vmk_t = torch.from_numpy(vin).to(DEVICE), torch.from_numpy(vmk).to(DEVICE)
    vy = torch.from_numpy(Xs[va][:, G0:G1][:, :, [tgt_all[i] for i in oi]]).to(DEVICE)
    best, bstate, waited, ep = 1e9, None, 0, 0
    for ep in range(1, EPOCHS + 1):
        rng = np.random.default_rng(1000 + ep)
        order = rng.permutation(len(tr))
        model.train()
        for i in range(0, len(tr), BATCH):
            bi = tr[order[i:i + BATCH]]
            xin, msk = batch(bi, use_opt)
            xb = torch.from_numpy(xin).to(DEVICE)
            mb = torch.from_numpy(msk).to(DEVICE)
            yb = torch.from_numpy(
                Xs[bi][:, G0:G1][:, :, [tgt_all[i] for i in oi]]).to(DEVICE)
            opt.zero_grad()
            ((model(xb, mb)[:, G0:G1] - yb) ** 2).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = ((model(vin_t, vmk_t)[:, G0:G1] - vy) ** 2).mean().item()
        if vl < best - 1e-6:
            best, waited = vl, 0
            bstate = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        if waited >= PATIENCE:
            break
    model.load_state_dict(bstate)
    xin, msk = batch(te, use_opt)
    with torch.no_grad():
        pr = model(torch.from_numpy(xin).to(DEVICE),
                   torch.from_numpy(msk).to(DEVICE)).cpu().numpy()
    print(f"  [{tag}] ep{ep} val={best:.4f}", flush=True)
    return pr[:, G0:G1] * y_std[oi] + y_mean[oi]


Y = Xw[te][:, G0:G1][:, :, tgt_all]
P = {}
P["interp"] = gap_interp(Xw[te][:, :, tgt_all])
P["persist"] = Xw[te][:, :G0][:, :, tgt_all]                    # previous day, as-is
P["opt"] = P["interp"].copy()
P["opt"][:, :, [TARGETS.index(t) for t in DISP]] = OPT[te]
print()
P["base"] = run(False, "base")
P["opt_in"] = run(True, "opt_in")

# ---- stage 1: renewables-only imputer (weather channels, no economics) -----------
REN = ["wind", "solar_utility", "curtailment_wind", "curtailment_solar"]
rix = [TARGETS.index(c) for c in REN]
P["ren"] = P["interp"].copy()
P["ren"][:, :, rix] = run(False, "ren", out_cols=REN)

# ---- the TWO-STAGE optimization arm: stage-1 renewables + QP dispatchables -------
P["opt_iren"] = P["opt"]                      # old version, kept for the comparison
P["opt"] = P["ren"].copy()
P["opt"][:, :, [TARGETS.index(t) for t in DISP]] = OPT[te]

# ---- hardnet: project the UNCONSTRAINED base onto the feasible set ---------------
sys.path.insert(0, str(HERE.parent.parent / "planB"))
import limits as _pl                                                    # noqa: E402
_pl.R_UP = RK_UP[0].astype(np.float32)      # k=1 row of the same envelope the QP uses
_pl.R_DN = RK_DN[0].astype(np.float32)
_pl.CAP = cap.astype(np.float32)
_pl._RUP_T = torch.tensor(_pl.R_UP, dtype=torch.float32)
_pl._RDN_T = torch.tensor(_pl.R_DN, dtype=torch.float32)
_pl._CAP_T = torch.tensor(_pl.CAP, dtype=torch.float32)
_pl.limits = lambda dev, dt: (_pl._SIGN_T.to(dev, dt), _pl._RUP_T.to(dev, dt),
                              _pl._RDN_T.to(dev, dt), _pl._CAP_T.to(dev, dt))
import heads as _hd                                                     # noqa: E402
_hd._limits = _pl.limits
print(f"\nhardnet limits overridden with the exp4 k=1 envelope:")
print(f"  r_up {np.round(_pl.R_UP, 1).tolist()}")
print(f"  r_dn {np.round(_pl.R_DN, 1).tolist()}")

_t = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)
_raw = _t(P["base"][:, :, [TARGETS.index(c) for c in DISP]])
_HN, _dbg = _hd.hardnet_head(_raw, _t(Xw[te][:, G0 - 1][:, di]),
                             _t(Xw[te][:, G1][:, di]), _t(nd_hat[te]),
                             return_debug=True)
P["hardnet"] = P["base"].copy()
P["hardnet"][:, :, [TARGETS.index(c) for c in DISP]] = _HN.numpy()
print(f"  projection moved {_dbg['correction'].mean().item():.1f} MW/step on average; "
      f"balance shortfall max {_dbg['bal_short'].max().item():.2f} MW")

peak = Xw[te][:, G0:G1, dem_i].max(1)
t20 = np.argsort(-peak)[:20]
res = {}
print("\n" + "=" * 96)
print("ONE-DAY GAP  (1 fold, 1 seed)")
print("=" * 96)
print(f"  {'arm':10s} {'macroWAPE':>10s} {'microWAPE':>10s} {'macroR2':>9s} "
      f"{'top20WAPE':>10s} {'top20micro':>11s} {'top20R2':>9s} {'MAE':>8s}")
for nm in ("interp", "persist", "opt_iren", "ren", "opt", "base", "opt_in", "hardnet"):
    per, mac = metrics(P[nm], Y)
    _, m20 = metrics(P[nm][t20], Y[t20])
    res[nm] = {"macroWAPE": mac["WAPE"], "microWAPE": mac["microWAPE"],
               "macroR2": mac["R2"], "top20WAPE": m20["WAPE"],
               "top20micro": m20["microWAPE"], "top20R2": m20["R2"],
               "MAE": mac["MAE"], **{t: per[t]["MAE"] for t in TARGETS}}
    r = res[nm]
    print(f"  {nm:10s} {r['macroWAPE']:10.4f} {r['microWAPE']:10.4f} {r['macroR2']:9.4f} "
          f"{r['top20WAPE']:10.4f} {r['top20micro']:11.4f} {r['top20R2']:9.4f} "
          f"{r['MAE']:8.2f}")
print(f"\n  per-channel MAE (MW)")
print(f"  {'arm':10s} " + "".join(f"{t[:9]:>10s}" for t in TARGETS))
for nm in res:
    print(f"  {nm:10s} " + "".join(f"{res[nm][t]:10.0f}" for t in TARGETS))

(HERE / "exp6_oneday_results.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(HERE / "exp6_stack_data.npz",
                    actual=Y, t20=t20, **{k: v for k, v in P.items()})
print("\nwrote exp6_oneday_results.json, exp6_stack_data.npz")
