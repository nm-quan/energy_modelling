"""EXPERIMENT 5 -- optimization as an imputer, and as an INPUT to the imputer.

Two things are tested against the same folds/seeds/metrics as exp2 and exp3:

  opt      the 3h gap is filled by SOLVING a constrained dispatch QP, no network:

               min_p  sum_t sum_i ( a_i p_it + 0.5 b_i p_it^2 )
               s.t.   sum_i s_i p_it = nd_t           balance, every step
                      0 <= p_it <= cap_i              box
                      -r_dn <= p_it - p_i,t-1 <= r_up ramps, with the two KNOWN
                                                      gap edges as boundary conditions

  opt_in   the BiLSTM imputer, given that solution as six extra input channels
           (optimizer inside the gap, observed dispatch outside it -- no leak,
           the outside values are already inputs)

  base     the same BiLSTM without them -- the control

The balance target nd_t is built from OBSERVED demand, which is visible inside the
gap (masked_batch zeroes only the 10 targets). nd deviation from the edge-interpolation
is regressed on demand deviation, fitted on train. So the optimizer gets genuine
within-gap information that interpolation does not have.

Cost parameters are re-anchored so marginal cost equals the median active price AT the
observed operating point: a_i = price_med_i - b_i*mu_i, b_i = 1/(dp/dlambda). Taking
design.md's a_i and b_i literally puts coal's marginal cost at 15,700 $/MWh.

    python colab/capacity_experiment/exp5_optimization_input.py
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
from pipeline import (CAL, GAP, GAP_EVAL_START, OBS, STEPS_PER_DAY, TARGETS,  # noqa: E402
                      load_table, make_days, metrics, top20_days)

HERE = Path(__file__).resolve().parent

DISP = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
        "battery_charging", "battery_discharging"]
SIGN = np.array([1., 1., 1., 1., -1., 1.])
SLOPE = np.array([0.224, 0.256, 0.042, 0.100, 0.150, 0.150])   # MW per $/MWh, design 5
PMED = np.array([91., 65., 171., 148., 11., 97.])              # median price when active

VAL_DAYS, TEST_DAYS, N_FOLDS = 45, 60, 4
HIDDEN, LAYERS, DROPOUT = 128, 2, 0.2
LR, WEIGHT_DECAY, BATCH = 1e-3, 1e-4, 64
EPOCHS, PATIENCE = 100, 12
SEEDS = [0, 1]
STRIDE = 12                          # gap-start grid; GAP_EVAL_START=132 is on it
QP_OUTER, QP_INNER = 30, 80

_mps = getattr(torch.backends, "mps", None)
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if (_mps is not None and _mps.is_available()) else "cpu")

cols = TARGETS + OBS + CAL
df = load_table()
X, days = make_days(df, cols)                       # (D, 288, C) MW
D = len(X)
ti = {c: i for i, c in enumerate(cols)}
di = [ti[t] for t in DISP]
tgt_all = [ti[t] for t in TARGETS]
dem_i = ti["demand"]

# both edges must exist: s-1 >= 0 and s+GAP <= 287
STARTS = np.arange(STRIDE, STEPS_PER_DAY - GAP, STRIDE)
assert GAP_EVAL_START in STARTS
o_max = D - VAL_DAYS - TEST_DAYS
origins = [int(round(x)) for x in np.linspace(o_max * 0.5, o_max, N_FOLDS)]
print(f"days={D} ({days[0].date()}..{days[-1].date()})  origins {origins}  device={DEVICE}")
print(f"gap-start grid: {len(STARTS)} positions, stride {STRIDE}", flush=True)

# ---------------------------------------------------------------- cost parameters
FIT = np.arange(0, origins[0])                       # subset of EVERY fold's train block
flat = X[FIT][:, :, di].reshape(-1, len(DISP))
cap = flat.max(0)
mu = flat.mean(0)
B_ = 1.0 / SLOPE
A_ = PMED - B_ * mu
A_[DISP.index("battery_charging")] *= -1             # load: charges while lambda < bid
d1 = np.diff(X[FIT][:, :, di], axis=1).reshape(-1, len(DISP))
R_UP = np.percentile(d1, 99.9, axis=0)
R_DN = -np.percentile(d1, 0.1, axis=0)
print("\ncost parameters (fitted on the first train block only)")
print(f"  {'channel':22s} {'mu':>8s} {'a':>10s} {'b':>7s} {'cap':>8s} {'r_up':>7s} {'r_dn':>7s}")
for k, t in enumerate(DISP):
    print(f"  {t:22s} {mu[k]:8.1f} {A_[k]:10.1f} {B_[k]:7.2f} {cap[k]:8.0f} "
          f"{R_UP[k]:7.1f} {R_DN[k]:7.1f}")

# gamma: how much of the within-gap demand deviation lands on net demand
ND = (X[:, :, di] * SIGN).sum(-1)                     # (D, 288)


def edge_interp(v, s):
    """Linear interpolation across a gap starting at s. v is (N, 288[, K])."""
    w = (np.arange(1, GAP + 1) / (GAP + 1))
    L, R = v[:, s - 1], v[:, s + GAP]
    if v.ndim == 3:
        return L[:, None, :] * (1 - w[None, :, None]) + R[:, None, :] * w[None, :, None]
    return L[:, None] * (1 - w[None, :]) + R[:, None] * w[None, :]


num = den = 0.0
for s in STARTS:
    nd_dev = (ND[FIT][:, s:s + GAP] - edge_interp(ND[FIT], s)).ravel()
    dm_dev = (X[FIT][:, s:s + GAP, dem_i] - edge_interp(X[FIT][:, :, dem_i], s)).ravel()
    num += float(dm_dev @ nd_dev)
    den += float(dm_dev @ dm_dev)
GAMMA = num / den
print(f"\ngamma (nd deviation per MW of demand deviation inside the gap): {GAMMA:.3f}")


# ---------------------------------------------------------------- the QP solver
def solve_qp(pL, pR, nd_t, tag=""):
    """Augmented-Lagrangian solve. pL/pR (N,6) known edges, nd_t (N,GAP) target."""
    t_ = lambda z: torch.tensor(np.asarray(z), dtype=torch.float64)
    a, b, capT = t_(A_), t_(B_), t_(cap)
    ru, rd, s = t_(R_UP), t_(R_DN), t_(SIGN)
    ndT, L, R = t_(nd_t), t_(pL), t_(pR)
    w = torch.tensor(np.arange(1, GAP + 1) / (GAP + 1), dtype=torch.float64)
    p = (L[:, None, :] * (1 - w[None, :, None]) + R[:, None, :] * w[None, :, None]).clone()
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
            dp = traj[:, 1:, :] - traj[:, :-1, :]
            vr = (torch.relu(dp - ru) ** 2 + torch.relu(-dp - rd) ** 2).sum(-1).sum(-1)
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
    dpn = np.diff(traj, axis=1)
    over = (np.maximum(dpn - R_UP, 0) + np.maximum(-dpn - R_DN, 0)).max()
    box = max(float(np.maximum(-P, 0).max()), float(np.maximum(P - cap, 0).max()))
    print(f"    QP{tag}: balance mean {bal.mean():7.3f} max {bal.max():7.2f} MW | "
          f"ramp over {over:6.3f} | box {box:6.3f}", flush=True)
    return P


print("\nsolving the QP on the gap-start grid ...", flush=True)
t0 = time.time()
OPT = np.zeros((len(STARTS), D, GAP, len(DISP)), np.float32)
for j, s in enumerate(STARTS):
    s_use = int(s)
    pL = X[:, s_use - 1][:, di]
    pR = X[:, s_use + GAP][:, di]
    nd_hat = edge_interp(ND, s_use) + GAMMA * (
        X[:, s_use:s_use + GAP, dem_i] - edge_interp(X[:, :, dem_i], s_use))
    OPT[j] = solve_qp(pL, pR, nd_hat, tag=f" s={s_use:3d}").astype(np.float32)
print(f"QP grid done in {time.time() - t0:.0f}s", flush=True)
START_POS = {int(s): j for j, s in enumerate(STARTS)}


# ---------------------------------------------------------------- the imputer
class BiLSTMImputer(nn.Module):
    def __init__(self, n_in, n_tgt):
        super().__init__()
        self.rnn = nn.LSTM(n_in + 1, HIDDEN, LAYERS, batch_first=True,
                           bidirectional=True, dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN * 2, n_tgt)

    def forward(self, x, mask):
        h, _ = self.rnn(torch.cat([x, mask], dim=-1))
        return self.head(h)


def scale(tr):
    flat = X[tr].reshape(-1, X.shape[2]).astype(np.float64)
    mean, std = flat.mean(0), flat.std(0)
    const = std <= 1e-9 * (np.abs(mean) + 1.0)
    Xs = X.astype(np.float32).copy()
    for cn in TARGETS + OBS:
        j = ti[cn]
        Xs[:, :, j] = (X[:, :, j] - mean[j]) / (1.0 if const[j] else std[j])
    return Xs, mean[tgt_all].astype(np.float32), std[tgt_all].astype(np.float32), mean, std


def batch(Xs, dayidx, starts, use_opt, omean, ostd):
    """Masked inputs; when use_opt, append 6 channels = optimizer inside the gap,
    observed dispatch outside it."""
    B = len(dayidx)
    xin = Xs[dayidx].copy()
    msk = np.ones((B, STEPS_PER_DAY, 1), np.float32)
    gidx = np.stack([np.arange(int(s), int(s) + GAP) for s in starts])
    for b in range(B):
        s = int(starts[b])
        xin[b, s:s + GAP, tgt_all] = 0.0
        msk[b, s:s + GAP, 0] = 0.0
    if not use_opt:
        return xin.astype(np.float32), msk, gidx
    ext = X[dayidx][:, :, di].astype(np.float32).copy()          # observed, MW
    for b in range(B):
        s = int(starts[b])
        ext[b, s:s + GAP] = OPT[START_POS[s], dayidx[b]]
    ext = (ext - omean) / ostd
    return np.concatenate([xin, ext], -1).astype(np.float32), msk, gidx


def run(tr, va, teS, seed, use_opt, tag):
    np.random.seed(seed)
    torch.manual_seed(seed)
    Xs, y_mean, y_std, mean_all, std_all = scale(tr)
    omean = mean_all[di].astype(np.float32)
    ostd = np.where(std_all[di] > 1e-9, std_all[di], 1.0).astype(np.float32)
    n_in = X.shape[2] + (len(DISP) if use_opt else 0)
    model = BiLSTMImputer(n_in, len(TARGETS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    vin, vmask, vg = batch(Xs, va, np.full(len(va), GAP_EVAL_START, int), use_opt,
                           omean, ostd)
    vin_t = torch.from_numpy(vin).to(DEVICE)
    vmask_t = torch.from_numpy(vmask).to(DEVICE)
    vy = torch.from_numpy(Xs[va][..., tgt_all]).to(DEVICE)
    vsel = torch.zeros(len(va), STEPS_PER_DAY, dtype=torch.bool)
    for b in range(len(va)):
        vsel[b, vg[b]] = True
    vsel = vsel.to(DEVICE)

    best, bstate, waited, ep = 1e9, None, 0, 0
    for ep in range(1, EPOCHS + 1):
        rng = np.random.default_rng(1000 + ep)
        order = rng.permutation(len(tr))
        model.train()
        for i in range(0, len(tr), BATCH):
            bi = tr[order[i:i + BATCH]]
            st = rng.choice(STARTS[1:], len(bi))
            xin, msk, gid = batch(Xs, bi, st, use_opt, omean, ostd)
            xb = torch.from_numpy(xin).to(DEVICE)
            mb = torch.from_numpy(msk).to(DEVICE)
            yb = torch.from_numpy(Xs[bi][..., tgt_all]).to(DEVICE)
            gs = torch.zeros(len(bi), STEPS_PER_DAY, dtype=torch.bool)
            for j in range(len(bi)):
                gs[j, gid[j]] = True
            gs = gs.to(DEVICE)
            opt.zero_grad()
            ((model(xb, mb) - yb)[gs] ** 2).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = ((model(vin_t, vmask_t) - vy)[vsel] ** 2).mean().item()
        if vl < best - 1e-6:
            best, waited = vl, 0
            bstate = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        if waited >= PATIENCE:
            break
    model.load_state_dict(bstate)

    xin, msk, gid = batch(Xs, teS, np.full(len(teS), GAP_EVAL_START, int), use_opt,
                          omean, ostd)
    with torch.no_grad():
        pr = model(torch.from_numpy(xin).to(DEVICE),
                   torch.from_numpy(msk).to(DEVICE)).cpu().numpy()
    P = np.stack([pr[b, gid[b]] for b in range(len(teS))]) * y_std + y_mean
    return P, best, ep


# ---------------------------------------------------------------- run the folds
g0, g1 = GAP_EVAL_START, GAP_EVAL_START + GAP
res = {}
stash = {}
t0 = time.time()
for fi, o in enumerate(origins):
    tr = np.arange(0, o)
    va = np.arange(o, o + VAL_DAYS)
    teS = np.arange(o + VAL_DAYS, o + VAL_DAYS + TEST_DAYS)
    key = f"fold{fi}"
    Y = X[teS][:, g0:g1][:, :, tgt_all]
    t20 = top20_days(X[teS], cols)
    res[key] = {"origin": int(o), "train_days": int(o),
                "test_from": str(days[teS[0]].date()), "test_to": str(days[teS[-1]].date())}
    print(f"\nfold{fi}: train {o}d -> test {res[key]['test_from']}..{res[key]['test_to']}",
          flush=True)

    # --- non-learned arms ---------------------------------------------------------
    I_all = edge_interp(X[teS][:, :, tgt_all], g0)
    P_opt = I_all.copy()
    P_opt[:, :, [TARGETS.index(t) for t in DISP]] = OPT[START_POS[g0], teS]
    for nm, P in (("interp", I_all), ("opt", P_opt)):
        per, mac = metrics(P, Y)
        _, m20 = metrics(P[t20], Y[t20])
        res[key][nm] = [{"_seed": None, "_macroWAPE": mac["WAPE"], "_macroR2": mac["R2"],
                         "_microWAPE": mac["microWAPE"], "_top20WAPE": m20["WAPE"],
                         "_top20R2": m20["R2"], "_top20micro": m20["microWAPE"], **per}]
        print(f"  [{nm:7s}] WAPE={mac['WAPE']:.4f} micro={mac['microWAPE']:.4f} "
              f"top20WAPE={m20['WAPE']:.4f} top20micro={m20['microWAPE']:.4f}", flush=True)

    # --- learned arms -------------------------------------------------------------
    for nm, use in (("base", False), ("opt_in", True)):
        res[key][nm] = []
        for sd in SEEDS:
            P, vl, ep = run(tr, va, teS, sd, use, f"{key}/{nm}/s{sd}")
            per, mac = metrics(P, Y)
            _, m20 = metrics(P[t20], Y[t20])
            row = {"_val": vl, "_seed": sd, "_epochs": ep,
                   "_macroWAPE": mac["WAPE"], "_macroR2": mac["R2"],
                   "_microWAPE": mac["microWAPE"], "_top20WAPE": m20["WAPE"],
                   "_top20R2": m20["R2"], "_top20micro": m20["microWAPE"], **per}
            res[key][nm].append(row)
            print(f"  [{nm:7s} s{sd}] ep{ep} val={vl:.4f} WAPE={mac['WAPE']:.4f} "
                  f"micro={mac['microWAPE']:.4f} top20WAPE={m20['WAPE']:.4f} "
                  f"top20micro={m20['microWAPE']:.4f}", flush=True)
            if fi == N_FOLDS - 1 and sd == SEEDS[0]:
                stash[nm] = P
    if fi == N_FOLDS - 1:
        stash.update({"actual": Y, "interp": I_all, "opt": P_opt,
                      "teS": teS, "t20": t20})

(HERE / "exp5_optimization_input_results.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(HERE / "exp5_stack_data.npz",
                    **{k: v for k, v in stash.items() if isinstance(v, np.ndarray)})
print(f"\ntotal {time.time() - t0:.0f}s -> exp5_optimization_input_results.json")

# ---------------------------------------------------------------- summary table
print("\n" + "=" * 100)
print("SUMMARY  (mean over folds; learned arms also mean over seeds)")
print("=" * 100)
print(f"  {'arm':10s} {'macroWAPE':>10s} {'microWAPE':>10s} {'macroR2':>9s} "
      f"{'top20WAPE':>10s} {'top20micro':>11s} {'top20R2':>9s}")
for nm in ("interp", "opt", "base", "opt_in"):
    g = lambda k: np.mean([np.mean([r[k] for r in res[f][nm]]) for f in res])
    print(f"  {nm:10s} {g('_macroWAPE'):10.4f} {g('_microWAPE'):10.4f} "
          f"{g('_macroR2'):9.4f} {g('_top20WAPE'):10.4f} {g('_top20micro'):11.4f} "
          f"{g('_top20R2'):9.4f}")
print("\nper fold, microWAPE:")
print(f"  {'arm':10s} " + "".join(f"{f:>10s}" for f in res))
for nm in ("interp", "opt", "base", "opt_in"):
    print(f"  {nm:10s} " +
          "".join(f"{np.mean([r['_microWAPE'] for r in res[f][nm]]):10.4f}" for f in res))
