"""EXPERIMENT 3 -- capacity-normalised targets.

Capacity as an INPUT CHANNEL fails (exp2: 18-79% worse across rolling origins) because
it reaches the network as a calendar-shaped era label. This tests the opposite use:
capacity never enters the network at all, it rescales the TARGET.

    train on   u(t) = p(t) / C(t)     instead of   p(t) in MW
    score on   u_hat(t) * C(t)        so numbers sit directly against exp2

C(t) is the monthly capacity table, known in advance, so this is not leakage. Only
wind, solar_utility, battery_charging and battery_discharging are normalised -- the
other six targets have a constant C over 2025-01..2026-08, where dividing by C is
absorbed by standardisation and is an exact no-op.

Same four rolling origins and seeds as exp2. The no_cap baseline is read from
exp2_rolling_origin_results.json rather than retrained (identical data, folds, seeds).

Data: data/updata only, via pipeline.py.

    python colab/capacity_experiment/exp3_capacity_normalised.py
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

# target -> capacity column used to normalise it. Only the fueltechs whose installed
# capacity actually moves on this window.
NORM = {"wind": "cap_wind",
        "solar_utility": "cap_solar_utility",
        "battery_charging": "cap_battery",
        "battery_discharging": "cap_battery"}

VAL_DAYS, TEST_DAYS, N_FOLDS = 45, 60, 4
HIDDEN, LAYERS, DROPOUT = 128, 2, 0.2
LR, WEIGHT_DECAY, BATCH = 1e-3, 1e-4, 64
EPOCHS, PATIENCE = 100, 12
SEEDS = [0, 1]

_mps = getattr(torch.backends, "mps", None)
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if (_mps is not None and _mps.is_available()) else "cpu")

cols = TARGETS + OBS + CAL
df = load_table()

XMW, days = make_days(df, cols)                    # ground truth, always MW
D = len(days)

# per-day capacity for each normalised target (capacity is constant within a day)
capcols = sorted(set(NORM.values()))
capday_all = df[capcols].groupby(df.index.normalize()).first().reindex(days)
ti = {t: i for i, t in enumerate(cols)}
norm_idx = [ti[t] for t in NORM]
capday = np.stack([capday_all[NORM[t]].to_numpy(np.float32) for t in NORM], 1)  # (D, 4)
assert np.isfinite(capday).all() and (capday > 0).all(), "capacity must be positive"

XN = XMW.copy()                                    # normalised copy for training
for k, t in enumerate(NORM):
    XN[:, :, ti[t]] = XMW[:, :, ti[t]] / capday[:, k][:, None]

o_max = D - VAL_DAYS - TEST_DAYS
origins = [int(round(x)) for x in np.linspace(o_max * 0.5, o_max, N_FOLDS)]
print(f"days={D} ({days[0].date()}..{days[-1].date()})  origins {origins}  device={DEVICE}")
print(f"normalised targets: {list(NORM)}", flush=True)


class BiLSTMImputer(nn.Module):
    def __init__(self, n_in, n_tgt):
        super().__init__()
        self.rnn = nn.LSTM(n_in + 1, HIDDEN, LAYERS, batch_first=True,
                           bidirectional=True, dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN * 2, n_tgt)

    def forward(self, x, mask):
        h, _ = self.rnn(torch.cat([x, mask], dim=-1))
        return self.head(h)


def scale(X, tr):
    tgt_idx = [ti[c] for c in TARGETS]
    flat = X[tr].reshape(-1, X.shape[2]).astype(np.float64)
    mean, std = flat.mean(0), flat.std(0)
    is_const = std <= 1e-9 * (np.abs(mean) + 1.0)
    Xs = X.astype(np.float32).copy()
    for cn in TARGETS + OBS:
        j = ti[cn]
        Xs[:, :, j] = (X[:, :, j] - mean[j]) / (1.0 if is_const[j] else std[j])
    return Xs, tgt_idx, mean[tgt_idx].astype(np.float32), std[tgt_idx].astype(np.float32)


def masked_batch(X, tgt_idx, starts):
    B = X.shape[0]
    xin = X.copy()
    msk = np.ones((B, STEPS_PER_DAY, 1), np.float32)
    gidx = np.stack([np.arange(int(s), int(s) + GAP) for s in starts])
    for b in range(B):
        s = int(starts[b])
        for t in tgt_idx:
            xin[b, s:s + GAP, t] = 0.0
        msk[b, s:s + GAP, 0] = 0.0
    return xin.astype(np.float32), msk, gidx


def run(tr, va, teS, seed, tag):
    np.random.seed(seed)
    torch.manual_seed(seed)
    Xs, tgt_idx, y_mean, y_std = scale(XN, tr)
    Xtr, Xva, Xte = Xs[tr], Xs[va], Xs[teS]
    tsel = np.asarray(tgt_idx)
    model = BiLSTMImputer(Xs.shape[2], len(TARGETS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    vin, vmask, vg = masked_batch(Xva, tgt_idx, np.full(len(Xva), GAP_EVAL_START, int))
    vin_t = torch.from_numpy(vin).to(DEVICE)
    vmask_t = torch.from_numpy(vmask).to(DEVICE)
    vy = torch.from_numpy(Xva[..., tsel]).to(DEVICE)
    vsel = torch.zeros(len(Xva), STEPS_PER_DAY, dtype=torch.bool)
    for b in range(len(Xva)):
        vsel[b, vg[b]] = True
    vsel = vsel.to(DEVICE)

    best, bstate, waited = 1e9, None, 0
    for ep in range(1, EPOCHS + 1):
        rng = np.random.default_rng(1000 + ep)
        order = rng.permutation(len(Xtr))
        model.train()
        for i in range(0, len(Xtr), BATCH):
            bi = order[i:i + BATCH]
            xb0 = Xtr[bi]
            xin, msk, gid = masked_batch(xb0, tgt_idx,
                                         rng.integers(0, STEPS_PER_DAY - GAP, len(bi)))
            xb = torch.from_numpy(xin).to(DEVICE)
            mb = torch.from_numpy(msk).to(DEVICE)
            yb = torch.from_numpy(xb0[..., tsel]).to(DEVICE)
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

    xin, msk, gid = masked_batch(Xte, tgt_idx, np.full(len(Xte), GAP_EVAL_START, int))
    with torch.no_grad():
        pr = model(torch.from_numpy(xin).to(DEVICE),
                   torch.from_numpy(msk).to(DEVICE)).cpu().numpy()
    # de-standardise -> normalised units, then multiply capacity back in -> MW
    P = np.stack([pr[b, gid[b]] for b in range(len(Xte))]) * y_std + y_mean   # (N, GAP, 10)
    capte = capday[teS]                                                       # (N, 4)
    for k, t in enumerate(NORM):
        P[:, :, TARGETS.index(t)] *= capte[:, k][:, None]
    # ground truth straight from the MW tensor -- never normalised
    Y = np.stack([XMW[teS][b, gid[b]][:, [ti[t] for t in TARGETS]]
                  for b in range(len(Xte))])
    per, mac = metrics(P, Y)
    t20 = top20_days(Xte, cols)
    _, mac20 = metrics(P[t20], Y[t20])
    out = {"_val": best, "_seed": seed,
           "_macroWAPE": mac["WAPE"], "_macroR2": mac["R2"], "_microWAPE": mac["microWAPE"],
           "_top20WAPE": mac20["WAPE"], "_top20R2": mac20["R2"],
           "_top20micro": mac20["microWAPE"]}
    out.update(per)
    print(f"  [{tag}] ep{ep} val={best:.4f} WAPE={out['_macroWAPE']:.4f} "
          f"R2={out['_macroR2']:.4f} top20WAPE={out['_top20WAPE']:.4f} "
          f"top20R2={out['_top20R2']:.4f}", flush=True)
    return out


res, t0 = {}, time.time()
for fi, o in enumerate(origins):
    tr = slice(0, o)
    va = slice(o, o + VAL_DAYS)
    teS = slice(o + VAL_DAYS, o + VAL_DAYS + TEST_DAYS)
    key = f"fold{fi}"
    res[key] = {"origin": o, "train_days": o,
                "test_from": str(days[o + VAL_DAYS].date()),
                "test_to": str(days[o + VAL_DAYS + TEST_DAYS - 1].date()), "norm": []}
    print(f"\nfold{fi}: train {o}d -> test {res[key]['test_from']}..{res[key]['test_to']}",
          flush=True)
    for seed in SEEDS:
        res[key]["norm"].append(run(tr, va, teS, seed, f"f{fi} norm s{seed}"))

print(f"\ntotal {time.time() - t0:.0f}s")
out = Path(__file__).with_name("exp3_capacity_normalised_results.json")
out.write_text(json.dumps(res, indent=1))

# ---- compare against exp2's no_cap and cap arms -------------------------------------
e2p = Path(__file__).with_name("exp2_rolling_origin_results.json")
e2p = Path(__file__).with_name("exp2_rolling_origin_results.json")
print("\n" + "=" * 104)
print("CAPACITY-NORMALISED TARGETS -- notebook metrics, mean over seeds")
print("=" * 104)
e2 = json.loads(e2p.read_text())
print(f"{'fold':>5}{'test window':>25}"
      f"{'WAPE nocap':>12}{'WAPE cap':>10}{'WAPE norm':>11}"
      f"{'R2 nocap':>10}{'R2 norm':>9}{'t20W nocap':>12}{'t20W norm':>11}")
d = {k: [] for k in ("_macroWAPE", "_macroR2", "_top20WAPE", "_top20R2", "_microWAPE")}
for k in res:
    A = {kk: np.mean([s[kk] for s in e2[k]["no_cap"]]) for kk in d}
    C = {kk: np.mean([s[kk] for s in e2[k]["cap"]]) for kk in d}
    N = {kk: np.mean([s[kk] for s in res[k]["norm"]]) for kk in d}
    for kk in d:
        d[kk].append(N[kk] - A[kk])
    win = f"{res[k]['test_from']}..{res[k]['test_to']}"
    print(f"{k[-1]:>5}{win:>25}"
          f"{A['_macroWAPE']:>12.4f}{C['_macroWAPE']:>10.4f}{N['_macroWAPE']:>11.4f}"
          f"{A['_macroR2']:>10.4f}{N['_macroR2']:>9.4f}"
          f"{A['_top20WAPE']:>12.4f}{N['_top20WAPE']:>11.4f}")
print("\ndelta (normalised - no_cap) per fold:")
for kk, lbl in [("_macroWAPE", "test_WAPE"), ("_macroR2", "test_R2"),
                ("_top20WAPE", "top20_WAPE"), ("_top20R2", "top20_R2"),
                ("_microWAPE", "micro_WAPE")]:
    v = d[kk]
    same = len({np.sign(x) for x in v}) == 1
    print(f"  {lbl:12s} {[f'{x:+.4f}' for x in v]}  mean {np.mean(v):+.4f}  "
          f"{'CONSISTENT' if same else 'SIGN FLIPS'}")
print(f"\nwrote {out.name}")
