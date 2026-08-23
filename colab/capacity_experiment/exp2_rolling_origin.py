"""EXPERIMENT 2 -- rolling-origin evaluation of the capacity ablation.

The notebook's conclusion rests on ONE chronological 471/58/60 cut. With a monotone
capacity trend in the data, a single split cannot separate "the fleet grew" from "this
test period's weather differs". Multiple origins can.

Expanding window: for each origin o,  train=days[0:o]  val=days[o:o+45]  test=next 60.

Arms: no_cap (16 ch) vs cap (+cap_battery, cap_solar_utility, cap_wind -> 19 ch).
Only those three capacity columns are non-constant on 2025-01..2026-08; the other six
are exactly constant, and a constant channel centred to zero is a no-op for the LSTM.

Data: data/updata only, via pipeline.py (same preprocessing as capacity.ipynb).

    python colab/capacity_experiment/exp2_rolling_origin.py
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

CAP_LIVE = ["cap_battery", "cap_solar_utility", "cap_wind"]
VAL_DAYS, TEST_DAYS, N_FOLDS = 45, 60, 4
HIDDEN, LAYERS, DROPOUT = 128, 2, 0.2
LR, WEIGHT_DECAY, BATCH = 1e-3, 1e-4, 64
EPOCHS, PATIENCE = 100, 12
SEEDS = [0, 1]

_mps = getattr(torch.backends, "mps", None)
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if (_mps is not None and _mps.is_available()) else "cpu")

df = load_table()
cols_A = TARGETS + OBS + CAL
cols_B = cols_A + CAP_LIVE
XA, days = make_days(df, cols_A)
XB, _ = make_days(df, cols_B)
D = len(days)
o_max = D - VAL_DAYS - TEST_DAYS
origins = [int(round(x)) for x in np.linspace(o_max * 0.5, o_max, N_FOLDS)]
print(f"days={D} ({days[0].date()}..{days[-1].date()})  origins {origins}  device={DEVICE}",
      flush=True)


class BiLSTMImputer(nn.Module):
    def __init__(self, n_in, n_tgt):
        super().__init__()
        self.rnn = nn.LSTM(n_in + 1, HIDDEN, LAYERS, batch_first=True,
                           bidirectional=True, dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN * 2, n_tgt)

    def forward(self, x, mask):
        h, _ = self.rnn(torch.cat([x, mask], dim=-1))
        return self.head(h)


def scale(X, cols, tr):
    ci = {c: i for i, c in enumerate(cols)}
    tgt_idx = [ci[c] for c in TARGETS]
    flat = X[tr].reshape(-1, X.shape[2]).astype(np.float64)
    mean, std = flat.mean(0), flat.std(0)
    is_const = std <= 1e-9 * (np.abs(mean) + 1.0)
    Xs = X.astype(np.float32).copy()
    for cn in TARGETS + OBS + [c for c in cols if c.startswith("cap_")]:
        j = ci[cn]
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


def run(X, cols, tr, va, teS, seed, tag):
    np.random.seed(seed)
    torch.manual_seed(seed)
    Xs, tgt_idx, y_mean, y_std = scale(X, cols, tr)
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
    P = np.stack([pr[b, gid[b]] for b in range(len(Xte))]) * y_std + y_mean   # (N,GAP,10)
    Y = np.stack([Xte[b, gid[b]][:, tgt_idx] for b in range(len(Xte))]) * y_std + y_mean

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


capday = df[CAP_LIVE].groupby(df.index.normalize()).first().reindex(days)
res, t0 = {}, time.time()
for fi, o in enumerate(origins):
    tr = slice(0, o)
    va = slice(o, o + VAL_DAYS)
    teS = slice(o + VAL_DAYS, o + VAL_DAYS + TEST_DAYS)
    nov = {c: bool(capday[c].values[teS].max() > capday[c].values[tr].max())
           for c in CAP_LIVE}
    key = f"fold{fi}"
    res[key] = {"origin": o, "train_days": o,
                "test_from": str(days[o + VAL_DAYS].date()),
                "test_to": str(days[o + VAL_DAYS + TEST_DAYS - 1].date()),
                "novelty": nov, "no_cap": [], "cap": []}
    print(f"\nfold{fi}: train {o}d -> test {res[key]['test_from']}..{res[key]['test_to']}  "
          f"novel caps: {[k for k, v in nov.items() if v] or 'none'}", flush=True)
    for seed in SEEDS:
        res[key]["no_cap"].append(run(XA, cols_A, tr, va, teS, seed, f"f{fi} no_cap s{seed}"))
        res[key]["cap"].append(run(XB, cols_B, tr, va, teS, seed, f"f{fi} cap    s{seed}"))

print(f"\ntotal {time.time() - t0:.0f}s")
out = Path(__file__).with_name("exp2_rolling_origin_results.json")
out.write_text(json.dumps(res, indent=1))

print("\n" + "=" * 100)
print("ROLLING ORIGIN -- notebook metrics (test_WAPE/test_R2/top20 macro), mean over seeds")
print("=" * 100)
hdr = (f"{'fold':>5}{'train':>7}{'test window':>25}"
       f"{'WAPE nocap':>12}{'WAPE cap':>10}{'R2 nocap':>10}{'R2 cap':>9}"
       f"{'t20W nocap':>12}{'t20W cap':>10}")
print(hdr)
agg = {k: [] for k in ("_macroWAPE", "_macroR2", "_top20WAPE", "_top20R2", "_microWAPE")}
for k, v in res.items():
    m = {arm: {kk: np.mean([s[kk] for s in v[arm]]) for kk in agg} for arm in ("no_cap", "cap")}
    for kk in agg:
        agg[kk].append(m["cap"][kk] - m["no_cap"][kk])
    win = f"{v['test_from']}..{v['test_to']}"
    print(f"{k[-1]:>5}{v['train_days']:>7}{win:>25}"
          f"{m['no_cap']['_macroWAPE']:>12.4f}{m['cap']['_macroWAPE']:>10.4f}"
          f"{m['no_cap']['_macroR2']:>10.4f}{m['cap']['_macroR2']:>9.4f}"
          f"{m['no_cap']['_top20WAPE']:>12.4f}{m['cap']['_top20WAPE']:>10.4f}")
print("\ndelta (cap - no_cap) per fold, and whether the sign is stable:")
for kk, lbl in [("_macroWAPE", "test_WAPE"), ("_macroR2", "test_R2"),
                ("_top20WAPE", "top20_WAPE"), ("_top20R2", "top20_R2"),
                ("_microWAPE", "micro_WAPE")]:
    d = agg[kk]
    same = len({np.sign(x) for x in d}) == 1
    worse = "worse" if (np.mean(d) > 0) == (kk != "_macroR2" and kk != "_top20R2") else "better"
    print(f"  {lbl:12s} {[f'{x:+.4f}' for x in d]}  mean {np.mean(d):+.4f}  "
          f"{'CONSISTENT ' + worse if same else 'SIGN FLIPS'}")
print(f"\nwrote {out.name}")
