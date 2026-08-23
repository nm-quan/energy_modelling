"""EXPERIMENT 8 -- the constrained whole-day imputer, and what each constraint costs.

The task: a whole day (288 five-minute steps) of the VIC dispatchable fuel mix is missing.
Demand, price, wind, solar, curtailment and interconnector flow are observed throughout;
the reservoir level is read at the two edges of the gap. Recover the five dispatch
channels so that the result is feasible on balance, capacity, the ramp table and SOC.

WHAT IS DIFFERENT FROM exp6, and why it matters
  nd COMES FROM OBSERVABLES. Every constrained model in this repo has been fed
  nd = SIGN . truth, which makes "balance exact by construction" a restatement of the
  answer; PUBLICATION_REVIEW.md measured the honest version at a 3,272 MW residual. Here
  nd is a fitted map over demand/wind/solar/net-import with a measured out-of-sample error
  (~55 MW against a 3,758 MW level), so the balance constraint is a real one. Both scorings
  are reported.

  THE HEAD IS WEIGHTED. exp6's plain projection made the constrained model WORSE than the
  unconstrained one (microWAPE 0.2441 -> 0.2638, gas_steam MAE 48.7 -> 155.5) because one
  scalar lambda splits the correction equally in MW. The allocation form is used here.

ONE BACKBONE, MANY HEADS. The ladder switches the head on a single trained network, so a
row-to-row difference is the constraint and cannot be a different random init. A separate
trained-through arm then measures what putting the head inside the training graph adds --
the repo's standing hypothesis is that projection at inference is nearly free while
training through a hard layer is not.

    python colab/capacity_experiment/exp8_constrained_imputer.py [--seeds 3] [--epochs 60]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constraint_set as CS  # noqa: E402
from head_traj import SocTorch, hardnet_traj, hardnet_traj_polished  # noqa: E402

HERE = Path(__file__).resolve().parent
N = CS.STEPS_PER_DAY
W = 3 * N
G0, G1 = N, 2 * N
C = len(CS.CHANNELS)
FEAT = CS.CHANNELS + CS.OBSERVED + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
HIDDEN, LAYERS, DROPOUT = 128, 2, 0.2
LR, WD, BATCH = 1e-3, 1e-4, 16
VAL_DAYS, TEST_DAYS = 45, 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- window build
def build_windows(cs):
    """Consecutive 3-day windows whose middle day is complete and SOC-bracketed."""
    df = cs["df"]
    import pandas as pd
    pos = pd.Series(np.arange(len(df)), index=df.index)
    Xall = df[FEAT].values.astype(np.float64)
    soc = df["soc_mwh"].values
    days, recs = pd.DatetimeIndex(sorted(set(df.index.normalize()))), []
    for d in days:
        a = d - pd.Timedelta(days=1)
        z = d + pd.Timedelta(days=2)
        if a not in pos.index or (z - pd.Timedelta("5min")) not in pos.index:
            continue
        i0 = int(pos[a])
        if int(pos[z - pd.Timedelta("5min")]) - i0 != W - 1:
            continue
        e0, eT = soc[i0 + G0 - 1], soc[i0 + G1]
        if not (np.isfinite(e0) and np.isfinite(eT)):
            continue
        recs.append(dict(day=d, i0=i0, e0=float(e0), eT=float(eT)))
    X = np.stack([Xall[r["i0"]:r["i0"] + W] for r in recs])
    return recs, X


def tensors(cs, recs, X):
    """Everything the head needs, aligned to the window batch."""
    ci = {c: FEAT.index(c) for c in CS.CHANNELS}
    tgt = [ci[c] for c in CS.CHANNELS]
    Y = X[:, G0:G1][:, :, tgt]                                     # (B,N,C) truth in MW
    pL = X[:, G0 - 1][:, tgt]
    pR = X[:, G1][:, tgt]
    df = cs["df"]
    pmin_all, pmax_all = CS.capacity_bounds(df, lift_to=df[CS.CHANNELS].values)
    Pmin = np.stack([pmin_all[r["i0"] + G0:r["i0"] + G1] for r in recs])
    Pmax = np.stack([pmax_all[r["i0"] + G0:r["i0"] + G1] for r in recs])
    nd_obs_all = CS.nd_observable(df, cs["nd_coef"])
    nd_obs = np.stack([nd_obs_all[r["i0"] + G0:r["i0"] + G1] for r in recs])
    nd_tru = (Y * CS.SIGN).sum(-1)
    e0 = np.array([r["e0"] for r in recs])
    eT = np.array([r["eT"] for r in recs])
    dem = X[:, G0:G1, FEAT.index("demand")]
    return dict(Y=Y, pL=pL, pR=pR, Pmin=Pmin, Pmax=Pmax, nd_obs=nd_obs, nd_tru=nd_tru,
                e0=e0, eT=eT, demand=dem)


# ------------------------------------------------------------------------------- backbone
class BiLSTMImputer(nn.Module):
    """Predicts a RESIDUAL over the linear-interpolation skeleton.

    The skeleton already carries the two pinned endpoints, so the network only has to
    supply the shape in between -- the same parameterisation imputation/train.py uses, and
    the reason a randomly initialised model starts at interpolation rather than at zero.
    Two extra outputs per channel are the allocation logits the weighted head consumes.
    """

    def __init__(self, n_in: int, n_ch: int):
        super().__init__()
        self.rnn = nn.LSTM(n_in + 1, HIDDEN, LAYERS, batch_first=True,
                           bidirectional=True, dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN * 2, 2 * n_ch)
        self.n_ch = n_ch

    def forward(self, x, mask):
        h, _ = self.rnn(torch.cat([x, mask], dim=-1))
        return self.head(h)


def interp_skeleton(pL, pR, n=N):
    w = np.arange(1, n + 1) / (n + 1)
    return pL[:, None, :] * (1 - w[None, :, None]) + pR[:, None, :] * w[None, :, None]


# ------------------------------------------------------------------------------- training
def train_backbone(X, T, tr, va, seed, epochs, patience, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    flat = X[tr].reshape(-1, X.shape[2])
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.where(sd <= 1e-9 * (np.abs(mu) + 1.0), 1.0, sd)
    Xs = ((X - mu) / sd).astype(np.float32)
    tgt = [FEAT.index(c) for c in CS.CHANNELS]
    skel = interp_skeleton(T["pL"], T["pR"])
    ys = ((T["Y"] - skel) / sd[tgt]).astype(np.float32)             # residual, scaled

    def batch(idx):
        xin = Xs[idx].copy()
        xin[:, G0:G1][:, :, tgt] = 0.0
        m = np.ones((len(idx), W, 1), np.float32)
        m[:, G0:G1, 0] = 0.0
        return torch.from_numpy(xin).to(DEVICE), torch.from_numpy(m).to(DEVICE)

    model = BiLSTMImputer(X.shape[2], C).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    vx, vm = batch(va)
    vy = torch.from_numpy(ys[va]).to(DEVICE)
    best, bstate, waited = 1e18, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(1000 + ep).permutation(len(tr))
        for i in range(0, len(tr), BATCH):
            bi = tr[order[i:i + BATCH]]
            xb, mb = batch(bi)
            yb = torch.from_numpy(ys[bi]).to(DEVICE)
            opt.zero_grad()
            ((model(xb, mb)[:, G0:G1, :C] - yb) ** 2).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = ((model(vx, vm)[:, G0:G1, :C] - vy) ** 2).mean().item()
        if vl < best - 1e-7:
            best, waited = vl, 0
            bstate = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"    ep{ep:3d} val {vl:.5f} (best {best:.5f})", flush=True)
    model.load_state_dict(bstate)
    return model, (mu, sd), best


def predict_raw(model, X, T, idx, scal):
    """(B,N,2C): channel levels in MW plus the allocation logits."""
    mu, sd = scal
    tgt = [FEAT.index(c) for c in CS.CHANNELS]
    Xs = ((X - mu) / sd).astype(np.float32)
    xin = Xs[idx].copy()
    xin[:, G0:G1][:, :, tgt] = 0.0
    m = np.ones((len(idx), W, 1), np.float32)
    m[:, G0:G1, 0] = 0.0
    model.eval()
    with torch.no_grad():
        o = model(torch.from_numpy(xin).to(DEVICE),
                  torch.from_numpy(m).to(DEVICE))[:, G0:G1].cpu().numpy()
    lev = o[..., :C] * sd[tgt] + interp_skeleton(T["pL"][idx], T["pR"][idx])
    return np.concatenate([lev, o[..., C:]], -1)


# ---------------------------------------------------------------------------------- arms
ARMS = [
    ("none",         dict(head=False)),
    ("+balance",     dict(ramp_k=0, tv_cap=False, soc=False, box=False)),
    ("+box",         dict(ramp_k=0, tv_cap=False, soc=False)),
    ("+box C(t)",    dict(ramp_k=0, tv_cap=True, soc=False)),
    ("+ramp k=1",    dict(ramp_k=1, tv_cap=True, soc=False)),
    ("+ramp R(k)",   dict(ramp_k=None, tv_cap=True, soc=False)),
    ("+SOC",         dict(ramp_k=None, tv_cap=True, soc=True)),
]


def t64(a):
    return torch.tensor(np.asarray(a), dtype=torch.float64)


def apply_head(raw, T, idx, cs, res, ramp_k=None, tv_cap=True, soc=True, box=True,
               nd_key="nd_obs", polish=6, alloc=True):
    Pmin, Pmax = T["Pmin"][idx].copy(), T["Pmax"][idx].copy()
    if not box:                                   # balance only: an effectively free box
        Pmin = np.full_like(Pmin, -1e6)
        Pmax = np.full_like(Pmax, 1e6)
    elif not tv_cap:                              # the old convention: one static cap
        Pmin = np.broadcast_to(Pmin.min((0, 1)), Pmin.shape).copy()
        Pmax = np.broadcast_to(Pmax.max((0, 1)), Pmax.shape).copy()
    st = SocTorch(res.eta, res.draw, res.e_max, e_min=res.e_min) if soc else None
    fn = hardnet_traj_polished if soc else hardnet_traj
    kw = dict(polish=polish) if soc else {}
    P = fn(t64(raw), t64(T["pL"][idx]), t64(T["pR"][idx]), t64(T[nd_key][idx]),
           t64(Pmin), t64(Pmax), t64(cs["R_up"]), t64(cs["R_dn"]), t64(CS.SIGN),
           batt_idx=CS.BATT, soc=st, e0=t64(T["e0"][idx]) if soc else None,
           alloc=alloc, ramp_k=ramp_k, **kw)
    return P.numpy()


def score(P, T, idx, cs, res):
    """Accuracy in the 6 reported channels + the four violation magnitudes."""
    Y6 = CS.to_report(T["Y"][idx])
    P6 = CS.to_report(P)
    e = P6 - Y6
    den = np.abs(Y6).sum((0, 1))
    per = {c: dict(MAE=float(np.abs(e[..., i]).mean()),
                   WAPE=float(np.abs(e[..., i]).sum() / max(den[i], 1e-9)))
           for i, c in enumerate(CS.REPORT)}
    ss = ((Y6 - Y6.mean((0, 1))) ** 2).sum((0, 1))
    r2 = 1.0 - (e ** 2).sum((0, 1)) / np.maximum(ss, 1e-9)
    out = dict(MAE=float(np.abs(e).mean()),
               macroWAPE=float(np.mean([per[c]["WAPE"] for c in CS.REPORT])),
               microWAPE=float(np.abs(e).sum() / np.abs(Y6).sum()),
               macroR2=float(np.mean(r2)), per=per)
    ru, rd = cs["R_up"], cs["R_dn"]
    full = np.concatenate([T["pL"][idx][:, None], P, T["pR"][idx][:, None]], 1)
    worst = 0.0
    for k in range(1, P.shape[1] + 2):
        d = full[:, k:] - full[:, :-k]
        worst = max(worst, float((np.maximum(d - ru[k], 0) + np.maximum(-d - rd[k], 0)).max()))
    E = res.trajectory(P[..., CS.BATT], T["e0"][idx], T["pL"][idx][:, CS.BATT])
    out["v"] = dict(
        balance_obs=float(np.abs((P * CS.SIGN).sum(-1) - T["nd_obs"][idx]).max()),
        balance_true=float(np.abs((P * CS.SIGN).sum(-1) - T["nd_tru"][idx]).max()),
        box=float(max(np.maximum(T["Pmin"][idx] - P, 0).max(),
                      np.maximum(P - T["Pmax"][idx], 0).max())),
        ramp=worst,
        soc=float(max(np.maximum(E - res.e_max, 0).max(),
                      np.maximum(res.e_min - E, 0).max())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--polish", type=int, default=6)
    a = ap.parse_args()

    cs = CS.build()
    res = cs["res"]
    recs, X = build_windows(cs)
    T = tensors(cs, recs, X)
    B = len(recs)
    o = B - VAL_DAYS - TEST_DAYS
    tr, va, te = np.arange(0, o), np.arange(o, o + VAL_DAYS), np.arange(o + VAL_DAYS, B)
    print(f"3-day windows with a complete SOC-bracketed middle day: {B}")
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}  "
          f"(test {recs[te[0]]['day'].date()} .. {recs[te[-1]]['day'].date()})")
    print(f"  device {DEVICE}   channels {CS.CHANNELS}")
    r = T["nd_obs"][te] - T["nd_tru"][te]
    print(f"  observable nd on the test gaps: MAE {np.abs(r).mean():.1f}  "
          f"max {np.abs(r).max():.1f} MW  (level {T['nd_tru'][te].mean():.0f} MW)\n")

    results = {"n_windows": B, "test_days": [str(recs[i]["day"].date()) for i in te]}

    # ---- reference rows that need no network --------------------------------------
    ref = {}
    skel = interp_skeleton(T["pL"], T["pR"])
    ref["interp"] = skel[te]
    prev = np.stack([X[i, :G0][:, [FEAT.index(c) for c in CS.CHANNELS]] for i in te])
    ref["persist"] = prev
    raw_interp = np.concatenate([skel[te], np.zeros_like(skel[te])], -1)
    ref["interp+head"] = apply_head(raw_interp, T, te, cs, res, polish=a.polish)

    print("=" * 118)
    print("REFERENCE ROWS (no trained parameters)")
    print("=" * 118)
    hdr = (f"  {'arm':16s}{'macroWAPE':>10s}{'microWAPE':>10s}{'MAE':>8s}{'macroR2':>9s}"
           f"{'bal|obs':>10s}{'box':>9s}{'ramp':>9s}{'SOC':>10s}")
    print(hdr)
    for k, P in ref.items():
        s = score(P, T, te, cs, res)
        results[k] = s
        print(f"  {k:16s}{s['macroWAPE']:10.4f}{s['microWAPE']:10.4f}{s['MAE']:8.2f}"
              f"{s['macroR2']:9.4f}{s['v']['balance_obs']:10.1e}{s['v']['box']:9.1e}"
              f"{s['v']['ramp']:9.1e}{s['v']['soc']:10.1e}")

    # ---- the ladder, one backbone per seed ----------------------------------------
    print("\n" + "=" * 118)
    print("CONSTRAINT LADDER -- one trained backbone, the head switched arm by arm")
    print(f"({a.seeds} seed{'s' if a.seeds > 1 else ''}, mean +- half-range)")
    print("=" * 118)
    per_seed = {name: [] for name, _ in ARMS}
    for sd in range(a.seeds):
        t0 = time.time()
        model, scal, vbest = train_backbone(X, T, tr, va, sd, a.epochs, a.patience)
        raw = predict_raw(model, X, T, te, scal)
        print(f"  seed {sd}: val {vbest:.5f}  ({time.time() - t0:.0f}s)", flush=True)
        for name, kw in ARMS:
            if kw.get("head") is False:
                P = raw[..., :C]
            else:
                P = apply_head(raw, T, te, cs, res, polish=a.polish,
                               **{k: v for k, v in kw.items() if k != "head"})
            per_seed[name].append(score(P, T, te, cs, res))
        print(f"    ladder done ({time.time() - t0:.0f}s)", flush=True)

    print("\n" + hdr)
    for name, _ in ARMS:
        ss = per_seed[name]
        g = lambda f: (np.mean([f(s) for s in ss]),
                       0.5 * (max(f(s) for s in ss) - min(f(s) for s in ss)))
        mw, mwr = g(lambda s: s["macroWAPE"])
        uw, uwr = g(lambda s: s["microWAPE"])
        ma, mar = g(lambda s: s["MAE"])
        r2, _ = g(lambda s: s["macroR2"])
        vb = max(s["v"]["balance_obs"] for s in ss)
        vx = max(s["v"]["box"] for s in ss)
        vr = max(s["v"]["ramp"] for s in ss)
        vs = max(s["v"]["soc"] for s in ss)
        print(f"  {name:16s}{mw:10.4f}{uw:10.4f}{ma:8.2f}{r2:9.4f}"
              f"{vb:10.1e}{vx:9.1e}{vr:9.1e}{vs:10.1e}")
        results[name] = dict(macroWAPE=mw, macroWAPE_hr=mwr, microWAPE=uw,
                             microWAPE_hr=uwr, MAE=ma, MAE_hr=mar, macroR2=r2,
                             v=dict(balance_obs=vb, box=vx, ramp=vr, soc=vs),
                             per={c: float(np.mean([s["per"][c]["MAE"] for s in ss]))
                                  for c in CS.REPORT})
    print("\n  violation columns are MAGNITUDES (MW, and MWh for SOC), worst over seeds")
    print("  and over every window -- not counts at a tolerance.")

    print("\n" + "=" * 118)
    print("PER-CHANNEL MAE (MW), mean over seeds")
    print("=" * 118)
    print(f"  {'arm':16s}" + "".join(f"{c[:11]:>13s}" for c in CS.REPORT))
    for k in list(ref) + [n for n, _ in ARMS]:
        s = results[k]
        pc = s["per"] if isinstance(next(iter(s["per"].values())), float) else \
            {c: s["per"][c]["MAE"] for c in CS.REPORT}
        print(f"  {k:16s}" + "".join(f"{pc[c]:13.1f}" for c in CS.REPORT))

    (HERE / "exp8_constrained_results.json").write_text(json.dumps(results, indent=1))
    print("\nwrote exp8_constrained_results.json")


if __name__ == "__main__":
    main()
