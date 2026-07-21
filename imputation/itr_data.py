"""Window sampler for the T=288 (whole-day) transformer-imputer ablation.

Stage-1 arms (training-method ablation, constraint = none):

  arm  gap sampling                               flank aug   aux recon loss
  T1   mixed: |G| in {3h,6h,12h} (+20% blackout)   off         on (weight 0.1)
  T2   mixed                                       on          on
  T3   blackout only (no flanks)                   n/a         on
  T4   mixed                                       on          off

"blackout" = the ENTIRE 288-step window has all 6 dispatch channels hidden --
the model sees only the drivers (net_demand, demand, price, calendar). "mixed"
places one contiguous gap of 36/72/144 steps inside the window (equal thirds of
the remaining 80%) with observed flanks either side. All gaps mask the 6
dispatch channels JOINTLY (one mask channel).

Flank aug (T2/T4): with p=0.5 per window, the OBSERVED flank dispatch (input
values, their aux-recon targets, and the pL/pR boundary rows) is scaled by
(1+delta), delta ~ U(-0.1, 0.1) -- one delta per window. Drivers are NOT
scaled, so the model learns not to over-trust flank levels vs the balance
identity (the counterfactual feeds a world where flanks are themselves inferred).

Windows are stored as (start, gap_start, gap_len) metadata only; batches are
materialized on the fly from the flats (a 40k x 288 x 17 tensor would be
~0.8 GB for nothing -- the flats are contiguous).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap_data import (load_flats, _hour_frac, _split_flats, TARGETS, SIGN,   # noqa: E402,F401
                      TARGET_FEAT_IDX, ND_COL, DEM_COL, PRICE_COL)

T = 288                          # one day of 5-min steps
GAP_CHOICES = (36, 72, 144)      # 3h, 6h, 12h
BLACKOUT = T
BLACKOUT_P = 0.20                # blackout share of "mixed"
FREE_W = (11, 14)                # free window for the counterfactual (repo convention)

ARMS = {"T1": dict(gaps="mixed",    flank_aug=False, aux=True),
        "T2": dict(gaps="mixed",    flank_aug=True,  aux=True),
        "T3": dict(gaps="blackout", flank_aug=False, aux=True),
        "T4": dict(gaps="mixed",    flank_aug=True,  aux=False)}

_TFI = np.asarray(TARGET_FEAT_IDX)


def sample_windows(f, split, n, mode="mixed", seed=0, glen_fixed=None):
    """Sample n window records {s, g0, glen}. s = flat row of the window start;
    g0 = gap start RELATIVE to the window; glen = gap length. Requires contiguous
    5-min spacing over [s-1, s+T] (both boundary rows exist) and, for val/test,
    the gap to sit past the lb_carry train-tail prefix. glen_fixed pins the gap
    length (the per-length test tables)."""
    X, _ = _split_flats(f, split)
    lb = f.lb_carry if split in ("val", "test") else 0
    hf = _hour_frac(X, f)
    ok = np.isclose(np.diff(hf) % 24.0, 1.0 / 12.0, atol=1e-3)
    run = np.concatenate([[0], np.cumsum(ok)])
    rng = np.random.default_rng(seed)
    s_l, g_l, n_l = [], [], []
    seen: set[tuple] = set()
    tries, cap = 0, max(n * 60, 8000)
    while len(s_l) < n and tries < cap:
        tries += 1
        s = int(rng.integers(1, len(X) - T))
        if run[s + T] - run[s - 1] != T + 1:               # window + both boundary rows
            continue
        if glen_fixed is not None:
            glen = int(glen_fixed)
        elif mode == "blackout":
            glen = BLACKOUT
        else:
            glen = BLACKOUT if rng.random() < BLACKOUT_P else int(rng.choice(GAP_CHOICES))
        # flanked gaps keep both boundaries INSIDE the window (pL/pR observed by the model)
        g0 = 0 if glen == BLACKOUT else int(rng.integers(1, T - glen))
        if s + g0 < lb:                                    # gap in genuine split rows
            continue
        key = (s, g0, glen)
        if key in seen:
            continue
        seen.add(key)
        s_l.append(s); g_l.append(g0); n_l.append(glen)
    if len(s_l) < min(n, 16):
        raise RuntimeError(f"only {len(s_l)} windows for split={split!r} mode={mode!r} "
                           f"glen={glen_fixed} -- is prepared.npz the committed one?")
    return {"s": np.array(s_l), "g0": np.array(g_l), "glen": np.array(n_l)}


def glen_groups(rec, batch, rng=None):
    """Yield (glen, idx) batches grouped by gap length: every batch has ONE gap
    length so in-graph constraint maps get a uniform (B,N,6) trajectory."""
    order = np.arange(len(rec["s"]))
    if rng is not None:
        order = rng.permutation(order)
    for glen in np.unique(rec["glen"]):
        grp = order[rec["glen"][order] == glen]
        for i in range(0, len(grp), batch):
            yield int(glen), grp[i:i + batch]


def build_batch(f, split, rec, idx, aug=False, rng=None):
    """Materialize one uniform-glen batch. Returns dict:
    X (B,T,17) z-scored, dispatch zeroed in gap | M (B,T,1) observed-mask |
    Y (B,T,6) y-scaled truth (flank-aug applied to flanks) | g0 (B,) | glen |
    pL/pR (B,6) MW boundaries | nd (B,glen) MW net_demand inside the gap."""
    Xf, Yf = _split_flats(f, split)
    s = rec["s"][idx]; g0 = rec["g0"][idx]; glen = int(rec["glen"][idx][0])
    assert (rec["glen"][idx] == glen).all(), "build_batch needs a uniform-glen batch"
    B = len(s)
    rows = s[:, None] + np.arange(T)[None, :]
    X = Xf[rows].astype(np.float32).copy()
    Y = Yf[rows].astype(np.float32).copy()
    gap = g0[:, None] + np.arange(glen)[None, :]           # (B,glen) window-relative
    M = np.ones((B, T, 1), np.float32)
    bi = np.arange(B)[:, None]
    M[bi, gap, 0] = 0.0
    pL = f.y_to_mw(Yf[s + g0 - 1]).astype(np.float64)      # blackout: g0=0 -> row s-1
    pR = f.y_to_mw(Yf[s + g0 + glen]).astype(np.float64)   # blackout: row s+T
    nd = f.col_mw(Xf, ND_COL)[s[:, None] + gap].astype(np.float64)

    if aug and rng is not None and glen < T:               # blackout has no flanks -> no-op
        delta = np.where(rng.random(B) < 0.5, rng.uniform(-0.1, 0.1, B), 0.0).astype(np.float32)
        obs = M[:, :, 0] == 1.0
        y_mw = Y * f.y_scale + f.y_mean
        y_mw = np.where(obs[:, :, None], y_mw * (1 + delta[:, None, None]), y_mw)
        Y = ((y_mw - f.y_mean) / f.y_scale).astype(np.float32)
        x_mw = X[:, :, _TFI] * f.x_scale[_TFI] + f.x_mean[_TFI]
        x_mw = np.where(obs[:, :, None], x_mw * (1 + delta[:, None, None]), x_mw)
        X[:, :, _TFI] = (x_mw - f.x_mean[_TFI]) / f.x_scale[_TFI]
        pL = pL * (1 + delta[:, None]); pR = pR * (1 + delta[:, None])

    for c in _TFI:                                         # hide the gap dispatch
        X[bi, gap, c] = 0.0
    return {"X": X, "M": M, "Y": Y, "g0": g0, "glen": glen, "pL": pL, "pR": pR, "nd": nd}


def day_windows(f):
    """One record per full calendar test day (00:00 start, 288 contiguous rows,
    both boundary rows) -- the deployment/blackout unit for D2/D3."""
    idx = f.test_index
    lb = f.Xte.shape[0] - len(idx)
    hf = _hour_frac(f.Xte, f)
    ok = np.isclose(np.diff(hf) % 24.0, 1.0 / 12.0, atol=1e-3)
    run = np.concatenate([[0], np.cumsum(ok)])
    day = idx.normalize()
    starts, days = [], []
    for d in pd.unique(day):
        pos = np.where(np.asarray(day == d))[0]
        if len(pos) != T or idx[pos[0]].hour != 0:
            continue
        s = lb + pos[0]
        if s < 1 or s + T >= f.Xte.shape[0]:
            continue
        if run[s + T] - run[s - 1] != T + 1:
            continue
        starts.append(s); days.append(pd.Timestamp(d))
    return {"s": np.array(starts), "g0": np.zeros(len(starts), int),
            "glen": np.full(len(starts), T)}, days


if __name__ == "__main__":
    f = load_flats()
    for mode in ("mixed", "blackout"):
        rec = sample_windows(f, "train", 64, mode=mode, seed=0)
        u, c = np.unique(rec["glen"], return_counts=True)
        print(f"{mode:9s}: 64 train windows, glen counts {dict(zip(u.tolist(), c.tolist()))}")
    rec = sample_windows(f, "val", 32, mode="blackout", seed=1)
    b = build_batch(f, "val", rec, np.arange(8))
    print(f"blackout batch: X{b['X'].shape} M{b['M'].shape} Y{b['Y'].shape} "
          f"pL{b['pL'].shape} nd{b['nd'].shape}  masked-steps/window={int((b['M']==0).sum()/8)}")
    rec = sample_windows(f, "test", 16, glen_fixed=72, seed=2)
    b = build_batch(f, "test", rec, np.arange(16), aug=True, rng=np.random.default_rng(0))
    gap_zero = all((b["X"][k][b["M"][k, :, 0] == 0][:, _TFI] == 0).all() for k in range(16))
    print(f"72-step aug batch: gap dispatch zeroed={gap_zero}  glen={b['glen']}")
    dw, days = day_windows(f)
    print(f"full test days for D2/D3: {len(days)}  ({days[0].date()} .. {days[-1].date()})")
