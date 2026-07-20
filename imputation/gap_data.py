"""Subspace-imputation dataset for the demand-scenario window (professor's reframe).

The simulation task recast as GAP IMPUTATION: inside a window (default the daily
11:00-14:00 free window) the 6-source dispatch BREAKDOWN is unknown; everything
else is known at every step -- net_demand (= the exact signed sum of the six, by
the balance identity), demand, price, calendar -- plus the dispatch on BOTH sides
of the gap (bidirectional boundaries). See constraints/lit_review.md theme 9.

Everything is built from data/preprocessed/hist/5min/.../prepared.npz (the ~4.8-yr
hist flats) -- no raw parquets, so it runs on a Colab clone.

Feature layout (prepared.npz feat_cols):
    [0-5]  hydro, gas_steam, gas_ocgt, coal_brown, batt_chg, batt_dis   <- UNKNOWN in gap
    [6]    net_demand   [7] demand_mw   [8] price                       <- KNOWN everywhere
    [9-16] hour_sin/cos, dow_sin/cos, season_sin/cos, is_weekend, is_peak

Truth for the 6 sources is the y-scaled target flat (TARGETS order), so we work in
TARGETS order = [hydro, coal_brown, gas_steam, gas_ocgt, batt_chg, batt_dis] with
SIGN = [1,1,1,1,-1,1] (charging is a load) -- consistent with the rest of the repo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_totdem" / "prepared.npz"

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]
SIGN = np.array([1, 1, 1, 1, -1, 1], dtype=np.float64)
# map each TARGETS entry to its column in the 17-feature vector (feat order above)
TARGET_FEAT_IDX = [0, 3, 1, 2, 4, 5]
ND_COL, DEM_COL, PRICE_COL = 6, 7, 8
HOUR_SIN_COL, HOUR_COS_COL = 9, 10
GAP_HOURS = (11, 14)
STEPS_PER_HOUR = 12


@dataclass
class Flats:
    Xtr: np.ndarray; Ytr: np.ndarray
    Xva: np.ndarray; Yva: np.ndarray
    Xte: np.ndarray; Yte: np.ndarray
    x_mean: np.ndarray; x_scale: np.ndarray
    y_mean: np.ndarray; y_scale: np.ndarray
    test_index: pd.DatetimeIndex
    feat_cols: list[str]
    lb_carry: int = 288        # context rows prepended to val/test flats (train tail)

    def y_to_mw(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.y_scale + self.y_mean

    def col_mw(self, X_scaled: np.ndarray, col: int) -> np.ndarray:
        return X_scaled[..., col] * self.x_scale[col] + self.x_mean[col]


def load_flats() -> Flats:
    z = np.load(NPZ, allow_pickle=False)
    return Flats(
        Xtr=z["Xtr_flat"], Ytr=z["Ytr_flat"],
        Xva=z["Xva_flat"], Yva=z["Yva_flat"],
        Xte=z["Xte_flat"], Yte=z["Yte_flat"],
        x_mean=z["x_mean"], x_scale=z["x_scale"],
        y_mean=z["y_mean"], y_scale=z["y_scale"],
        test_index=pd.DatetimeIndex(z["test_index"]),
        feat_cols=[str(c) for c in z["feat_cols"]],
        lb_carry=int(z["lookback_steps"]) if "lookback_steps" in z.files else 288,
    )


# --------------------------- test gaps (deployment-matched) ---------------------------

@dataclass
class GapWindow:
    """One imputation window: indices into the TEST flat (which carries lb context
    rows before the test split, so context_L is available for the earliest days)."""
    day: pd.Timestamp
    gap_idx: np.ndarray        # (G,) flat rows of the gap (11:00..13:55)
    ctxL_idx: np.ndarray       # (K,) rows before the gap
    ctxR_idx: np.ndarray       # (K,) rows after the gap
    pL_mw: np.ndarray          # (6,) dispatch at the step just before the gap (10:55)
    pR_mw: np.ndarray          # (6,) dispatch at the step just after the gap (14:00)
    truth_mw: np.ndarray       # (G,6) true dispatch in the gap (TARGETS order)
    nd_mw: np.ndarray          # (G,) net_demand in the gap
    hour: float = -1.0         # clock hour the gap OPENS at (for per-time-of-day eval)


def test_gap_windows(f: Flats, context: int = 72,
                     gap_hours=GAP_HOURS) -> list[GapWindow]:
    """Build one GapWindow per full 11:00-14:00 test day with `context` steps of
    valid, gap-free, contiguous data on each side. The test flat has 288 lb-context
    rows prepended, so test_index aligns to the LAST len(test_index) rows."""
    idx = f.test_index
    lb = f.Xte.shape[0] - len(idx)                     # prepended context rows
    G = (gap_hours[1] - gap_hours[0]) * STEPS_PER_HOUR
    step = np.timedelta64(5, "m")

    # map each test stamp to its flat row
    hour = idx.hour.to_numpy()
    day = idx.normalize()
    out: list[GapWindow] = []
    for d in pd.unique(day):
        in_day = np.where(np.asarray(day == d))[0]      # positions within test_index
        gap_pos = in_day[(hour[in_day] >= gap_hours[0]) & (hour[in_day] < gap_hours[1])]
        if len(gap_pos) != G:
            continue                                    # partial day / DST / gaps -> skip
        g0, g1 = gap_pos[0], gap_pos[-1]                # first/last gap position in test_index
        if g0 - 1 < 0 or g1 + 1 >= len(idx):
            continue
        # require contiguous 5-min spacing across [g0-context-1 .. g1+context]
        lo, hi = g0 - context - 1, g1 + context + 1
        if lo < 0 or hi >= len(idx):
            continue
        span = idx[lo:hi + 1].values
        if not (np.diff(span) == step).all():
            continue
        gap_flat = lb + gap_pos
        out.append(GapWindow(
            day=pd.Timestamp(d),
            gap_idx=gap_flat,
            ctxL_idx=lb + np.arange(g0 - context, g0),
            ctxR_idx=lb + np.arange(g1 + 1, g1 + 1 + context),
            pL_mw=f.y_to_mw(f.Yte[lb + g0 - 1]),
            pR_mw=f.y_to_mw(f.Yte[lb + g1 + 1]),
            truth_mw=f.y_to_mw(f.Yte[gap_flat]),
            nd_mw=f.col_mw(f.Xte, ND_COL)[gap_flat],
        ))
    return out


# --------------------------- train sampling (random-position masks) ---------------------------

def _build_windows(f: Flats, Xflat: np.ndarray, Yflat: np.ndarray, starts: np.ndarray,
                   context: int, gap: int) -> dict:
    """Assemble model-ready window tensors from flat-row start positions:
      X    (n, W, 17)  features, with the 6 source cols ZEROED inside the gap
      mask (n, W, 1)   1 outside the gap, 0 inside (which steps are blanks)
      Y    (n, G, 6)   true sources (y-scaled, TARGETS order) in the gap
      interp (n, G, 6) linear-interp skeleton between the two boundary steps
      pL_mw/pR_mw (n,6), nd_mw (n,G)  boundaries + net_demand in MW, for the in-graph
                                      projection used by the `unrolled`/`rayen_traj`
                                      constraint modes (posthoc mode ignores them).
    W = 2*context + gap; the gap is centred (rows [context, context+gap))."""
    W = 2 * context + gap
    tfi = np.asarray(TARGET_FEAT_IDX)
    X = np.stack([Xflat[s:s + W] for s in starts]).astype(np.float32)   # (n,W,17)
    # y-scaled truth in the gap, TARGETS order (from the target flat, not X)
    gs, ge = context, context + gap
    Y = np.stack([Yflat[s + gs:s + ge] for s in starts]).astype(np.float32)  # (n,G,6)
    # linear-interp skeleton (y-scaled) from the boundary steps 10:55 / 14:00 -> the
    # model learns the DEVIATION from this, so it starts at the interp baseline and
    # can only improve (fill = interp + dev). Boundaries are real (outside the gap).
    pL = Yflat[starts + gs - 1]                                        # (n,6) y-scaled
    pR = Yflat[starts + ge]                                            # (n,6)
    t = (np.arange(1, gap + 1) / (gap + 1))[None, :, None]             # (1,G,1)
    interp = (pL[:, None, :] + t * (pR - pL)[:, None, :]).astype(np.float32)  # (n,G,6)
    mask = np.ones((len(starts), W, 1), dtype=np.float32)
    mask[:, gs:ge, :] = 0.0
    X[:, gs:ge, tfi] = 0.0                                              # blank the unknown subspace
    nd_mw = np.stack([f.col_mw(Xflat, ND_COL)[s + gs:s + ge] for s in starts]).astype(np.float32)
    return {"X": X, "mask": mask, "Y": Y, "interp": interp,
            "pL_mw": f.y_to_mw(pL).astype(np.float32),                 # (n,6)
            "pR_mw": f.y_to_mw(pR).astype(np.float32),                 # (n,6)
            "nd_mw": nd_mw,                                            # (n,G)
            "context": context, "gap": gap, "W": W}


def sample_train_windows(f: Flats, n: int, context: int = 48, gap: int = 36,
                         seed: int = 0, split: str = "train") -> dict:
    """Sample `n` random-position imputation windows from a flat split (`train` or
    `val`): a centred gap with `context` real steps on each side. Random positions
    are right for TRAINING (more variety, calendar features carry time-of-day) --
    but NOT for early-stopping validation: use sample_val_midday_windows for that,
    which matches the test task (see its docstring for the leakage history)."""
    Xflat, Yflat = (f.Xtr, f.Ytr) if split == "train" else (f.Xva, f.Yva)
    rng = np.random.default_rng(seed)
    W = 2 * context + gap
    starts = rng.integers(0, Xflat.shape[0] - W, size=n)
    return _build_windows(f, Xflat, Yflat, starts, context, gap)


# --------------------------- validation gaps (midday-matched) ---------------------------

def _hour_frac(Xflat: np.ndarray, f: Flats) -> np.ndarray:
    """Recover fractional hour-of-day from the standardized hour_sin/cos features
    (the flats carry no timestamps). The prep encoded hour = h + min/60
    (lib/pipeline.py), so atan2 inverts it exactly at 5-min resolution."""
    s = Xflat[:, HOUR_SIN_COL] * f.x_scale[HOUR_SIN_COL] + f.x_mean[HOUR_SIN_COL]
    c = Xflat[:, HOUR_COS_COL] * f.x_scale[HOUR_COS_COL] + f.x_mean[HOUR_COS_COL]
    return (np.arctan2(s, c) % (2 * np.pi)) / (2 * np.pi) * 24.0


def sample_val_midday_windows(f: Flats, context: int = 48, gap: int = 36) -> dict:
    """Early-stopping validation windows that MATCH the test task: gaps pinned to
    11:00-14:00 in the held-out val split (~one window per val day).

    Why this exists -- the leakage/mismatch history of train.py:
      v1  picked the best epoch by TEST recon WAPE            -> test leakage.
      v2  early-stopped on RANDOM-position val gaps           -> leak fixed, but a
          difficulty MISMATCH: random gaps average over easy night hours, while the
          test gaps are all midday (solar trough, battery-heavy) -- val 0.31 looked
          better than test 0.44 and could still mis-rank epochs.
      v3  (this) same held-out val split, gaps at the deployment clock position --
          the val metric now estimates exactly the deployed quantity. Test is
          scored ONCE, after training, on the val-selected model.

    Same return contract as sample_train_windows. Gap rows are required to lie in
    genuine val territory (past the lb_carry train-tail rows prepended for context),
    and the recovered clock must advance 5 min/row across the whole window (skips
    splices left by dropna in the table build)."""
    Xflat, Yflat = f.Xva, f.Yva
    W = 2 * context + gap
    hf = _hour_frac(Xflat, f)
    cand = np.where(np.abs(hf - GAP_HOURS[0]) < 1.0 / 24)[0]   # rows whose clock reads 11:00
    starts = cand - context
    starts = starts[(starts >= 0) & (starts + W <= len(Xflat))]
    starts = starts[starts + context >= f.lb_carry]            # gap fully in real val rows
    ok = np.isclose(np.diff(hf) % 24.0, 1.0 / 12.0, atol=1e-3)  # 5-min steps (mod midnight)
    run = np.concatenate([[0], np.cumsum(ok)])
    starts = starts[run[starts + W - 1] - run[starts] == W - 1]
    if len(starts) == 0:
        raise RuntimeError("no contiguous 11:00-14:00 windows recoverable from the val flat")
    return _build_windows(f, Xflat, Yflat, starts, context, gap)


# --------------------------- general reconstruction windows (any split, any hour) ---------------------------

def _split_flats(f: Flats, split: str):
    return {"train": (f.Xtr, f.Ytr), "val": (f.Xva, f.Yva), "test": (f.Xte, f.Yte)}[split]


def sample_recon_windows(f: Flats, split: str = "test", n: int = 400, context: int = 48,
                         gap: int = 36, seed: int = 0, midday: bool = False,
                         min_windows: int = 30) -> list[GapWindow]:
    """GENERAL-position gap windows for reconstruction scoring, as GapWindow objects
    (projection quantities pL/pR/truth/nd + the recovered gap-open hour). Unlike
    test_gap_windows (deployment-only 11:00-14:00), gaps land at ALL times of day so
    the eval measures gap-filling in general, not just the midday shape (user concern
    #1). `midday=True` keeps only gaps opening at 11:00 -- the deployment slice we
    still report. Gaps are placed in genuine split rows (past the lb_carry context
    prefix) with contiguous 5-min spacing. **Fails loud** if fewer than `min_windows`
    are found, so a stale/rebuilt prepared.npz can't silently shrink the eval."""
    Xflat, Yflat = _split_flats(f, split)
    lb = f.lb_carry if split in ("val", "test") else 0
    rng = np.random.default_rng(seed)
    W = 2 * context + gap
    hf = _hour_frac(Xflat, f)
    ok = np.isclose(np.diff(hf) % 24.0, 1.0 / 12.0, atol=1e-3)     # contiguous 5-min steps
    run = np.concatenate([[0], np.cumsum(ok)])
    idx = f.test_index if split == "test" else None
    out: list[GapWindow] = []
    seen: set[int] = set()
    tries, cap = 0, max(n * 50, 6000)
    while len(out) < n and tries < cap:
        s = int(rng.integers(0, len(Xflat) - W)); tries += 1
        if s in seen:
            continue
        seen.add(s)
        g0 = s + context                                          # first gap row
        if g0 < lb:                                               # keep the gap in real split rows
            continue
        if midday and abs(hf[g0] - GAP_HOURS[0]) > 1.0 / 24:
            continue
        if run[s + W - 1] - run[s] != W - 1:                      # window must be contiguous
            continue
        gidx = np.arange(g0, g0 + gap)
        day = pd.Timestamp(idx[g0 - lb]).normalize() if (idx is not None and g0 - lb < len(idx)) else None
        out.append(GapWindow(
            day=day, gap_idx=gidx,
            ctxL_idx=np.arange(s, g0), ctxR_idx=np.arange(g0 + gap, s + W),
            pL_mw=f.y_to_mw(Yflat[g0 - 1]), pR_mw=f.y_to_mw(Yflat[g0 + gap]),
            truth_mw=f.y_to_mw(Yflat[gidx]), nd_mw=f.col_mw(Xflat, ND_COL)[gidx],
            hour=float(hf[g0]),
        ))
    if len(out) < min_windows:
        raise RuntimeError(
            f"only {len(out)} recon windows for split={split!r} midday={midday} "
            f"(need >={min_windows}). X{split}_flat has {len(Xflat)} rows -- is "
            f"prepared.npz the committed one?")
    return out


if __name__ == "__main__":
    f = load_flats()
    print("SIGN . truth == net_demand identity check (test):")
    nd = f.col_mw(f.Xte, ND_COL)
    resid = np.abs(f.y_to_mw(f.Yte) @ SIGN - nd)
    print(f"  max |SIGN.P - net_demand| = {resid.max():.3f} MW, mean {resid.mean():.4f} MW")
    gws = test_gap_windows(f)
    print(f"test 11:00-14:00 gap-days built: {len(gws)}  (gap {len(gws[0].gap_idx)} steps each)")
    tr = sample_train_windows(f, 4)
    print("train sample:", {k: v.shape for k, v in tr.items() if hasattr(v, 'shape')})
    va = sample_val_midday_windows(f)
    # verify the masked steps really sit at 11:00-13:55: recover the clock from the
    # window features themselves (the source cols are zeroed, hour_sin/cos are not)
    gs, ge = va["context"], va["context"] + va["gap"]
    hrs = np.concatenate([_hour_frac(w[gs:ge], f) for w in va["X"]])
    print(f"val midday windows: {len(va['X'])}  gap clock span "
          f"{hrs.min():.3f}-{hrs.max():.3f}h (want 11.000-13.917)")
    # general reconstruction windows (any hour) for test + val, plus the midday slice
    for split in ("val", "test"):
        gen = sample_recon_windows(f, split=split, n=400, context=48, seed=1)
        hh = np.array([w.hour for w in gen])
        print(f"recon '{split}' general: {len(gen)} windows, gap-open hours "
              f"{hh.min():.1f}-{hh.max():.1f} (mean {hh.mean():.1f})")
    slc = sample_recon_windows(f, split="test", n=400, context=48, seed=1, midday=True)
    print(f"recon 'test' midday slice: {len(slc)} windows all opening at "
          f"{np.mean([w.hour for w in slc]):.2f}h")
