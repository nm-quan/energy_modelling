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

def sample_train_windows(f: Flats, n: int, context: int = 72, gap: int = 36,
                         seed: int = 0, split: str = "train") -> dict:
    """Sample `n` random imputation windows from a flat split (`train` or `val`):
    a centred gap with `context` real steps on each side. Returns model-ready
    tensors:
      X    (n, W, 17)  features, with the 6 source cols ZEROED inside the gap
      mask (n, W, 1)   1 outside the gap, 0 inside (which steps are blanks)
      Y    (n, G, 6)   true sources (y-scaled, TARGETS order) in the gap
      interp (n, G, 6) linear-interp skeleton between the two boundary steps
    W = 2*context + gap. `val` draws from the held-out val split for early stopping
    (no test leakage). No timestamps needed -- calendar features carry time-of-day."""
    Xflat, Yflat = (f.Xtr, f.Ytr) if split == "train" else (f.Xva, f.Yva)
    rng = np.random.default_rng(seed)
    W = 2 * context + gap
    N = Xflat.shape[0]
    starts = rng.integers(0, N - W, size=n)
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
    mask = np.ones((n, W, 1), dtype=np.float32)
    mask[:, gs:ge, :] = 0.0
    X[:, gs:ge, tfi] = 0.0                                              # blank the unknown subspace
    return {"X": X, "mask": mask, "Y": Y, "interp": interp,
            "context": context, "gap": gap, "W": W}


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
