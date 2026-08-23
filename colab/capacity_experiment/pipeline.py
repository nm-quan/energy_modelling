"""Shared data pipeline for the capacity experiments.

Single source of data: data/updata/, window 2025-01-01 .. 2026-08-14.
Preprocessing is exactly what capacity.ipynb cells 4 and 6 do -- read the three
updata parquets, join, add calendar features, join the MONTHLY capacity table,
then keep only complete clean days. Nothing is read from data/preprocessed/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging",
           "wind", "solar_utility", "curtailment_wind", "curtailment_solar"]
OBS = ["demand", "price"]
CAL = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
GEN_FT = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
          "battery_charging", "battery_discharging", "wind", "solar_utility"]

STEPS_PER_DAY = 288
GAP = 36                      # 3h gap (36 x 5min)
GAP_EVAL_START = 132          # eval gap pinned to 11:00-14:00


def data_dir() -> Path:
    """Locate data/updata, in Colab or from anywhere in the repo."""
    try:
        import google.colab  # noqa: F401
        p = Path("/content/updata")
        return p if p.exists() else Path("/content")
    except Exception:
        d = Path.cwd()
        while d != d.parent and not (d / "data" / "updata").exists():
            d = d.parent
        return d / "data" / "updata"


def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


def load_table(dd: Path | None = None) -> pd.DataFrame:
    """The joined 5-min table: generation + market + curtailment + calendar + cap_*."""
    dd = dd or data_dir()
    g = pd.read_parquet(dd / "vic_generation_20250101_20260814.parquet")
    g["interval"] = _naive(g["interval"])
    gw = (g.pivot_table(index="interval", columns="fueltech", values="power_mw",
                        aggfunc="first").reindex(columns=GEN_FT))

    m = pd.read_parquet(dd / "vic_market_20250101_20260814.parquet")
    m["interval"] = _naive(m["interval"])
    m = (m.set_index("interval")
           .rename(columns={"demand_mw": "demand", "price_aud_per_mwh": "price"})[OBS])

    c = pd.read_parquet(dd / "vic_curtailment_20250101_20260814.parquet")
    c["interval"] = _naive(c["interval"])
    c = (c.set_index("interval")
           .rename(columns={"curtailment_wind_mw": "curtailment_wind",
                            "curtailment_solar_mw": "curtailment_solar"})
           [["curtailment_wind", "curtailment_solar"]])

    df = gw.join(m, how="outer").join(c, how="outer").sort_index()

    h = df.index.hour + df.index.minute / 60.0
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    dow = df.index.dayofweek
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    # MONTHLY capacity, forward-filled onto the 5-min grid (piecewise constant)
    cap = pd.read_parquet(dd / "capacity_VIC1_by_fueltech_MS.parquet")
    cap["interval"] = _naive(cap["interval"])
    cw = (cap.pivot_table(index="interval", columns="fueltech", values="capacity_mw",
                          aggfunc="first").reindex(df.index, method="ffill"))
    cw.columns = [f"cap_{k}" for k in cw.columns]
    return df.join(cw)


def metrics(P: np.ndarray, Y: np.ndarray, targets: list[str] = TARGETS):
    """Per-channel WAPE/MAE/RMSE/R2 + macro (nanmean). P, Y are (..., n_targets) in MW.

    Identical to metrics() in capacity.ipynb so the numbers are directly comparable.
    """
    p, y = P.reshape(-1, len(targets)), Y.reshape(-1, len(targets))
    out = {}
    for i, t in enumerate(targets):
        e = p[:, i] - y[:, i]
        den = np.abs(y[:, i]).sum()
        sstot = ((y[:, i] - y[:, i].mean()) ** 2).sum()
        out[t] = {"WAPE": float(np.abs(e).sum() / den) if den > 0 else np.nan,
                  "MAE": float(np.abs(e).mean()),
                  "RMSE": float(np.sqrt((e ** 2).mean())),
                  "R2": float(1 - (e ** 2).sum() / sstot) if sstot > 0 else np.nan}
    macro = {mk: float(np.nanmean([out[t][mk] for t in targets]))
             for mk in ["WAPE", "MAE", "RMSE", "R2"]}
    # magnitude-weighted WAPE: one number over all channels, not an unweighted mean
    macro["microWAPE"] = float(np.abs(p - y).sum() / np.abs(y).sum())
    return out, macro


def top20_days(X: np.ndarray, cols: list[str]) -> np.ndarray:
    """Indices of the 20 highest-demand days, ranked by within-day peak demand.

    Same selection as capacity.ipynb. X may be raw or standardised -- demand scaling is
    monotone, so the ranking is unchanged either way.
    """
    di = cols.index("demand")
    return np.argsort(-X[:, :, di].max(1))[:20]


def make_days(frame: pd.DataFrame, cols: list[str]):
    """Whole-day tensors (D, 288, C) over complete, clean days only."""
    sub = frame[cols].copy()
    for cc in ("curtailment_wind", "curtailment_solar"):
        if cc in sub:
            sub[cc] = sub[cc].fillna(0.0)
    sub = sub.interpolate(limit=6, limit_direction="both")
    arrs, days = [], []
    for d, gi in sub.groupby(sub.index.normalize()).groups.items():
        b = sub.loc[gi].sort_index()
        if len(b) != STEPS_PER_DAY or b.isna().any().any():
            continue
        arrs.append(b.to_numpy(np.float32))
        days.append(pd.Timestamp(d))
    return np.stack(arrs), pd.DatetimeIndex(days)
