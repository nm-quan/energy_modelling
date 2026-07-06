"""Shared preprocessing pipeline for the VIC NEM forecasting studies.

Loads market + generation data, builds a feature/target table at a chosen
resolution, splits chronologically (~80-10-10 by date), scales with
StandardScaler (fit on train only), and produces sliding windows.

Targets (all >= 0 by construction):
    hydro, coal_brown, gas (= gas_steam + gas_ocgt),
    battery_charging, battery_discharging
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]

# Named datasets. "last365" is the original bundled year (legacy demand semantics,
# frozen so every existing model/result stays reproducible). "hist" is the
# 2021-10 -> 2026-07 pro-plan pull (script/pull_data_hist.py) with VIC-island
# demand: demand_mw := demand_mw - net_import_mw, so the system is self-contained
# (local generation serves exactly the adjusted demand; exports are extra local
# load, imports are demand met elsewhere). NOTE: downstream scripts that build a
# demand-side net_demand by subtracting net_import themselves (sweep_eqnd etc.)
# must NOT subtract it again on "hist".
DATASETS = {
    "last365": dict(
        market="vic_market_last365.parquet",
        generation="vic_generation_last365.parquet",
        interconnector="vic_interconnector_last365.parquet",
        train_end=pd.Timestamp("2026-02-28 23:55:00+10:00"),
        val_end=pd.Timestamp("2026-04-11 23:55:00+10:00"),
        vic_island_demand=False,
    ),
    "hist": dict(
        market="vic_market_20211001_20260706.parquet",
        generation="vic_generation_20211001_20260706.parquet",
        interconnector="vic_interconnector_20211001_20260706.parquet",
        # ~79/10.5/10.5 by duration; test = Jan-Jul 2026 (record 10,784 MW peak
        # + winter), val = Jul-Dec 2025, train = Oct 2021 - Jun 2025.
        train_end=pd.Timestamp("2025-06-30 23:55:00+10:00"),
        val_end=pd.Timestamp("2025-12-31 23:55:00+10:00"),
        vic_island_demand=True,
    ),
}

# Legacy module-level split constants (many scripts import these); they describe
# the default "last365" dataset.
TRAIN_END = DATASETS["last365"]["train_end"]
VAL_END = DATASETS["last365"]["val_end"]


def load_raw(dataset: str = "last365") -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = DATASETS[dataset]
    market = pd.read_parquet(DATA_DIR / cfg["market"])
    gen = pd.read_parquet(DATA_DIR / cfg["generation"])
    market["interval"] = pd.to_datetime(market["interval"])
    gen["interval"] = pd.to_datetime(gen["interval"])
    return market, gen


def build_table(resolution: str = "1h", dataset: str = "last365") -> pd.DataFrame:
    """Return a wide table indexed by interval at the given resolution.

    resolution: "5min" (native) or "1h".
    Columns: per-fueltech power_mw (wide), demand_mw, price_aud_per_mwh,
    plus derived gas, net_demand, and calendar features.
    """
    cfg = DATASETS[dataset]
    market, gen = load_raw(dataset)

    # generation: long -> wide on power_mw
    gw = gen.pivot_table(index="interval", columns="fueltech",
                         values="power_mw", aggfunc="mean")
    gw = gw.sort_index()
    # the API's combined "battery" fueltech (hist pull) is 60% NaN and redundant
    # with battery_charging/discharging -- it would wipe those rows at dropna()
    gw = gw.drop(columns=["battery"], errors="ignore")

    m = market.set_index("interval").sort_index()[["demand_mw", "price_aud_per_mwh"]]

    df = gw.join(m, how="inner")

    if cfg["vic_island_demand"]:
        inter = pd.read_parquet(DATA_DIR / cfg["interconnector"])
        ni = (inter.assign(interval=pd.to_datetime(inter["interval"]))
                   .set_index("interval").sort_index()["net_import_mw"]
                   .reindex(df.index).interpolate(limit_direction="both"))
        df["demand_mw"] = df["demand_mw"] - ni

    if resolution == "1h":
        df = df.resample("1h").mean()
    elif resolution == "30min":
        df = df.resample("30min").mean()
    elif resolution != "5min":
        raise ValueError(f"unknown resolution {resolution!r}")

    df = df.interpolate(limit_direction="both")  # fill the small battery gaps

    # battery_charging is stored as negative draw in some feeds; targets must be >=0
    if "battery_charging" in df:
        df["battery_charging"] = df["battery_charging"].abs()
    if "battery_discharging" in df:
        df["battery_discharging"] = df["battery_discharging"].clip(lower=0)

    # net_demand = sum of dispatchable generation (load actually served by dispatchable
    # plant). battery_charging is a load, so it is subtracted; discharging adds supply.
    def _col(c):
        return df[c].fillna(0) if c in df.columns else 0.0
    df["net_demand"] = (
        _col("hydro") + _col("coal_brown") + _col("gas_steam") + _col("gas_ocgt")
        + _col("battery_discharging") - _col("battery_charging")
    )

    # calendar features (cyclical)
    idx = df.index
    hour = idx.hour + idx.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = idx.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    doy = idx.dayofyear
    df["season_sin"] = np.sin(2 * np.pi * doy / 365)
    df["season_cos"] = np.cos(2 * np.pi * doy / 365)
    df["is_weekend"] = (dow >= 5).astype(float)
    # peak: 07:00-22:00 local
    df["is_peak"] = ((idx.hour >= 7) & (idx.hour < 22)).astype(float)

    return df.dropna()


DISPATCHABLE_HIST = ["hydro", "gas_steam", "gas_ocgt", "coal_brown",
                     "battery_charging", "battery_discharging"]
ALL_ENERGY_HIST = ["wind", "solar_utility", "solar_rooftop"] + DISPATCHABLE_HIST


def select_features(df: pd.DataFrame, input_mode: str = "total_all") -> list[str]:
    """Feature columns for the two comparison configs (test #1).

    input_mode:
      'total_all'           -> total demand + ALL energy history (incl. wind & solar).
      'net_dispatch'        -> net demand   + dispatchable-only energy history.
      'net_dispatch_totdem' -> net_dispatch + total operational demand_mw as an
                               extra exogenous feature (for channel-mixing models
                               and the demand/price simulation).
    """
    extra: list[str] = []
    if input_mode == "total_all":
        energy_hist = [c for c in ALL_ENERGY_HIST if c in df.columns]
        demand_col = "demand_mw"
    elif input_mode == "net_dispatch":
        energy_hist = [c for c in DISPATCHABLE_HIST if c in df.columns]
        demand_col = "net_demand"
    elif input_mode == "net_dispatch_totdem":
        energy_hist = [c for c in DISPATCHABLE_HIST if c in df.columns]
        demand_col = "net_demand"
        extra = ["demand_mw"]
    elif input_mode == "total_dispatch":
        energy_hist = [c for c in DISPATCHABLE_HIST if c in df.columns]
        demand_col = "demand_mw"
    elif input_mode == "demand_price":
        energy_hist = []
        demand_col = "demand_mw"
    else:
        raise ValueError(f"unknown input_mode {input_mode!r}")
    cal = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
           "season_sin", "season_cos", "is_weekend", "is_peak"]
    return energy_hist + [demand_col] + extra + ["price_aud_per_mwh"] + cal


def make_windows(X: np.ndarray, Y: np.ndarray, lookback: int, horizon: int = 1):
    """Sliding windows: use X[i-lookback:i] -> target Y[i+horizon-1].

    Vectorized with stride tricks (no Python loop) for speed/memory on 5-min data.
    """
    n = len(X)
    n_valid = n - lookback - horizon + 1
    if n_valid <= 0:
        return (np.empty((0, lookback, X.shape[1]), np.float32),
                np.empty((0, Y.shape[1]), np.float32))
    win = sliding_window_view(X, lookback, axis=0)      # (n-lookback+1, feat, lookback)
    win = win.transpose(0, 2, 1)[:n_valid]              # (n_valid, lookback, feat)
    tgt = Y[lookback + horizon - 1: lookback + horizon - 1 + n_valid]
    return win.astype(np.float32, copy=False), tgt.astype(np.float32, copy=False)


PREPROCESSED_DIR = DATA_DIR / "preprocessed"


def prepare(resolution="1h", lookback=24, horizon=1, input_mode="total_all", save=True,
            dataset="last365"):
    """Full pipeline -> dict of windowed train/val/test arrays + scalers + meta.

    lookback is given in HOURS; converted to steps based on resolution.
    If save=True, writes the preprocessed table, windows and scalers under
    data/preprocessed/<resolution>/<input_mode>/ (dataset != last365 gets a
    dataset-named subtree and skips the windows archive -- see save_preprocessed).

    Memory: features are scaled to float32 BEFORE windowing, so the window
    arrays stay zero-copy strided views over the flat (n, F) array (~35 MB for
    the 4.75-year hist dataset instead of ~10 GB materialized). Batch indexing
    (numpy fancy index / DataLoader) copies only the batch.
    """
    cfg = DATASETS[dataset]
    df = build_table(resolution, dataset)
    feat_cols = select_features(df, input_mode)

    steps_per_hour = {"5min": 12, "30min": 2, "1h": 1}[resolution]
    lb_steps = lookback * steps_per_hour

    # chronological split by date
    train_df = df[df.index <= cfg["train_end"]]
    val_df = df[(df.index > cfg["train_end"]) & (df.index <= cfg["val_end"])]
    test_df = df[df.index > cfg["val_end"]]

    x_scaler = StandardScaler().fit(train_df[feat_cols].values)
    y_scaler = StandardScaler().fit(train_df[TARGETS].values)

    def scale(d):
        return (x_scaler.transform(d[feat_cols].values).astype(np.float32),
                y_scaler.transform(d[TARGETS].values).astype(np.float32))

    # Flat scaled arrays per split; val/test are prepended with lb_steps of
    # context from the prior split so their first windows are valid. Windowing
    # these flats is equivalent to the old windows_for() trimming logic and
    # exposes the flats for the compressed prepared.npz artifact.
    flats = {
        "tr": scale(train_df),
        "va": scale(pd.concat([train_df.tail(lb_steps), val_df])),
        "te": scale(pd.concat([val_df.tail(lb_steps), test_df])),
    }
    Xtr, Ytr = make_windows(*flats["tr"], lb_steps, horizon)
    Xva, Yva = make_windows(*flats["va"], lb_steps, horizon)
    Xte, Yte = make_windows(*flats["te"], lb_steps, horizon)

    # align test index to windowed targets
    test_target_idx = test_df.index[-len(Xte):]

    out = {
        "Xtr": Xtr, "Ytr": Ytr,
        "Xva": Xva, "Yva": Yva,
        "Xte": Xte, "Yte": Yte,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
        "feat_cols": feat_cols, "targets": TARGETS,
        "test_index": test_target_idx,
        "resolution": resolution, "lookback_steps": lb_steps, "horizon": horizon,
        "input_mode": input_mode, "dataset": dataset,
    }
    if save:
        save_preprocessed(out, df, flats)
    return out


def save_preprocessed(out, df, flats=None):
    """Persist the wide table, prepared arrays and scalers under
    data/preprocessed/[<dataset>/]<resolution>/<input_mode>/.

    The split+preprocessed+compressed artifact is prepared.npz: the FLAT scaled
    per-split arrays (val/test carry lb_steps of context rows) plus scalers and
    metadata -- everything load_prepared() needs, in one ~45 MB file for the
    4.75-year hist set. Windows are NOT stored (each row appears in 288
    overlapping windows -> ~10 GB); they reconstruct as zero-copy views in
    milliseconds on load. last365 additionally keeps its legacy windows.npz."""
    res = out["resolution"]
    dataset = out.get("dataset", "last365")
    base = PREPROCESSED_DIR if dataset == "last365" else PREPROCESSED_DIR / dataset
    d = base / res / out["input_mode"]
    d.mkdir(parents=True, exist_ok=True)

    df.to_parquet(d / "table.parquet")                       # full feature+target table
    if flats is not None:
        xs, ys = out["x_scaler"], out["y_scaler"]
        np.savez_compressed(
            d / "prepared.npz",
            Xtr_flat=flats["tr"][0], Ytr_flat=flats["tr"][1],
            Xva_flat=flats["va"][0], Yva_flat=flats["va"][1],
            Xte_flat=flats["te"][0], Yte_flat=flats["te"][1],
            test_index=np.asarray(out["test_index"].astype(str), dtype=str),
            x_mean=xs.mean_, x_scale=xs.scale_,
            y_mean=ys.mean_, y_scale=ys.scale_,
            feat_cols=np.array(out["feat_cols"]), targets=np.array(out["targets"]),
            lookback_steps=out["lookback_steps"], horizon=out["horizon"],
            resolution=res, dataset=dataset,
        )
    if dataset == "last365":
        np.savez_compressed(
            d / "windows.npz",
            Xtr=out["Xtr"], Ytr=out["Ytr"],
            Xva=out["Xva"], Yva=out["Yva"],
            Xte=out["Xte"], Yte=out["Yte"],
            test_index=out["test_index"].astype(str).values,
        )
    xs, ys = out["x_scaler"], out["y_scaler"]
    np.savez(
        d / "scalers.npz",
        x_mean=xs.mean_, x_scale=xs.scale_,
        y_mean=ys.mean_, y_scale=ys.scale_,
        feat_cols=np.array(out["feat_cols"]), targets=np.array(out["targets"]),
    )
    meta = {
        "resolution": res, "input_mode": out["input_mode"],
        "lookback_steps": out["lookback_steps"], "horizon": out["horizon"],
        "n_train": int(len(out["Xtr"])), "n_val": int(len(out["Xva"])),
        "n_test": int(len(out["Xte"])),
        "features": out["feat_cols"], "targets": out["targets"],
        "dataset": dataset,
        "train_end": str(DATASETS[dataset]["train_end"]),
        "val_end": str(DATASETS[dataset]["val_end"]),
    }
    import json
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved preprocessed -> {d}")


def load_prepared(resolution="5min", lookback=24, horizon=1,
                  input_mode="net_dispatch_totdem", dataset="hist"):
    """Load prepared.npz and reconstruct the prepare(save=False) dict without
    touching parquets or refitting scalers -- one small file, window views
    rebuilt on the fly. This is the entry point for remote (Colab) training:
    the npz ships in the repo, so a clone is all a GPU box needs.

    lookback/horizon must match what the artifact was saved with (the flat
    val/test arrays embed exactly lookback_steps rows of boundary context).
    """
    base = PREPROCESSED_DIR if dataset == "last365" else PREPROCESSED_DIR / dataset
    z = np.load(base / resolution / input_mode / "prepared.npz", allow_pickle=False)
    steps_per_hour = {"5min": 12, "30min": 2, "1h": 1}[resolution]
    lb_steps = lookback * steps_per_hour
    if lb_steps != int(z["lookback_steps"]) or horizon != int(z["horizon"]):
        raise ValueError(f"prepared.npz was saved with lookback_steps="
                         f"{int(z['lookback_steps'])}, horizon={int(z['horizon'])}")

    def scaler(mean, scale):
        s = StandardScaler()
        s.mean_, s.scale_ = mean, scale
        s.var_, s.n_features_in_ = scale ** 2, len(mean)
        return s

    Xtr, Ytr = make_windows(z["Xtr_flat"], z["Ytr_flat"], lb_steps, horizon)
    Xva, Yva = make_windows(z["Xva_flat"], z["Yva_flat"], lb_steps, horizon)
    Xte, Yte = make_windows(z["Xte_flat"], z["Yte_flat"], lb_steps, horizon)
    return {
        "Xtr": Xtr, "Ytr": Ytr, "Xva": Xva, "Yva": Yva, "Xte": Xte, "Yte": Yte,
        "x_scaler": scaler(z["x_mean"], z["x_scale"]),
        "y_scaler": scaler(z["y_mean"], z["y_scale"]),
        "feat_cols": [str(c) for c in z["feat_cols"]],
        "targets": [str(t) for t in z["targets"]],
        "test_index": pd.DatetimeIndex(z["test_index"]),
        "resolution": resolution, "lookback_steps": lb_steps, "horizon": horizon,
        "input_mode": input_mode, "dataset": dataset,
    }


if __name__ == "__main__":
    d = prepare("1h")
    print("features:", d["feat_cols"])
    print("targets :", d["targets"])
    for k in ["Xtr", "Ytr", "Xva", "Yva", "Xte", "Yte"]:
        print(f"{k}: {d[k].shape}")
