"""Verify a prediction CSV against the per-target 5-min data-max caps and the
per-target asymmetric data ramp rates from CONSTRAINT.md /
data/capacity_constraints.json + data/empirical_bounds.json.

Reports per target:
  1. upper cap violations (pred > data max)
  2. negative predictions (pred < 0)
  3. ramp-up violations (pred[t] - pred[t-1] > ramp_up)
  4. ramp-down violations (pred[t] - pred[t-1] < ramp_down)
Plus two aggregate checks:
  5. sum identity: hydro + coal_brown + gas_steam + gas_ocgt
                   + battery_discharging - battery_charging  ==  net_demand
     (net_demand is read from data/preprocessed/<res>/net_dispatch/table.parquet)
  6. battery SOC feasibility: with no observed SOC at t=0, test whether ANY
     initial SOC keeps the battery energy reservoir within [0, capacity] over
     the whole horizon. Feasible <=> swing(cumulative energy) <= capacity.
Ramps and SOC are only accumulated across consecutive 5-min intervals (gaps
start a new segment).

Usage:
    python script/check_caps.py
    python script/check_caps.py --csv data/gru_predictions_5min.csv --res 5min
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# CANON caps + ramps = empirical max over the FULL history (CONSTRAINT.md):
# data/preprocessed/hist/5min/net_dispatch_totdem/table.parquet, 500,832 rows,
# 2021-10-01..2026-07-05 (gap-free 5-min), so every historical actual is feasible.
# Supersedes the last365 numbers (capacity_constraints.json / empirical_bounds.json).

# per-target max output (MW). Rounded OUTWARD (up) to 0.1 so the exact historical
# extreme sits inside the cap (0 actual violations across all 500,832 rows).
CAPS = {
    "hydro":               2168.0,
    "coal_brown":          4895.8,
    "gas_steam":            516.4,
    "gas_ocgt":            1748.6,
    "battery_charging":    1611.6,
    "battery_discharging": 1687.5,
}

# per-target max 5-min step (MW per 5 min), asymmetric: (down, up). Rounded OUTWARD
# (down more negative, up higher) to 0.1. coal_brown down = -1553.6 is a single
# unit-trip (2024-02-13); kept per the empirical-max policy, so the coal down-ramp
# is effectively non-binding.
RAMPS = {
    "hydro":               (-735.4,  956.5),
    "coal_brown":         (-1553.6,  333.4),
    "gas_steam":           (-499.9,   82.5),
    "gas_ocgt":            (-363.9,  400.3),
    "battery_charging":    (-863.1,  795.0),
    "battery_discharging": (-670.9,  715.1),
}

# Battery energy reservoir, from capacity_constraints.json (full VIC fleet).
BATT_CAP_MWH = 4735.75            # nameplate installed storage
# Hours per interval, used to turn MW -> MWh when integrating the reservoir.
RES_HOURS = {"5min": 5 / 60.0, "30min": 0.5, "1h": 1.0}


def check(csv_path: Path, suffix: str = "_pred") -> None:
    df = pd.read_csv(csv_path)
    print(f"file: {csv_path}")
    print(f"rows: {len(df):,}  (checking '{{target}}{suffix}' columns)\n")

    # Sort + identify consecutive 5-min pairs for ramp checks
    if "interval" in df.columns:
        df = df.sort_values("interval").reset_index(drop=True)
        ts = pd.to_datetime(df["interval"], utc=False, errors="coerce")
        dt = ts.diff().dt.total_seconds().values  # seconds since prev row
        consec = (dt == 300)                      # True for valid consecutive pairs
    else:
        consec = np.ones(len(df), dtype=bool)
        consec[0] = False

    rows_upper, rows_lower, rows_ramp = [], [], []
    for t, cap in CAPS.items():
        col = f"{t}{suffix}"
        if col not in df.columns:
            print(f"  skip {t}: column {col} not found")
            continue
        p = df[col].values

        over = p > cap
        n_over = int(over.sum())
        rows_upper.append((t, cap, n_over,
                           float((p[over] - cap).max()) if n_over else 0.0,
                           float((p[over] - cap).mean()) if n_over else 0.0))

        under = p < 0
        n_under = int(under.sum())
        if n_under:
            rows_lower.append((t, n_under, float(-p[under].min())))

        # ramp check on consecutive pairs only
        rdown, rup = RAMPS[t]
        delta = np.diff(p, prepend=p[0])
        mask = consec  # already len(df), first entry False
        up_viol = (delta > rup) & mask
        dn_viol = (delta < rdown) & mask
        n_up, n_dn = int(up_viol.sum()), int(dn_viol.sum())
        max_up = float((delta[up_viol] - rup).max()) if n_up else 0.0
        max_dn = float((rdown - delta[dn_viol]).max()) if n_dn else 0.0
        rows_ramp.append((t, rdown, rup, n_up, max_up, n_dn, max_dn))

    print("Upper-cap violations (pred > data max):")
    print(f"  {'target':20s} {'cap':>9s} {'n_over':>8s} {'max_excess':>12s} {'mean_excess':>12s}")
    for t, cap, n, mx, mn in rows_upper:
        print(f"  {t:20s} {cap:9.1f} {n:8d} {mx:12.2f} {mn:12.2f}")
    total = sum(r[2] for r in rows_upper)
    print(f"\n  total cap violations: {total:,} ({100 * total / (len(df) * len(CAPS)):.3f}% of cells)")

    print("\nNegative predictions (pred < 0):")
    if rows_lower:
        print(f"  {'target':20s} {'n_under':>8s} {'max_below':>10s}")
        for t, n, m in rows_lower:
            print(f"  {t:20s} {n:8d} {m:10.2f}")
    else:
        print("  none")

    n_consec = int(consec.sum())
    print(f"\nRamp violations (consecutive 5-min pairs only; {n_consec:,} pairs):")
    print(f"  {'target':20s} {'down_lim':>9s} {'up_lim':>9s} "
          f"{'n_up':>6s} {'max_up_excess':>14s} {'n_down':>7s} {'max_dn_excess':>14s}")
    total_ramp = 0
    for t, rd, ru, nu, mu, nd, md in rows_ramp:
        total_ramp += nu + nd
        print(f"  {t:20s} {rd:9.1f} {ru:9.1f} {nu:6d} {mu:14.2f} {nd:7d} {md:14.2f}")
    print(f"\n  total ramp violations: {total_ramp:,} "
          f"({100 * total_ramp / (n_consec * len(CAPS)):.3f}% of consecutive cells)")


def check_sum_identity(csv_path: Path, res: str, suffix: str = "_pred") -> None:
    """sum(dispatch) == net_demand, where
    net_demand = hydro + coal_brown + gas_steam + gas_ocgt + battery_discharging - battery_charging.

    Reference net_demand: the matching in-CSV net_demand column when present
    (net_demand_pred_sim / _pred_base / _actual), otherwise the actual
    net_demand from the preprocessed table at the given resolution.
    """
    pred_df = pd.read_csv(csv_path)
    pred_df["interval"] = pd.to_datetime(pred_df["interval"])
    nd_map = {"_pred_sim": "net_demand_pred_sim", "_pred_base": "net_demand_pred_base",
              "_actual": "net_demand_actual"}
    nd_col = nd_map.get(suffix)
    if nd_col and nd_col in pred_df.columns:
        merged = pred_df.set_index("interval")
        merged = merged.assign(net_demand_actual=merged[nd_col])
        src = f"CSV {nd_col}"
    else:
        table = ROOT / "data" / "preprocessed" / res / "net_dispatch" / "table.parquet"
        if not table.exists():
            print(f"\nSum identity: skipped (no {table})")
            return
        tab = pd.read_parquet(table)
        tab.index = pd.to_datetime(tab.index)
        merged = pred_df.set_index("interval").join(
            tab["net_demand"].rename("net_demand_actual"), how="left")
        if merged["net_demand_actual"].isna().any():
            n_miss = int(merged["net_demand_actual"].isna().sum())
            print(f"\nSum identity: warning, {n_miss} rows missing net_demand (skipped)")
            merged = merged.dropna(subset=["net_demand_actual"])
        src = "preprocessed table (actual)"

    sum_pred = (merged[f"hydro{suffix}"] + merged[f"coal_brown{suffix}"]
                + merged[f"gas_steam{suffix}"] + merged[f"gas_ocgt{suffix}"]
                + merged[f"battery_discharging{suffix}"] - merged[f"battery_charging{suffix}"])
    resid = (sum_pred - merged["net_demand_actual"]).values

    print(f"\nSum identity (sum dispatch{suffix} vs net_demand [{src}], {len(merged):,} rows):")
    print(f"  mean residual: {resid.mean():+8.2f} MW  std: {resid.std():7.2f}  max |r|: {np.abs(resid).max():7.2f}")
    print(f"  {'threshold':>10s} {'n_violations':>14s} {'%':>8s}")
    for thr in [1.0, 5.0, 10.0, 50.0, 100.0]:
        n = int((np.abs(resid) > thr).sum())
        print(f"  {thr:10.2f} {n:14d} {100*n/len(resid):8.2f}")


def calibrate_round_trip_eta(res: str) -> float | None:
    """Empirical round-trip efficiency = energy discharged / energy charged,
    integrated over the full preprocessed series at this resolution.

    Lossless accounting is non-physical for an aggregate battery: you charge
    more energy than you discharge, so the loss energy would pile up forever.
    eta_rt = sum(discharge) / sum(charge) is the loss factor that makes net
    energy balance over the calibration window. Returns None if no table."""
    table = ROOT / "data" / "preprocessed" / res / "net_dispatch" / "table.parquet"
    if not table.exists():
        return None
    tab = pd.read_parquet(table)
    dt = RES_HOURS[res]
    e_in = float(tab["battery_charging"].clip(lower=0).sum()) * dt
    e_out = float(tab["battery_discharging"].clip(lower=0).sum()) * dt
    return e_out / e_in if e_in > 0 else None


def _soc_swing(chg: np.ndarray, dis: np.ndarray, seg: np.ndarray,
               eta_side: float, dt: float):
    """Per-segment cumulative reservoir energy and its peak-to-trough swing.

    Reservoir step dE = chg*eta_side*dt - dis/eta_side*dt (symmetric split of
    eta_rt = eta_side**2). Within a segment SOC(t) = SOC0 + cumsum(dE); a
    feasible SOC0 in [0, cap] exists iff swing = max(cumE) - min(cumE) <= cap.
    Returns (worst_swing, worst_min_cum, worst_max_cum, n_segments)."""
    dE = (chg * eta_side - dis / eta_side) * dt
    worst = (-1.0, 0.0, 0.0)
    n_seg = 0
    for s in np.unique(seg):
        m = seg == s
        n_seg += 1
        cum = np.concatenate([[0.0], np.cumsum(dE[m])])
        sw = float(cum.max() - cum.min())
        if sw > worst[0]:
            worst = (sw, float(cum.min()), float(cum.max()))
    return worst[0], worst[1], worst[2], n_seg


def check_soc_feasibility(csv_path: Path, res: str,
                          cap_mwh: float = BATT_CAP_MWH,
                          eta_rt: float | None = None,
                          suffix: str = "_pred",
                          cap_from_data: bool = False) -> None:
    """Existence test: does any initial SOC keep the predicted battery energy
    reservoir inside [0, cap] for the whole horizon? (No observed SOC at t=0.)
    Runs on the chosen series (suffix) and, when present, *_actual as baseline.

    cap_from_data: use the actual series' own reservoir swing as the capacity
    (the empirically-demonstrated ceiling) instead of the nameplate storage.
    """
    df = pd.read_csv(csv_path)
    primary = suffix.lstrip("_")
    if f"battery_charging_{primary}" not in df.columns:
        print(f"\nSOC feasibility: skipped (no battery_charging_{primary} column)")
        return

    dt = RES_HOURS[res]
    if eta_rt is None:
        eta_rt = calibrate_round_trip_eta(res)
    if eta_rt is None:
        print("\nSOC feasibility: skipped (no preprocessed table to calibrate eta)")
        return
    eta_side = np.sqrt(eta_rt)

    # consecutive-interval segments (gaps start a new reservoir trajectory)
    if "interval" in df.columns:
        df = df.sort_values("interval").reset_index(drop=True)
        ts = pd.to_datetime(df["interval"], errors="coerce")
        gap = ts.diff().dt.total_seconds().values != dt * 3600
        gap[0] = True
        seg = np.cumsum(gap)
    else:
        seg = np.ones(len(df), dtype=int)

    # pass 1: reservoir swing for each available series
    results = []  # (kind, swing, cmin, cmax, n_seg)
    for kind in dict.fromkeys(["actual", primary]):
        cc, dc = f"battery_charging_{kind}", f"battery_discharging_{kind}"
        if cc not in df.columns:
            continue
        chg = np.clip(df[cc].values, 0, None)
        dis = np.clip(df[dc].values, 0, None)
        results.append((kind, *_soc_swing(chg, dis, seg, eta_side, dt)))

    # capacity ceiling: nameplate storage, or the actual series' own swing
    cap_src = "nameplate capacity_storage"
    if cap_from_data:
        act = next((r for r in results if r[0] == "actual"), None)
        if act is not None:
            cap_mwh = act[1]
            cap_src = "actual-data reservoir swing"

    print(f"\nBattery SOC feasibility (cap {cap_mwh:.1f} MWh [{cap_src}], "
          f"eta_rt={eta_rt:.4f} -> per-side {eta_side:.4f}):")
    print(f"  {'series':10s} {'n_seg':>5s} {'swing_MWh':>10s} "
          f"{'headroom':>9s} {'%cap':>6s}  feasible  SOC0_window_MWh")
    for kind, sw, cmin, cmax, n_seg in results:
        feasible = sw <= cap_mwh + 1e-6
        lo, hi = -cmin, cap_mwh - cmax            # feasible SOC0 range
        win = f"[{lo:8.1f}, {hi:8.1f}]" if feasible else "  (empty)"
        print(f"  {kind:10s} {n_seg:5d} {sw:10.1f} "
              f"{cap_mwh - sw:9.1f} {100 * sw / cap_mwh:6.1f}  "
              f"{'YES' if feasible else 'NO ':>8s}  {win}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "ml" / "lstm_5min_mse" / "predictions.csv")
    p.add_argument("--res", default="5min", choices=["1h", "30min", "5min"])
    p.add_argument("--batt-cap", type=float, default=BATT_CAP_MWH,
                   help="battery energy reservoir capacity in MWh")
    p.add_argument("--eta", type=float, default=None,
                   help="round-trip efficiency override (default: calibrate from data)")
    p.add_argument("--suffix", default="_pred",
                   help="prediction column suffix to check, e.g. _pred (default), "
                        "_pred_sim or _pred_base or _actual for the demand-simulation CSVs")
    p.add_argument("--soc-cap-from-data", action="store_true",
                   help="use the actual series' own reservoir swing as the SOC capacity "
                        "(empirical ceiling) instead of the nameplate storage")
    a = p.parse_args()
    check(a.csv, a.suffix)
    check_sum_identity(a.csv, a.res, a.suffix)
    check_soc_feasibility(a.csv, a.res, cap_mwh=a.batt_cap, eta_rt=a.eta, suffix=a.suffix,
                          cap_from_data=a.soc_cap_from_data)


if __name__ == "__main__":
    main()
