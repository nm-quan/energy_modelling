"""Pull per-unit battery POWER for VIC1 from OpenElectricity.

The companion to pull_battery_soc.py. That script gives E(t) per unit in MWh; this one
gives P(t) per unit in MW, which is the other half of the SOC balance:

    E(t+1) = E(t) + eta_c * P_charge * dt  -  P_discharge * dt / eta_d

Without per-unit power the balance cannot be tested or enforced: the repo has fleet
power (vic_generation, two summed channels) and per-unit energy, and there is no way to
attribute a fleet-level MW to an individual reservoir.

Same endpoint and the same traps as the SOC pull:
  * filter by unit_code so only BIDIRECTIONAL units come back -- filtering by
    facility_code returns each battery 2-3x (BALB1 / BALBG1 / BALBL1 are one asset)
  * 30-day chunks (PRO tier limit at 5m)
  * default window starts 2025-06-30 to align with the SOC file, so the two join

POWER IS INSTANTANEOUS, NOT AN INTERVAL AVERAGE. Per the OpenElectricity energy guide,
energy is the trapezoid  E_i = (P_i + P_i-1)/2 * dt,  dt = 1/12 h at 5m. Using
P_i * dt biases every unit's discharge efficiency above 1. Do not skip this.

Sign convention is checked on arrival: on the BIDIRECTIONAL unit the series is signed,
negative = charging.

Output: data/updata/vic_battery_power.csv -- interval, then <UNIT>_mw per unit.

Usage:
    OPENELECTRICITY_API_KEY=... python script/pull_battery_power.py
    python script/pull_battery_power.py --start 2025-06-30 --end 2026-08-14
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
def out_path(region: str, network: str = "NEM"):
    """Region-named output. vic_battery_power.csv is kept for VIC1/NEM so
    existing analyses do not break."""
    if region == "VIC1":
        return ROOT / "data" / "updata" / "vic_battery_power.csv"
    return ROOT / "data" / "updata" / f"{region.lower()}_battery_power.csv"
GEN = ROOT / "data" / "updata" / "vic_generation_20250101_20260814.parquet"

BASE_URL = "https://api.openelectricity.org.au/v4"
NETWORK = "NEM"          # WEM units need network="WEM"
CHUNK_DAYS = 30                 # PRO tier limit for the 5m interval
SOC_HISTORY_STARTS = "2025-06-30"


def api_key() -> str:
    k = os.environ.get("OPENELECTRICITY_API_KEY")
    if not k and (HERE / ".oe_key").exists():
        k = (HERE / ".oe_key").read_text().strip()
    if not k:
        sys.exit("no API key: set OPENELECTRICITY_API_KEY or write script/.oe_key")
    return k


def get(session, url, params, tries=5):
    for attempt in range(1, tries + 1):
        r = session.get(url, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        if r.status_code >= 500 and attempt < tries:
            time.sleep(2 ** attempt)
            continue
        sys.exit(f"HTTP {r.status_code}: {r.text[:300]}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="VIC1")
    ap.add_argument("--network", default=NETWORK,
                    help="NEM for NSW1/QLD1/SA1/VIC1/TAS1; WEM for WA")
    ap.add_argument("--start", default=SOC_HISTORY_STARTS)
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--interval", default="5m")
    a = ap.parse_args()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key()}",
                            "Accept": "application/json"})

    # --- 1. the same BIDIRECTIONAL unit set the SOC pull uses ---------------------
    fac = get(session, f"{BASE_URL}/facilities/", {})
    cap = {}
    for f in fac.get("data", []):
        if f.get("network_region") != a.region:
            continue
        for un in f.get("units", []):
            if un.get("fueltech_id") != "battery":       # BIDIRECTIONAL record only
                continue
            cap[un["code"]] = {"facility": f.get("name"),
                               "facility_code": f.get("code"),
                               "capacity_storage": un.get("capacity_storage"),
                               "capacity_registered": un.get("capacity_registered"),
                               "commenced": un.get("commencement_date")}
    live = {k: v for k, v in cap.items() if v["commenced"]}
    print(f"{a.region} battery units: {len(cap)} registered, {len(live)} operating")
    for k, v in sorted(live.items()):
        pw = v.get("capacity_registered")
        print(f"  {k:10s} {v['facility']:24s} {v['capacity_storage']:>8.1f} MWh  "
              f"{(f'{pw:.1f} MW' if pw else 'MW n/a'):>10s}")

    # --- 2. power time series ----------------------------------------------------
    units = sorted(live)
    rows = []
    cur, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    print()
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=CHUNK_DAYS), end)
        j = get(session, f"{BASE_URL}/data/facilities/{a.network}",
                {"metrics": "power", "interval": a.interval,
                 "facility_code": sorted({live[u]["facility_code"] for u in units}),
                 "unit_code": units,
                 "date_start": cur.strftime("%Y-%m-%dT%H:%M:%S"),
                 "date_end": nxt.strftime("%Y-%m-%dT%H:%M:%S")})
        n = 0
        for ser in j.get("data", []):
            for rr in ser.get("results", []):
                code = (rr.get("columns") or {}).get("unit_code") \
                    or (rr.get("name") or "").replace("power_", "")
                if code not in live:
                    continue
                for ts, v in rr.get("data", []):
                    if v is not None:
                        rows.append((ts, code, float(v)))
                        n += 1
        print(f"  {cur.date()}..{nxt.date()}  {n:,} points", flush=True)
        cur = nxt

    if not rows:
        sys.exit("nothing returned")

    df = pd.DataFrame(rows, columns=["interval", "unit", "mw"])
    df["interval"] = pd.DatetimeIndex(df["interval"]).tz_localize(None)
    mw = df.pivot_table(index="interval", columns="unit", values="mw",
                        aggfunc="first").sort_index()
    out = mw.add_suffix("_mw")
    out.index.name = "interval"
    OUT = out_path(a.region, a.network)
    out.to_csv(OUT)
    print(f"\nrows {len(out):,}   {out.index.min()} .. {out.index.max()}")
    print(f"units with data: {len(mw.columns)} of {len(live)}")
    print(f"wrote {OUT.relative_to(ROOT)}")

    # --- 3. checks the memo says to run on arrival -------------------------------
    print("\n" + "=" * 88)
    print("CHECKS")
    print("=" * 88)
    neg = float((mw.values[~np.isnan(mw.values)] < 0).mean())
    print(f"  signed series (negative = charging): {100 * neg:.1f}% of readings negative")
    if neg < 0.05:
        print("    WARNING: looks unsigned -- charging may be on a separate unit code")
    print(f"  per-unit coverage:")
    for u in mw.columns:
        s = mw[u].dropna()
        print(f"    {u:10s} {len(s):8,d} rows  {100 * len(s) / len(mw):5.1f}%  "
              f"[{s.min():8.1f}, {s.max():8.1f}] MW")

    # fleet reconciliation against the aggregate generation table
    if GEN.exists() and a.region == "VIC1":
        g = pd.read_parquet(GEN)
        g["interval"] = pd.DatetimeIndex(g["interval"]).tz_localize(None)
        gw = g.pivot_table(index="interval", columns="fueltech", values="power_mw",
                           aggfunc="first")
        net = (gw["battery_discharging"] - gw["battery_charging"]).rename("fleet")
        mine = mw.sum(axis=1, min_count=1).rename("sum_units")
        j = pd.concat([net, mine], axis=1).dropna()
        # only compare where every unit reports, else a dropout looks like a mismatch
        full = mw.notna().all(axis=1)
        jf = j[j.index.isin(mw.index[full])]
        print(f"\n  fleet reconciliation  (sum of units vs generation table)")
        print(f"    all rows        n={len(j):,}  mean {(j.sum_units - j.fleet).mean():+.2f} "
              f"MW  mean|d| {(j.sum_units - j.fleet).abs().mean():.2f} MW")
        if len(jf):
            d = jf.sum_units - jf.fleet
            print(f"    all units report n={len(jf):,}  mean {d.mean():+.2f} MW  "
                  f"mean|d| {d.abs().mean():.2f} MW  max|d| {d.abs().max():.1f} MW")
        by = (j.sum_units - j.fleet).abs().groupby(j.index.to_period("M")).mean()
        print("    mean|d| by month: " +
              "  ".join(f"{str(k)[2:]}:{v:.1f}" for k, v in by.items()))
        print("    (memo: matches to <1 MW from 2026-02; 2025-09..12 off by 30-53 MW"
              " mean-absolute with near-zero mean = misalignment, not a missing unit)")


if __name__ == "__main__":
    main()
