"""Pull per-unit POWER for EVERY fueltech in a region, not just batteries.

Why: the fueltech series in vic_generation_*.parquet is NOT read off the units. It is a
separate ClickHouse materialized view (`fueltech_intervals_mv`) that aggregates
`unit_intervals` grouped by fueltech, and OpenNEM's own changelog records it diverging
from the unit sum three times in 2026 (v4.5.0 duplicate-row double-counting, v4.5.4
"network fueltech series could report less energy than the sum of the individual units,
most visibly for battery discharge" over 18 months / 41 GWh, v4.5.7 battery discharge
wrongly in renewable+fossil totals).

Measured here on VIC1 batteries: the fueltech series equals the unit sum on only 136 of
410 days; on the rest it is higher by exactly one or two units' output.

So this pulls the units directly, which lets every fueltech channel be rebuilt from the
primary records and the rollup bypassed entirely.

Battery handling follows the same trap as pull_battery_power.py: a battery appears as
three unit records (BIDIRECTIONAL + GENERATOR + LOAD) for one physical asset. Only the
BIDIRECTIONAL record is pulled, and it is signed (negative = charging), so
battery_charging = -min(p,0) and battery_discharging = max(p,0).

Output: data/updata/<region>_units_power.parquet  (long: interval, unit, fueltech, mw)
plus  <region>_fueltech_from_units.parquet  (wide: interval x fueltech, MW)

Usage:
    OPENELECTRICITY_API_KEY=... python script/pull_units_power.py --region VIC1
    python script/pull_units_power.py --region NSW1 --start 2025-01-01 --end 2026-08-14
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTDIR = ROOT / "data" / "updata"

BASE_URL = "https://api.openelectricity.org.au/v4"
CHUNK_DAYS = 30
MAX_FACILITIES = 30            # API hard-caps facility_code at 30 items per request
MAX_UNITS = 30                 # ... and unit_code at 30 as well

# a battery is one asset with three unit records; keep only the bidirectional one
SKIP_FUELTECH = {"battery_charging", "battery_discharging"}


def api_key() -> str:
    k = os.environ.get("OPENELECTRICITY_API_KEY")
    if not k and (HERE / ".oe_key").exists():
        k = (HERE / ".oe_key").read_text().strip()
    if not k:
        sys.exit("no API key: set OPENELECTRICITY_API_KEY or write script/.oe_key")
    return k


def get(session, url, params, tries=5):
    for attempt in range(1, tries + 1):
        r = session.get(url, params=params, timeout=180)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        if r.status_code >= 500 and attempt < tries:
            time.sleep(2 ** attempt)
            continue
        print(f"    HTTP {r.status_code}: {r.text[:200]}", flush=True)
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="VIC1")
    ap.add_argument("--network", default="NEM")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--interval", default="5m")
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key()}",
                            "Accept": "application/json"})

    fac = get(session, f"{BASE_URL}/facilities/", {})
    meta, byfac = {}, {}
    for f in fac.get("data", []):
        if f.get("network_region") != a.region:
            continue
        for un in f.get("units", []):
            ft = str(un.get("fueltech_id"))
            if ft in SKIP_FUELTECH or ft == "None":
                continue
            if not un.get("commencement_date"):
                continue                       # committed, not operating
            meta[un["code"]] = {"fueltech": ft, "facility_code": f.get("code"),
                                "facility": f.get("name"),
                                "capacity": un.get("capacity_registered")}
            byfac.setdefault(f.get("code"), []).append(un["code"])

    n_ft = pd.Series([m["fueltech"] for m in meta.values()]).value_counts()
    print(f"{a.region}: {len(meta)} operating unit records across {len(byfac)} facilities")
    print(n_ft.to_string())

    # batch so that BOTH facility_code and unit_code stay within their 30-item caps
    batches, cf, cu = [], [], []
    for c in sorted(byfac):
        us = byfac[c]
        if cf and (len(cf) + 1 > MAX_FACILITIES or len(cu) + len(us) > MAX_UNITS):
            batches.append((cf, cu))
            cf, cu = [], []
        cf.append(c)
        cu.extend(us)
    if cf:
        batches.append((cf, cu))
    assert all(len(f) <= MAX_FACILITIES and len(u) <= MAX_UNITS for f, u in batches)
    print(f"{len(batches)} request batches "
          f"(max {max(len(f) for f, _ in batches)} facilities, "
          f"{max(len(u) for _, u in batches)} units)")
    rows = []
    cur, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    t0 = time.time()
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=CHUNK_DAYS), end)
        n = 0
        for fb, ub in batches:
            j = get(session, f"{BASE_URL}/data/facilities/{a.network}",
                    {"metrics": "power", "interval": a.interval,
                     "facility_code": sorted(fb), "unit_code": sorted(ub),
                     "date_start": cur.strftime("%Y-%m-%dT%H:%M:%S"),
                     "date_end": nxt.strftime("%Y-%m-%dT%H:%M:%S")})
            if j is None:
                continue
            for ser in j.get("data", []):
                for rr in ser.get("results", []):
                    code = (rr.get("columns") or {}).get("unit_code") \
                        or (rr.get("name") or "").replace("power_", "")
                    if code not in meta:
                        continue
                    for ts, v in rr.get("data", []):
                        if v is not None:
                            rows.append((ts, code, float(v)))
                            n += 1
        print(f"  {cur.date()}..{nxt.date()}  {n:,} points  ({time.time()-t0:.0f}s)",
              flush=True)
        cur = nxt

    if not rows:
        sys.exit("nothing returned")

    df = pd.DataFrame(rows, columns=["interval", "unit", "mw"])
    df["interval"] = pd.DatetimeIndex(df["interval"]).tz_localize(None)
    df["fueltech"] = df["unit"].map(lambda u: meta[u]["fueltech"])
    long_p = OUTDIR / f"{a.region.lower()}_units_power.parquet"
    df.to_parquet(long_p, index=False)
    print(f"\nwrote {long_p.relative_to(ROOT)}  {df.shape}")

    # rebuild the fueltech mix FROM the units, splitting the signed battery series
    bat = df[df.fueltech == "battery"]
    oth = df[df.fueltech != "battery"]
    parts = [oth.pivot_table(index="interval", columns="fueltech", values="mw",
                             aggfunc="sum")]
    if len(bat):
        chg = (-bat.mw.clip(upper=0)).groupby(bat.interval).sum().rename("battery_charging")
        dis = bat.mw.clip(lower=0).groupby(bat.interval).sum().rename("battery_discharging")
        parts += [chg.to_frame(), dis.to_frame()]
    wide = pd.concat(parts, axis=1).sort_index()
    wide.index.name = "interval"
    wide_p = OUTDIR / f"{a.region.lower()}_fueltech_from_units.parquet"
    wide.to_parquet(wide_p)
    print(f"wrote {wide_p.relative_to(ROOT)}  {wide.shape}")
    print(f"\nunits reporting per fueltech (median over intervals):")
    cnt = df.groupby(["interval", "fueltech"]).size().unstack(fill_value=0)
    print(cnt.median().astype(int).to_string())
    print(f"\nfueltech means (MW) rebuilt from units:")
    print(wide.mean().round(1).to_string())


if __name__ == "__main__":
    main()
