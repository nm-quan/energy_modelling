"""Pull battery state of charge for VIC1 from OpenElectricity.

Does exactly what the OpenElectricity battery guide describes:

    "Open Electricity stores the current state of charge for a battery as the total
     MWh of energy stored in the battery. This is stored against the bidirectional
     battery unit for each facility... Combined with the capacity_storage field from
     facility data, this can be used to calculate the state of charge as a percentage
     of the total storage capacity."

Two API calls:
  1. GET /v4/facilities/            -> capacity_storage per unit (MWh)
  2. GET /v4/data/facilities/NEM    -> metrics=storage_battery (MWh, 5-minute)

Filtered by unit_code so only the BIDIRECTIONAL units come back. Filtering by
facility_code instead returns each battery 2-3 times (BALB1 / BALBG1 / BALBL1 are the
same asset seen as bidirectional / generator / load) and would triple-count.

Output: data/updata/vic_battery_soc.csv -- interval, then per unit the MWh stored and
the same as a percentage of that unit's capacity_storage.

Note: SOC history begins 2025-06-30. Earlier dates return HTTP 200 with all values
null, so a pull that looks successful can contain nothing.

Usage:
    OPENELECTRICITY_API_KEY=... python script/pull_battery_soc.py
    python script/pull_battery_soc.py --start 2025-06-30 --end 2026-08-14 --region VIC1
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
def out_path(region: str, network: str = "NEM"):
    """Region-named output. vic_battery_soc.csv is kept for VIC1/NEM so
    existing analyses do not break."""
    if region == "VIC1":
        return ROOT / "data" / "updata" / "vic_battery_soc.csv"
    return ROOT / "data" / "updata" / f"{region.lower()}_battery_soc.csv"

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
    ap.add_argument("--interval", default="5m",
                    choices=["5m", "1h", "1d", "7d", "1M", "3M", "season", "1y", "fy"])
    a = ap.parse_args()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key()}",
                            "Accept": "application/json"})

    # --- 1. capacity_storage per bidirectional battery unit ----------------------
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
                               "commenced": un.get("commencement_date")}
    live = {k: v for k, v in cap.items() if v["commenced"]}
    print(f"{a.region} battery units: {len(cap)} registered, {len(live)} with a "
          f"commencement date (the rest are committed, not operating)")
    for k, v in sorted(live.items()):
        print(f"  {k:10s} {v['facility']:24s} {v['capacity_storage']:>8.1f} MWh")

    # --- 2. storage_battery time series -----------------------------------------
    units = sorted(live)
    rows = []
    cur, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    print()
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=CHUNK_DAYS), end)
        j = get(session, f"{BASE_URL}/data/facilities/{a.network}",
                {"metrics": "storage_battery", "interval": a.interval,
                 "facility_code": sorted({live[u]["facility_code"] for u in units}),
                 "unit_code": units,
                 "date_start": cur.strftime("%Y-%m-%dT%H:%M:%S"),
                 "date_end": nxt.strftime("%Y-%m-%dT%H:%M:%S")})
        n = 0
        for ser in j.get("data", []):
            for rr in ser.get("results", []):
                code = (rr.get("columns") or {}).get("unit_code") \
                    or (rr.get("name") or "").replace("storage_battery_", "")
                if code not in live:
                    continue
                for ts, v in rr.get("data", []):
                    if v is not None:
                        rows.append((ts, code, float(v)))
                        n += 1
        print(f"  {cur.date()}..{nxt.date()}  {n:,} points", flush=True)
        cur = nxt

    if not rows:
        sys.exit(f"nothing returned -- SOC history starts {SOC_HISTORY_STARTS}")

    df = pd.DataFrame(rows, columns=["interval", "unit", "mwh"])
    df["interval"] = pd.DatetimeIndex(df["interval"]).tz_localize(None)
    mwh = df.pivot_table(index="interval", columns="unit", values="mwh",
                         aggfunc="first").sort_index()

    # --- 3. percentage of capacity_storage, per the guide -----------------------
    pct = pd.DataFrame({c: mwh[c] / live[c]["capacity_storage"] * 100
                        for c in mwh.columns}, index=mwh.index)
    out = pd.concat([mwh.add_suffix("_mwh"), pct.add_suffix("_pct")], axis=1)
    out = out[[f"{c}{s}" for c in mwh.columns for s in ("_mwh", "_pct")]]
    out.index.name = "interval"
    OUT = out_path(a.region, a.network)
    out.to_csv(OUT)

    print(f"\nrows {len(out):,}   {out.index.min()} .. {out.index.max()}")
    print(f"units with data: {len(mwh.columns)} of {len(live)}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
