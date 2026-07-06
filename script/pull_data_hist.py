"""Historical VIC1 pull from the OpenElectricity v4 API (pro plan).

Extends script/pull_data.py (last-365d, free tier) to an arbitrary range --
default 2021-10-01 (5-minute-settlement start; modern fleet incl. batteries)
through today. See constraint_research.md for why pre-2018 data is not useful
for the 6-target model.

Key differences from the original:
  - auth: OPENELECTRICITY_API_KEY env var, else script/.oe_key file
  - per-chunk parquet cache under data/pull_cache/<region>/<label>/ so a killed
    run RESUMES instead of restarting (~220 chunks x 3 streams for 5 years)
  - retry with exponential backoff, honours 429 Retry-After
  - parquet-only outputs (a 5-year CSV would be ~250 MB for nothing)
  - --probe mode: discovers valid metrics (from the API's 422 enum error) and
    tests data availability at old dates before committing to a big pull

Outputs (suffix = <start>_<end> dates):
  data/vic_generation_<suffix>.parquet     interval, fueltech, power_mw, emissions_t
  data/vic_market_<suffix>.parquet         interval, demand_mw, price_aud_per_mwh
  data/vic_interconnector_<suffix>.parquet interval, exports_mw, imports_mw, net_import_mw
                                           (flow_imports/flow_exports metrics, market
                                           endpoint -- same schema as the bundled
                                           vic_interconnector_last365.parquet)

Usage:
  python3 script/pull_data_hist.py --probe
  python3 script/pull_data_hist.py --start 2021-10-01
  python3 script/pull_data_hist.py --start 2021-10-01 --end 2026-07-06
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent          # script/
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "pull_cache"

BASE_URL = "https://api.openelectricity.org.au/v4"
NETWORK = "NEM"
INTERVAL = "5m"
CHUNK_DAYS = 8                       # API hard limit for 5m
SLEEP_S = 0.4
MAX_TRIES = 5


def api_key() -> str:
    k = os.environ.get("OPENELECTRICITY_API_KEY")
    if not k and (HERE / ".oe_key").exists():
        k = (HERE / ".oe_key").read_text().strip()
    if not k:
        sys.exit("no API key: set OPENELECTRICITY_API_KEY or write script/.oe_key")
    return k


def get(session, endpoint, params):
    """GET with retry/backoff; returns (status_code, json_or_text)."""
    url = f"{BASE_URL}/{endpoint}/network/{NETWORK}"
    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = session.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            if attempt == MAX_TRIES:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return 200, r.json()
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2 ** attempt))
            print(f"    429 rate-limited, sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500 and attempt < MAX_TRIES:
            time.sleep(2 ** attempt)
            continue
        return r.status_code, r.text
    return r.status_code, r.text


def params_for(metrics, region, s, e, secondary=None):
    p = {"metrics": metrics, "interval": INTERVAL,
         "date_start": s.strftime("%Y-%m-%dT%H:%M:%S"),
         "date_end": e.strftime("%Y-%m-%dT%H:%M:%S"),
         "network_region": region, "primary_grouping": "network_region"}
    if secondary:
        p["secondary_grouping"] = secondary
    return p


def flatten(payload, has_fueltech):
    rows = []
    for series in payload.get("data", []):
        metric, unit = series.get("metric"), series.get("unit")
        for result in series.get("results", []):
            cols = result.get("columns", {}) or {}
            ft = cols.get("fueltech")
            for point in result.get("data", []):
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    ts, value = point[0], point[1]
                else:
                    ts = point.get("interval") or point.get("date")
                    value = point.get("value")
                rows.append({"interval": ts, "metric": metric, "unit": unit,
                             "fueltech": ft if has_fueltech else None, "value": value})
    return rows


def chunk_ranges(start, end, days=CHUNK_DAYS):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def pull_stream(session, label, endpoint, metrics, secondary, region, start, end):
    """Chunked pull with per-chunk parquet cache; returns the assembled long df."""
    cache = CACHE_DIR / region / label
    cache.mkdir(parents=True, exist_ok=True)
    chunks = list(chunk_ranges(start, end))
    frames, pulled = [], 0
    for i, (s, e) in enumerate(chunks, 1):
        f = cache / f"{s:%Y%m%d}_{e:%Y%m%d}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            continue
        code, payload = get(session, endpoint, params_for(metrics, region, s, e, secondary))
        if code != 200:
            print(f"  [{label}] chunk {i}/{len(chunks)} {s.date()} HTTP {code}: "
                  f"{str(payload)[:200]}", flush=True)
            continue
        df = pd.DataFrame(flatten(payload, secondary is not None))
        df.to_parquet(f, index=False)
        frames.append(df)
        pulled += 1
        if pulled % 10 == 0 or i == len(chunks):
            print(f"  [{label}] {i}/{len(chunks)} chunks ({s.date()} -> {e.date()})", flush=True)
        time.sleep(SLEEP_S)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).dropna(subset=["interval"])
    out["interval"] = pd.to_datetime(out["interval"])
    out = out.drop_duplicates(subset=["interval", "metric", "fueltech"])
    return out


def probe(session, region):
    """Discover valid metrics (via 422 enum error) + data availability at old dates."""
    day = timedelta(days=1)
    print("== metric discovery (from validation errors) ==")
    for endpoint in ("data", "market"):
        s = datetime(2026, 6, 1)
        code, payload = get(session, endpoint,
                            params_for(["bogus_metric"], region, s, s + day))
        print(f"[{endpoint}] HTTP {code}: {str(payload)[:600]}\n")

    print("== availability probe (market demand, 1 day each) ==")
    for d in ("1999-01-01", "2005-01-01", "2010-01-01", "2015-01-01",
              "2018-01-01", "2021-10-01", "2024-01-01"):
        s = datetime.fromisoformat(d)
        code, payload = get(session, "market", params_for(["demand"], region, s, s + day))
        n = len(flatten(payload, False)) if code == 200 else 0
        print(f"  {d}: HTTP {code}, {n} rows")
        time.sleep(SLEEP_S)

    print("\n== fueltech coverage (gen power, 1 day each) ==")
    for d in ("2015-01-01", "2018-01-01", "2021-10-01", "2024-01-01"):
        s = datetime.fromisoformat(d)
        code, payload = get(session, "data",
                            params_for(["power"], region, s, s + day, "fueltech"))
        if code == 200:
            fts = sorted({r["fueltech"] for r in flatten(payload, True) if r["value"] is not None})
            print(f"  {d}: {fts}")
        else:
            print(f"  {d}: HTTP {code}")
        time.sleep(SLEEP_S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default=None, help="exclusive; default today 00:00")
    ap.add_argument("--region", default="VIC1")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key()}",
                            "Accept": "application/json"})

    if args.probe:
        probe(session, args.region)
        return

    start = datetime.fromisoformat(args.start)
    end = (datetime.fromisoformat(args.end) if args.end
           else datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
    suffix = f"{start:%Y%m%d}_{end:%Y%m%d}"
    print(f"Pulling {args.region} {start.date()} -> {end.date()} at {INTERVAL}", flush=True)

    gen = pull_stream(session, "gen", "data", ["power", "emissions"], "fueltech",
                      args.region, start, end)
    if not gen.empty:
        gw = (gen.pivot_table(index=["interval", "fueltech"], columns="metric",
                              values="value", aggfunc="first")
                 .reset_index()
                 .rename(columns={"power": "power_mw", "emissions": "emissions_t"}))
        p = DATA_DIR / f"vic_generation_{suffix}.parquet"
        gw.to_parquet(p, index=False)
        print(f"  -> {p.name}: {len(gw):,} rows, "
              f"{gw['interval'].min()} .. {gw['interval'].max()}", flush=True)

    mkt = pull_stream(session, "market", "market", ["demand", "price"], None,
                      args.region, start, end)
    if not mkt.empty:
        mw = (mkt.pivot_table(index="interval", columns="metric",
                              values="value", aggfunc="first")
                 .reset_index()
                 .rename(columns={"demand": "demand_mw", "price": "price_aud_per_mwh"}))
        p = DATA_DIR / f"vic_market_{suffix}.parquet"
        mw.to_parquet(p, index=False)
        print(f"  -> {p.name}: {len(mw):,} rows, "
              f"{mw['interval'].min()} .. {mw['interval'].max()}", flush=True)

    flows = pull_stream(session, "flows", "market", ["flow_imports", "flow_exports"],
                        None, args.region, start, end)
    if not flows.empty:
        fw = (flows.pivot_table(index="interval", columns="metric",
                                values="value", aggfunc="first")
                   .reset_index()
                   .rename(columns={"flow_exports": "exports_mw",
                                    "flow_imports": "imports_mw"}))
        fw["net_import_mw"] = fw["imports_mw"] - fw["exports_mw"]
        p = DATA_DIR / f"vic_interconnector_{suffix}.parquet"
        fw.to_parquet(p, index=False)
        print(f"  -> {p.name}: {len(fw):,} rows, "
              f"{fw['interval'].min()} .. {fw['interval'].max()}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
