"""Pull VIC1 wind + solar-utility curtailment over the full hist window.

Curtailment lives on the /market endpoint (not /data) under metrics
`curtailment_wind` and `curtailment_solar_utility` (MW). The pro key removes the
365-day free-tier cap, so this pulls the whole 2021-10-01 → today range in 8-day
chunks with a per-chunk parquet cache (resumes if killed).

Key: OPENELECTRICITY_API_KEY env var, else script/.oe_key (never hardcode).

    OPENELECTRICITY_API_KEY=oe_xxx python3 script/pull_curtailment_hist.py
    ... --test        # one 8-day chunk, sanity check
"""
from __future__ import annotations
import argparse, os, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CACHE = DATA / "pull_cache" / "curtailment_hist"
BASE = "https://api.openelectricity.org.au/v4"
REGION, INTERVAL, CHUNK_DAYS, SLEEP = "VIC1", "5m", 8, 0.4
METRICS = ["curtailment_wind", "curtailment_solar_utility"]
COLMAP = {"curtailment_wind": "curtailment_wind_mw",
          "curtailment_solar_utility": "curtailment_solar_mw"}


def api_key() -> str:
    k = os.environ.get("OPENELECTRICITY_API_KEY")
    if not k and (HERE / ".oe_key").exists():
        k = (HERE / ".oe_key").read_text().strip()
    if not k:
        sys.exit("no key: set OPENELECTRICITY_API_KEY or write script/.oe_key")
    return k


def fetch_chunk(session, s, e):
    params = {"metrics": METRICS, "interval": INTERVAL,
              "date_start": s.strftime("%Y-%m-%dT%H:%M:%S"),
              "date_end": e.strftime("%Y-%m-%dT%H:%M:%S"),
              "network_region": REGION, "primary_grouping": "network_region"}
    for attempt in range(5):
        r = session.get(f"{BASE}/market/network/NEM", params=params, timeout=60)
        if r.status_code == 200:
            break
        wait = int(r.headers.get("Retry-After", 2 ** attempt))
        print(f"    HTTP {r.status_code}, retry in {wait}s", file=sys.stderr)
        time.sleep(wait)
    else:
        raise RuntimeError(f"failed chunk {s}..{e}: {r.status_code} {r.text[:200]}")
    rows = []
    for series in r.json().get("data", []):
        col = COLMAP.get(series.get("metric"))
        for res in series.get("results", []):
            for p in res.get("data", []):
                rows.append({"interval": p[0], col: p[1]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default=None, help="default: today")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    start = datetime.fromisoformat(args.start)
    end = (start + timedelta(days=CHUNK_DAYS) if args.test
           else datetime.fromisoformat(args.end) if args.end
           else datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
    CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key()}", "Accept": "application/json"})

    chunks, cur = [], start
    while cur < end:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), end)
        chunks.append((cur, nxt)); cur = nxt
    print(f"curtailment pull {start.date()}→{end.date()}  {len(chunks)} chunks")
    parts = []
    for i, (s, e) in enumerate(chunks, 1):
        cf = CACHE / f"{s.date()}.parquet"
        if cf.exists():
            parts.append(pd.read_parquet(cf)); continue
        df = fetch_chunk(session, s, e)
        if not df.empty:
            df = df.groupby("interval", as_index=False).first()  # merge the 2 metric rows per ts
            df.to_parquet(cf, index=False)
            parts.append(df)
        print(f"  {i:>3}/{len(chunks)} {s.date()}→{e.date()} (+{len(df)} rows)", flush=True)
        time.sleep(SLEEP)

    out = pd.concat(parts, ignore_index=True)
    out["interval"] = pd.to_datetime(out["interval"])
    out = out.sort_values("interval").drop_duplicates("interval").reset_index(drop=True)
    for c in ["curtailment_wind_mw", "curtailment_solar_mw"]:
        if c not in out:
            out[c] = 0.0
    out["curtailment_mw"] = out["curtailment_wind_mw"].fillna(0) + out["curtailment_solar_mw"].fillna(0)
    dst = DATA / ("vic_curtailment_hist_test.parquet" if args.test else "vic_curtailment_hist.parquet")
    out.to_parquet(dst, index=False)
    print(f"\nwrote {dst}  rows={len(out):,}  {out['interval'].min()} → {out['interval'].max()}")
    print(out[["curtailment_wind_mw", "curtailment_solar_mw"]].describe().loc[["min", "max", "mean"]].round(1).to_string())


if __name__ == "__main__":
    main()
