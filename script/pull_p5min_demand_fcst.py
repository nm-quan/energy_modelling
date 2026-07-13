"""Pull AEMO's own one-step-ahead demand forecast from NEMWEB P5MIN reports.

OpenElectricity's API only exposes actuals (see script/pull_data.py /
pull_data_hist.py) -- there is no forecast/predispatch metric on its v4 API
(confirmed via the metrics enum + full openapi.json). AEMO's real pre-dispatch
demand forecast lives on NEMWEB instead:

  http://nemweb.com.au/Reports/Archive/P5_Reports/PUBLIC_P5MIN_<YYYYMMDD>.zip

Each daily zip holds 288 nested per-5-min-run zips (one per P5MIN dispatch
run). Each run at RUN_DATETIME=t publishes a P5MIN_REGIONSOLUTION row for
INTERVAL_DATETIME=t (a same-interval nowcast) plus 11 more rows out to t+55min
in 5-min steps. We keep only the INTERVAL_DATETIME=t+5min row per run -- AEMO's
genuine one-step-ahead TOTALDEMAND forecast, known at time t, for the exact
target our LSTM predicts (t -> t+1). This mirrors the model's own 1-step task
so it's a fair "does forward information help" experiment.

Archive retention is ~13 months (2025-07-03 onward as of 2026-07) -- this is
NOT the 5-year "hist" window, hence the separate short-window dataset in
future_demand/.

Per-day extracted parquet is cached under data/pull_cache/p5min/<region>/ (the
~57MB daily zip itself is NOT kept -- only the ~288-row extract). A killed run
resumes: already-cached days are skipped.

Usage:
  python3 script/pull_p5min_demand_fcst.py --start 2026-05-29 --end 2026-07-12
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "pull_cache" / "p5min"

ARCHIVE_URL = "http://nemweb.com.au/Reports/Archive/P5_Reports/PUBLIC_P5MIN_{day}.zip"
NEM_TZ = pytz.FixedOffset(600)          # AEST, fixed +10:00 (matches the OE pulls)
MAX_TRIES = 5


def _get(url: str) -> bytes:
    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = requests.get(url, timeout=180)
        except requests.RequestException as e:
            if attempt == MAX_TRIES:
                raise
            print(f"    network error ({e}), retry {attempt}/{MAX_TRIES}", flush=True)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.content
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2 ** attempt))
            print(f"    429 rate-limited, sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500 and attempt < MAX_TRIES:
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    raise RuntimeError(f"exhausted retries for {url}")


def _parse_region_solution(csv_text: str, region: str) -> tuple[str, float] | None:
    """Return (run_datetime_str, totaldemand) for the t+5min row, or None."""
    lines = csv_text.splitlines()
    cols = None
    run_idx = interval_idx = region_idx = demand_idx = None
    for line in lines:
        if line.startswith("I,P5MIN,REGIONSOLUTION"):
            cols = line.split(",")
            run_idx = cols.index("RUN_DATETIME")
            interval_idx = cols.index("INTERVAL_DATETIME")
            region_idx = cols.index("REGIONID")
            demand_idx = cols.index("TOTALDEMAND")
            continue
        if cols is None or not line.startswith("D,P5MIN,REGIONSOLUTION"):
            continue
        parts = line.split(",")
        if parts[region_idx] != region:
            continue
        run_dt = parts[run_idx].strip('"')
        interval_dt = parts[interval_idx].strip('"')
        run_ts = datetime.strptime(run_dt, "%Y/%m/%d %H:%M:%S")
        interval_ts = datetime.strptime(interval_dt, "%Y/%m/%d %H:%M:%S")
        if interval_ts - run_ts == timedelta(minutes=5):
            return run_dt, float(parts[demand_idx])
    return None


def pull_day(day: datetime, region: str) -> pd.DataFrame:
    """Download one day's P5MIN zip, extract the region's 1-step demand
    forecast from every nested run, return a tidy (interval, demand_fcst_mw)
    frame. Raises on HTTP failure (caller decides whether to skip)."""
    url = ARCHIVE_URL.format(day=day.strftime("%Y%m%d"))
    raw = _get(url)
    outer = zipfile.ZipFile(io.BytesIO(raw))
    rows = []
    for name in outer.namelist():
        inner_bytes = outer.read(name)
        inner = zipfile.ZipFile(io.BytesIO(inner_bytes))
        csv_name = inner.namelist()[0]
        text = inner.read(csv_name).decode("utf-8", errors="replace")
        result = _parse_region_solution(text, region)
        if result is not None:
            rows.append(result)
    if not rows:
        return pd.DataFrame(columns=["interval", "demand_fcst_mw"])
    df = pd.DataFrame(rows, columns=["run_datetime", "demand_fcst_mw"])
    df["interval"] = (
        pd.to_datetime(df["run_datetime"], format="%Y/%m/%d %H:%M:%S")
        .dt.tz_localize(NEM_TZ)
    )
    df = df.drop(columns=["run_datetime"]).sort_values("interval")
    df = df.drop_duplicates(subset="interval")
    return df.reset_index(drop=True)


def daterange(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--region", default="VIC1")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    cache = CACHE_DIR / args.region
    cache.mkdir(parents=True, exist_ok=True)

    days = list(daterange(start, end))
    print(f"Pulling P5MIN 1-step demand forecast for {args.region}: "
          f"{start.date()} -> {end.date()} ({len(days)} days)", flush=True)

    frames = []
    for i, day in enumerate(days, 1):
        f = cache / f"{day:%Y%m%d}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            print(f"  [{i}/{len(days)}] {day.date()} (cached)", flush=True)
            continue
        try:
            df = pull_day(day, args.region)
        except Exception as err:
            print(f"  [{i}/{len(days)}] {day.date()} FAILED: {err}", file=sys.stderr, flush=True)
            continue
        df.to_parquet(f, index=False)
        frames.append(df)
        print(f"  [{i}/{len(days)}] {day.date()}  (+{len(df)} rows)", flush=True)
        time.sleep(0.2)

    if not frames:
        sys.exit("no data pulled")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset="interval").sort_values("interval")
    suffix = f"{start:%Y%m%d}_{end:%Y%m%d}"
    out_path = DATA_DIR / f"vic_p5min_demand_fcst_{suffix}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"-> {out_path.name}: {len(out):,} rows, "
          f"{out['interval'].min()} .. {out['interval'].max()}")


if __name__ == "__main__":
    main()
