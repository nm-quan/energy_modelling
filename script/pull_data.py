from __future__ import annotations
"""
Pull Victorian (VIC1) electricity data from Open Electricity API for 2025.

Uses the REST API directly via `requests` — no openelectricity SDK needed
(SDK requires Python 3.12+; this works on 3.8+).

API constraints (from docs.openelectricity.org.au):
  - Base URL: https://api.openelectricity.org.au/v4
  - Auth: Bearer token via OPENELECTRICITY_API_KEY env var
  - 5-min interval: max 8 days per request → ~46 chunks for full year

Outputs in ./data/:
  - vic_generation_2025.{parquet,csv}: interval, fueltech, power_mw, emissions_t
  - vic_market_2025.{parquet,csv}:     interval, demand_mw, price_aud_per_mwh

Setup:
  pip install requests pandas pyarrow
  export OPENELECTRICITY_API_KEY=oe_xxx

Usage:
  python pull_vic_2025.py --test      # 1 chunk (~8 days), sanity check
  python pull_vic_2025.py             # full year
  python pull_vic_2025.py --debug     # print raw response from first chunk
  api_key = "oe_Hpa4Kw55aDCnsPupnSEpa9"
"""

"""
Pull Victorian (VIC1) electricity data from Open Electricity API for 2025.

Uses the REST API directly via `requests` — no openelectricity SDK needed
(SDK requires Python 3.12+; this works on 3.8+).

API constraints (from docs.openelectricity.org.au):
  - Base URL: https://api.openelectricity.org.au/v4
  - Auth: Bearer token via OPENELECTRICITY_API_KEY env var
  - 5-min interval: max 8 days per request → ~46 chunks for full year

Outputs in ./data/:
  - vic_generation_2025.{parquet,csv}: interval, fueltech, power_mw, emissions_t
  - vic_market_2025.{parquet,csv}:     interval, demand_mw, price_aud_per_mwh

Setup:
  pip install requests pandas pyarrow
  export OPENELECTRICITY_API_KEY=oe_xxx

Usage:
  python pull_vic_2025.py --test      # 1 chunk (~8 days), sanity check
  python pull_vic_2025.py             # full year
  python pull_vic_2025.py --debug     # print raw response from first chunk
"""


import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_URL = "https://api.openelectricity.org.au/v4"
NETWORK = "NEM"
REGION = "VIC1"
INTERVAL = "5m"
CHUNK_DAYS = 8                       # API hard limit for 5m

# Free tier allows data from the last 367 days only.
# Use last 365 days from today, with a 2-day safety buffer.
_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
START = _today - timedelta(days=365)
END = _today                          # exclusive
OUTPUT_DIR = Path("./data")
SLEEP_BETWEEN_CALLS_S = 0.5


# -----------------------------------------------------------------------------
# HTTP
# -----------------------------------------------------------------------------
def fetch_network_data(session, metrics, date_start, date_end,
                       secondary_grouping=None, endpoint="data"):
    """
    One call to OpenElectricity. Returns parsed JSON.

    endpoint: "data" for generation/emissions, "market" for demand/price.
    """
    params = {
        "metrics": metrics,                           # requests handles array params
        "interval": INTERVAL,
        "date_start": date_start.strftime("%Y-%m-%dT%H:%M:%S"),
        "date_end": date_end.strftime("%Y-%m-%dT%H:%M:%S"),
        "network_region": REGION,
        "primary_grouping": "network_region",
    }
    if secondary_grouping:
        params["secondary_grouping"] = secondary_grouping

    url = f"{BASE_URL}/{endpoint}/network/{NETWORK}"
    r = session.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {r.url}\n{r.text[:500]}")
    return r.json()


# -----------------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------------
def flatten(payload, has_fueltech):
    """
    Flatten the API response into row dicts.

    Expected response shape:
      payload["data"] : list of TimeSeries
        ts["metric"]  : "power" | "demand" | "price" | "emissions"
        ts["unit"]    : "MW" / "AUD/MWh" / "tCO2e"
        ts["results"] : list of result objects
          result["columns"] : dict (may contain "network_region", "fueltech")
          result["data"]    : list of [timestamp_str, value]
    """
    rows = []
    for series in payload.get("data", []):
        metric = series.get("metric")
        unit = series.get("unit")
        for result in series.get("results", []):
            cols = result.get("columns", {}) or {}
            fueltech = cols.get("fueltech")
            data_points = result.get("data", [])
            for point in data_points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    ts, value = point[0], point[1]
                else:
                    ts = point.get("interval") or point.get("date")
                    value = point.get("value")
                rows.append({
                    "interval": ts,
                    "metric": metric,
                    "unit": unit,
                    "fueltech": fueltech if has_fueltech else None,
                    "value": value,
                })
    return rows


# -----------------------------------------------------------------------------
# Chunked driver
# -----------------------------------------------------------------------------
def chunk_ranges(start, end, days):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def fetch_chunked(session, metrics, secondary_grouping, label, start, end,
                  endpoint="data", debug=False):
    all_rows = []
    chunks = list(chunk_ranges(start, end, CHUNK_DAYS))
    has_fueltech = secondary_grouping is not None

    for i, (s, e) in enumerate(chunks, 1):
        try:
            payload = fetch_network_data(session, metrics, s, e,
                                         secondary_grouping, endpoint)
        except Exception as err:
            print(f"  [{label}] chunk {i}/{len(chunks)} FAILED: {err}", file=sys.stderr)
            time.sleep(5)
            payload = fetch_network_data(session, metrics, s, e,
                                         secondary_grouping, endpoint)

        if debug and i == 1:
            print(f"\n--- raw response, first chunk ({label}) ---")
            print(json.dumps(payload, indent=2)[:3000])
            print("--- end raw ---\n")

        rows = flatten(payload, has_fueltech)
        all_rows.extend(rows)
        print(f"  [{label}] {i:>2}/{len(chunks)}  {s.date()} -> {e.date()}  "
              f"(+{len(rows)} rows, total {len(all_rows)})")
        time.sleep(SLEEP_BETWEEN_CALLS_S)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["interval"] = pd.to_datetime(df["interval"])
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="Pull only one 8-day chunk")
    ap.add_argument("--debug", action="store_true",
                    help="Print raw JSON of first chunk")
    args = ap.parse_args()

    api_key = "oe_Hpa4Kw55aDCnsPupnSEpa9"

    end = START + timedelta(days=CHUNK_DAYS) if args.test else END
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    suffix = "test" if args.test else "last365"

    print(f"Pulling VIC1 data: {START.date()} -> {end.date()} at 5-min resolution")
    print(f"Output: {OUTPUT_DIR.resolve()}\n")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    })

    # --- 1. Generation by fueltech + emissions ---
    print("Pull 1: power + emissions, grouped by fueltech")
    gen_long = fetch_chunked(
        session,
        metrics=["power", "emissions"],
        secondary_grouping="fueltech",
        label="gen", start=START, end=end, debug=args.debug,
    )

    if gen_long.empty:
        print("  WARNING: empty generation pull")
    else:
        gen_wide = (
            gen_long
            .pivot_table(index=["interval", "fueltech"],
                         columns="metric", values="value", aggfunc="first")
            .reset_index()
            .rename(columns={"power": "power_mw", "emissions": "emissions_t"})
        )
        gen_wide.to_parquet(OUTPUT_DIR / f"vic_generation_{suffix}.parquet", index=False)
        gen_wide.to_csv(OUTPUT_DIR / f"vic_generation_{suffix}.csv", index=False)
        print(f"  -> saved {len(gen_wide):,} rows to vic_generation_{suffix}.{{parquet,csv}}\n")

    # --- 2. Demand + price (market endpoint, no fueltech) ---
    print("Pull 2: demand + price (market endpoint)")
    mkt_long = fetch_chunked(
        session,
        metrics=["demand", "price"],
        secondary_grouping=None,
        label="market", start=START, end=end,
        endpoint="market", debug=args.debug,
    )

    if mkt_long.empty:
        print("  WARNING: empty market pull")
    else:
        mkt_wide = (
            mkt_long
            .pivot_table(index="interval", columns="metric",
                         values="value", aggfunc="first")
            .reset_index()
            .rename(columns={"demand": "demand_mw",
                             "price": "price_aud_per_mwh"})
        )
        mkt_wide.to_parquet(OUTPUT_DIR / f"vic_market_{suffix}.parquet", index=False)
        mkt_wide.to_csv(OUTPUT_DIR / f"vic_market_{suffix}.csv", index=False)
        print(f"  -> saved {len(mkt_wide):,} rows to vic_market_{suffix}.{{parquet,csv}}\n")

    print("Done.")


if __name__ == "__main__":
    main()