"""Pull NSW1 generation + market (demand, price) from OpenElectricity.

Thin wrapper over pull_data_hist.py so the region is right in the FILENAME too --
that script hardcodes a `vic_` prefix, which would silently mislabel NSW data.
Everything else (chunking, per-chunk parquet cache, retry/backoff) is reused.

    OPENELECTRICITY_API_KEY=oe_xxx python3 script/pull_nsw.py \
        --start 2026-05-01 --end 2026-08-01
"""
from __future__ import annotations
import argparse, sys
from datetime import datetime
from pathlib import Path
import pandas as pd, requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pull_data_hist import api_key, pull_stream, DATA_DIR          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-08-01")
    ap.add_argument("--region", default="NSW1")
    a = ap.parse_args()
    s, e = datetime.fromisoformat(a.start), datetime.fromisoformat(a.end)
    pre = a.region[:3].lower()
    suffix = f"{s:%Y%m%d}_{e:%Y%m%d}"
    ses = requests.Session()
    ses.headers.update({"Authorization": f"Bearer {api_key()}",
                        "Accept": "application/json"})
    print(f"{a.region} {s.date()} -> {e.date()} at 5m", flush=True)

    gen = pull_stream(ses, "gen", "data", ["power"], "fueltech", a.region, s, e)
    if not gen.empty:
        gw = (gen.pivot_table(index=["interval", "fueltech"], columns="metric",
                              values="value", aggfunc="first").reset_index()
                 .rename(columns={"power": "power_mw"}))
        p = DATA_DIR / f"{pre}_generation_{suffix}.parquet"
        gw.to_parquet(p, index=False)
        print(f"  -> {p.name}: {len(gw):,} rows, "
              f"{gw['interval'].min()} .. {gw['interval'].max()}", flush=True)

    mkt = pull_stream(ses, "market", "market", ["demand", "price"], None, a.region, s, e)
    if not mkt.empty:
        mw = (mkt.pivot_table(index="interval", columns="metric", values="value",
                              aggfunc="first").reset_index()
                 .rename(columns={"demand": "demand_mw", "price": "price_aud_per_mwh"}))
        p = DATA_DIR / f"{pre}_market_{suffix}.parquet"
        mw.to_parquet(p, index=False)
        print(f"  -> {p.name}: {len(mw):,} rows, "
              f"{mw['interval'].min()} .. {mw['interval'].max()}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
