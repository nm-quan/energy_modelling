"""Refresh the four VIC1 data streams into data/updata/ for a fixed window.

Thin orchestrator over the existing reference pullers in script/:
  - pull_data_hist.py   -> generation (energy/fuel) + market (demand + price)
  - pull_curtailment_hist.py -> wind + solar-utility curtailment
  - pull_capacity_hist.py    -> monthly capacity by fueltech

It reuses those modules' functions verbatim but repoints BOTH their output dir
and their chunk cache into data/updata/ so the tracked data/ tree is left alone
and every artifact from this pull lands under one folder.

Window (inclusive start, exclusive end): 2025-01-01 -> 2026-08-14.

Auth: OPENELECTRICITY_API_KEY env var (never hardcode).

    OPENELECTRICITY_API_KEY=oe_xxx python script/pull_updata.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT / "script"
UPDATA = ROOT / "data" / "updata"
CACHE = UPDATA / "pull_cache"

REGION = "VIC1"
START = "2025-01-01"
END = "2026-08-14"          # exclusive


def load(name: str):
    """Import a sibling script as a module without triggering its __main__."""
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def session_for(key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
    return s


def main() -> None:
    key = os.environ.get("OPENELECTRICITY_API_KEY")
    if not key:
        sys.exit("set OPENELECTRICITY_API_KEY")

    UPDATA.mkdir(parents=True, exist_ok=True)
    start = datetime.fromisoformat(START)
    end = datetime.fromisoformat(END)
    suffix = f"{start:%Y%m%d}_{end:%Y%m%d}"
    session = session_for(key)

    print(f"== updata pull {REGION} {START} -> {END} into {UPDATA.relative_to(ROOT)} ==")

    # --- 1 & 2: generation (energy/fuel) + market (demand + price) -------------
    hist = load("pull_data_hist")
    hist.CACHE_DIR = CACHE                      # isolate chunk cache under updata

    print("\n[1/4] generation (power + emissions by fueltech)")
    gen = hist.pull_stream(session, "gen", "data", ["power", "emissions"],
                           "fueltech", REGION, start, end)
    if not gen.empty:
        gw = (gen.pivot_table(index=["interval", "fueltech"], columns="metric",
                              values="value", aggfunc="first")
                 .reset_index()
                 .rename(columns={"power": "power_mw", "emissions": "emissions_t"}))
        p = UPDATA / f"vic_generation_{suffix}.parquet"
        gw.to_parquet(p, index=False)
        print(f"  -> {p.name}: {len(gw):,} rows, {gw['interval'].min()} .. {gw['interval'].max()}")

    print("\n[2/4] market (demand + price)")
    mkt = hist.pull_stream(session, "market", "market", ["demand", "price"],
                           None, REGION, start, end)
    if not mkt.empty:
        mw = (mkt.pivot_table(index="interval", columns="metric",
                              values="value", aggfunc="first")
                 .reset_index()
                 .rename(columns={"demand": "demand_mw", "price": "price_aud_per_mwh"}))
        p = UPDATA / f"vic_market_{suffix}.parquet"
        mw.to_parquet(p, index=False)
        print(f"  -> {p.name}: {len(mw):,} rows, {mw['interval'].min()} .. {mw['interval'].max()}")

    # --- 3: curtailment --------------------------------------------------------
    print("\n[3/4] curtailment (wind + solar_utility)")
    curt = load("pull_curtailment_hist")
    curt.DATA = UPDATA                          # output dir
    curt.CACHE = CACHE / "curtailment_hist"     # isolate cache under updata
    sys.argv = ["pull_curtailment_hist", "--start", START, "--end", END]
    curt.main()
    src = UPDATA / "vic_curtailment_hist.parquet"
    if src.exists():
        dst = UPDATA / f"vic_curtailment_{suffix}.parquet"
        src.replace(dst)
        print(f"  -> renamed to {dst.name}")

    # --- 4: monthly capacity ---------------------------------------------------
    print("\n[4/4] monthly capacity by fueltech (facilities refreshed)")
    cap = load("pull_capacity_hist")
    cap.DATA_DIR = UPDATA                        # output dir
    cap.CACHE_DIR = CACHE                        # isolate facilities cache under updata
    sys.argv = ["pull_capacity_hist", "--region", REGION, "--group", "fueltech",
                "--freq", "MS", "--start", START, "--end", END, "--refresh"]
    cap.main()

    print("\n== done ==")


if __name__ == "__main__":
    main()
