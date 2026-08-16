"""Build a per-fueltech capacity time series for a NEM region from unit metadata.

WHY THIS EXISTS
---------------
The OpenElectricity API has NO `capacity` metric. The full metric enum (26 entries,
verified against https://api.openelectricity.org.au/v4/openapi.json) is power, energy,
price, market_value, demand*, generation_renewable*, curtailment*, emissions,
renewable_proportion, pollution, storage_battery and flow_*. Passing `metrics=capacity`
to /v4/data/network/{net} returns HTTP 422.

So the `5m / 1h / 1d / 7d / 1M / 3M / season / 1y / fy` interval list in the docs does
NOT apply to capacity -- those are the *data* endpoints. The website's monthly
capacity-by-fueltech chart is not backed by a data endpoint either; it is derived from
per-unit metadata on /v4/facilities/, which is what this script reproduces.

Capacity is a STEP FUNCTION. It changes when a unit commissions or retires, not on a
sampling interval. That means:
  - the underlying event resolution is a DATE, and for many units only a YEAR
    (see the *_specificity fields; `year` means day/month are not authoritative)
  - you can still emit the series on ANY grid, including 5m, by forward-filling the
    steps. `--freq 5min` gives a 5-minute-indexed capacity column that is
    piecewise-constant -- correct, and exactly what you need to join against 5m
    dispatch data to compute capacity factor.
  - a 5m grid does NOT buy you 5m-resolved information. If you want "how much capacity
    was actually biddable at 13:05", that is availability, not capacity, and it lives in
    AEMO bid tables (MAXAVAIL / PASAAVAILABILITY), not in OpenElectricity.

Outputs in ./data/:
  capacity_<region>_by_<group>_<freq>.parquet  interval, <group>, capacity_mw, capacity_mwh, n_units
  capacity_<region>_units.parquet              flat per-unit registry (one row per unit)
  capacity_<region>_specificity.csv            date-quality audit: which units are year-only

Usage:
  python3 script/pull_capacity_hist.py                          # VIC1, monthly, by fueltech
  python3 script/pull_capacity_hist.py --freq 5min --start 2021-10-01
  python3 script/pull_capacity_hist.py --group fueltech_group --freq D
  python3 script/pull_capacity_hist.py --audit                  # print date-quality report only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent          # script/
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "pull_cache"

BASE_URL = "https://api.openelectricity.org.au/v4"
NETWORK = "NEM"
MAX_TRIES = 5

# Units whose status means they never physically generated. Excluded from history by
# default -- a `committed` unit has an expected_operation_date, not a commencement_date,
# so including it would fabricate capacity that never existed.
FORWARD_LOOKING = {"committed"}

# OE stores every battery as THREE unit records for the same physical asset:
#   BALB1  battery              BIDIRECTIONAL   30 MW / 30 MWh
#   BALBL1 battery_charging     LOAD            30 MW / 30 MWh
#   BALBG1 battery_discharging  GENERATOR       30 MW / 30 MWh
# Summing all three triple-counts both MW and MWh. Verified on VIC1: all 15 battery
# facilities carry exactly 3 records, and no non-battery unit is LOAD or BIDIRECTIONAL,
# so keying the dedupe on these three fueltechs is exact.
BATTERY_BIDIRECTIONAL = "battery"
BATTERY_DIRECTIONAL = {"battery_charging", "battery_discharging"}
BATTERY_ALL = BATTERY_DIRECTIONAL | {BATTERY_BIDIRECTIONAL}


def api_key() -> str:
    k = os.environ.get("OPENELECTRICITY_API_KEY")
    if not k and (HERE / ".oe_key").exists():
        k = (HERE / ".oe_key").read_text().strip()
    if not k:
        sys.exit("no API key: set OPENELECTRICITY_API_KEY or write script/.oe_key")
    return k


def fetch_facilities(session, region: str, refresh: bool) -> list[dict]:
    """GET /v4/facilities/ for a region, with an on-disk cache.

    Facilities is a metadata endpoint -- small, unpaginated, and it costs one credit.
    Cache it so repeated resampling at different --freq does not re-hit the API.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"facilities_{NETWORK}_{region}.json"
    if cache.exists() and not refresh:
        print(f"  using cached {cache.relative_to(ROOT)} "
              f"(--refresh to re-pull)", flush=True)
        return json.loads(cache.read_text())["data"]

    params = {"network_id": NETWORK, "network_region": region}
    for attempt in range(1, MAX_TRIES + 1):
        r = session.get(f"{BASE_URL}/facilities/", params=params, timeout=60)
        if r.status_code == 200:
            payload = r.json()
            cache.write_text(json.dumps(payload))
            return payload["data"]
        if r.status_code == 401:
            sys.exit("HTTP 401: API key rejected. Both keys currently in this repo are "
                     "dead -- write a live one to script/.oe_key.")
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2 ** attempt))
            print(f"    429 rate-limited, sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500 and attempt < MAX_TRIES:
            time.sleep(2 ** attempt)
            continue
        sys.exit(f"HTTP {r.status_code}: {r.text[:300]}")
    sys.exit("exhausted retries")


# NEM market time is a fixed UTC+10 (no daylight saving), which is also what the
# /v4/data/* endpoints return, so every timestamp here is emitted as tz-naive NEM time
# to join directly against the dispatch pulls in pull_data_hist.py.
NEM_OFFSET = pd.Timedelta(hours=10)


def _declared_date(v):
    """Parse a declared date field (commencement / closure / expected_*).

    These come back looking like '1972-12-31T14:00:00+10:00'. The offset label is WRONG:
    the numeric part is UTC, so the true NEM-local value is the numbers + 10h, which
    lands exactly on local midnight of the intended date. Verified against Yallourn W --
    reading the offset literally dates W1 to 1972, whereas +10h gives 1973, which is the
    actual commissioning year (W2 1975, W3 1981, W4 1982 all line up the same way).

    Parsing these with utc=True instead SUBTRACTS 10h and lands on 04:00 the day before,
    shifting roughly every unit in the fleet back by one day.
    """
    if not v:
        return pd.NaT
    naive = pd.to_datetime(str(v)[:19], errors="coerce")   # drop the bogus offset
    if pd.isna(naive):
        return pd.NaT
    return (naive + NEM_OFFSET).normalize()


def _observed_ts(v):
    """Parse an observation field (data_first_seen / data_last_seen) to NEM-local time.

    Unlike the declared dates, these ARE correctly labelled -- data_last_seen tracks the
    live dispatch feed to the minute -- so the offset is honoured and converted to +10.
    """
    if not v:
        return pd.NaT
    ts = pd.to_datetime(v, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return (ts + NEM_OFFSET).tz_localize(None)


def flatten_units(facilities: list[dict]) -> pd.DataFrame:
    """One row per unit, with the fields needed to place it on a timeline."""
    rows = []
    for f in facilities:
        for u in f.get("units", []) or []:
            rows.append({
                "facility_code": f.get("code"),
                "facility_name": f.get("name"),
                "network_region": f.get("network_region"),
                "unit_code": u.get("code"),
                "fueltech": u.get("fueltech_id"),
                "status": u.get("status_id"),
                "dispatch_type": u.get("dispatch_type"),
                "capacity_registered": u.get("capacity_registered"),
                "capacity_maximum": u.get("capacity_maximum"),
                "capacity_storage": u.get("capacity_storage"),
                "max_generation": u.get("max_generation"),
                "commencement_date": _declared_date(u.get("commencement_date")),
                "commencement_spec": u.get("commencement_date_specificity"),
                "closure_date": _declared_date(u.get("closure_date")),
                "closure_spec": u.get("closure_date_specificity"),
                "expected_operation_date": _declared_date(u.get("expected_operation_date")),
                "expected_closure_date": _declared_date(u.get("expected_closure_date")),
                "data_first_seen": _observed_ts(u.get("data_first_seen")),
                "data_last_seen": _observed_ts(u.get("data_last_seen")),
            })
    return pd.DataFrame(rows)


# Fueltech groups mirror the API's fueltech_group_id enum, so `--group fueltech_group`
# lines up with what /v4/data/network returns under secondary_grouping=fueltech_group.
FUELTECH_GROUP = {
    "coal_black": "coal", "coal_brown": "coal",
    "gas_ccgt": "gas", "gas_ocgt": "gas", "gas_recip": "gas",
    "gas_steam": "gas", "gas_wcmg": "gas",
    "solar_utility": "solar", "solar_rooftop": "solar", "solar_thermal": "solar",
    "wind": "wind", "wind_offshore": "wind",
    "hydro": "hydro", "pumps": "pumps",
    "battery": "battery", "battery_charging": "battery_charging",
    "battery_discharging": "battery_discharging",
    "bioenergy_biogas": "bioenergy", "bioenergy_biomass": "bioenergy",
    "distillate": "distillate",
}


def resolve_window(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each unit a [live_from, live_to) window and a capacity value.

    Precedence is deliberate:
      start = commencement_date -> data_first_seen -> expected_operation_date
      end   = closure_date -> data_last_seen (retired only) -> open

    data_first_seen/data_last_seen are interval-precise observations from dispatch data,
    whereas commencement_date is a declared date that may be year-only. Where a declared
    date is missing we fall back to the observation, which is the stronger evidence.
    """
    d = df.copy()
    d["live_from"] = d["commencement_date"] \
        .fillna(d["data_first_seen"]) \
        .fillna(d["expected_operation_date"])

    retired = d["status"].eq("retired")
    d["live_to"] = d["closure_date"]
    d.loc[retired & d["live_to"].isna(), "live_to"] = d.loc[
        retired & d["live_to"].isna(), "data_last_seen"]

    d["capacity_mwh"] = d["capacity_storage"]
    return d


def pick_capacity(units: pd.DataFrame, basis: str) -> pd.DataFrame:
    """Choose which capacity column drives the series.

    The web UI plots `capacity_maximum` (demonstrated), NOT `capacity_registered`
    (nameplate) -- confirmed against the VIC1 chart for May 2026, where maximum
    reproduces solar 1,776 / wind 5,630 / battery 2,260 MW and registered does not
    (2,146 / 5,753 / 2,759). Default to maximum so output is comparable to the site;
    pass --capacity registered for nameplate, which is the right basis for headroom.
    """
    d = units.copy()
    if basis == "maximum":
        d["capacity_mw"] = d["capacity_maximum"].fillna(d["capacity_registered"])
    else:
        d["capacity_mw"] = d["capacity_registered"].fillna(d["capacity_maximum"])
    return d


def build_series(units: pd.DataFrame, group: str, freq: str,
                 start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Sum live capacity onto a regular grid.

    A unit contributes to bucket t when live_from <= t and (live_to is null or t <
    live_to). The grid can be any pandas freq; the result is a step function sampled on
    that grid, so `5min` is legal and piecewise-constant rather than newly informative.
    """
    idx = pd.date_range(start, end, freq=freq)
    if len(idx) == 0:
        sys.exit(f"empty index for freq={freq} over {start.date()}..{end.date()}")

    u = units.dropna(subset=["live_from"]).copy()
    u = u[u["capacity_mw"].notna() & (u["capacity_mw"] > 0)]
    if u.empty:
        sys.exit("no units with a usable start date and capacity")

    grid = idx.values.astype("datetime64[ns]")[:, None]          # (T, 1)
    frm = u["live_from"].values.astype("datetime64[ns]")[None, :]  # (1, N)
    to = u["live_to"].values.astype("datetime64[ns]")[None, :]

    live = (grid >= frm) & (np.isnat(to) | (grid < to))          # (T, N) bool

    cap = np.nan_to_num(u["capacity_mw"].to_numpy(dtype=float))
    mwh = np.nan_to_num(u["capacity_mwh"].to_numpy(dtype=float))
    keys = u[group].fillna("unknown").to_numpy()

    frames = []
    for k in pd.unique(keys):
        m = keys == k
        frames.append(pd.DataFrame({
            "interval": idx,
            group: k,
            "capacity_mw": live[:, m] @ cap[m],
            "capacity_mwh": live[:, m] @ mwh[m],
            "n_units": live[:, m].sum(axis=1).astype(int),
        }))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["interval", group]).reset_index(drop=True)


def dedupe_batteries(units: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Collapse OE's triple battery records down to one physical asset.

    bidirectional : keep the BIDIRECTIONAL `battery` record only (correct totals)
    directional   : keep charging + discharging, drop `battery` (per-direction view)
    all           : keep everything -- inspection only, TOTALS WILL TRIPLE-COUNT
    """
    if mode == "all":
        return units
    drop = BATTERY_DIRECTIONAL if mode == "bidirectional" else {BATTERY_BIDIRECTIONAL}
    kept = units[~units["fueltech"].isin(drop)]
    n = len(units) - len(kept)
    if n:
        print(f"  battery dedupe ({mode}): dropped {n} duplicate unit records", flush=True)
    return kept


def audit(units: pd.DataFrame) -> pd.DataFrame:
    """Flag units whose timeline placement is weak, so the series carries its caveats."""
    d = units.copy()
    d["year_only_start"] = d["commencement_spec"].eq("year")
    d["no_declared_start"] = d["commencement_date"].isna()
    d["start_from_observation"] = d["commencement_date"].isna() & d["data_first_seen"].notna()
    d["no_start_at_all"] = d["live_from"].isna()
    d["no_capacity"] = d["capacity_mw"].isna() | d["capacity_mw"].le(0)
    cols = ["unit_code", "facility_name", "fueltech", "status", "capacity_mw",
            "commencement_date", "commencement_spec", "data_first_seen",
            "year_only_start", "no_declared_start", "start_from_observation",
            "no_start_at_all", "no_capacity"]
    flagged = d[d["year_only_start"] | d["no_declared_start"] | d["no_capacity"]]
    return flagged[cols].sort_values(["fueltech", "unit_code"])


# A 5m series over 5 years is ~4.6M rows -> roughly 200 MB of CSV. pull_data_hist.py
# deliberately went parquet-only for the same reason, so CSV is skipped past this
# unless explicitly forced.
CSV_ROW_WARN = 1_000_000


def write_csvs(series: pd.DataFrame, group: str, out_parquet: Path,
               force: bool) -> list[Path]:
    """Write long + wide CSV siblings of the parquet output.

    The wide pivot (interval x group, with a TOTAL column) is the one worth opening in
    a spreadsheet; the long form mirrors the parquet exactly for code that wants it.
    """
    written = []
    stem = out_parquet.with_suffix("")

    if len(series) > CSV_ROW_WARN and not force:
        print(f"  skipped long CSV: {len(series):,} rows would be ~"
              f"{len(series) * 45 / 1e6:.0f} MB. Pass --force-csv to write it anyway; "
              f"the wide CSV below carries the same information.", flush=True)
    else:
        long_csv = stem.with_suffix(".csv")
        series.to_csv(long_csv, index=False, float_format="%.3f")
        written.append(long_csv)

    wide = series.pivot_table(index="interval", columns=group,
                              values="capacity_mw", aggfunc="sum").fillna(0.0)
    wide["TOTAL"] = wide.sum(axis=1)
    wide_csv = Path(f"{stem}_wide.csv")
    wide.round(1).to_csv(wide_csv, float_format="%.1f")
    written.append(wide_csv)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="VIC1")
    ap.add_argument("--group", default="fueltech",
                    choices=["fueltech", "fueltech_group"],
                    help="aggregate by raw fueltech_id or the coarser group")
    ap.add_argument("--freq", default="MS",
                    help="pandas freq for the output grid: MS (month start, matches the "
                         "website), D, YS, 1h, 5min. Capacity is a step function -- finer "
                         "grids forward-fill, they do not add resolution.")
    ap.add_argument("--start", default="2021-10-01",
                    help="grid start (default matches pull_data_hist.py)")
    ap.add_argument("--end", default=None, help="grid end (default: today)")
    ap.add_argument("--status", default=None,
                    help="comma-separated status filter, e.g. operating,retired. "
                         "Default excludes only forward-looking (committed) units.")
    ap.add_argument("--capacity", default="maximum",
                    choices=["maximum", "registered"],
                    help="which capacity column drives the series. maximum (default) "
                         "reproduces the website chart; registered is nameplate.")
    ap.add_argument("--rooftop-mw", type=float, default=None,
                    help="add a flat rooftop-solar capacity band, in MW. The API exposes "
                         "NO rooftop capacity (no solar_rooftop facility record) even "
                         "though it reports rooftop power, so the website's total cannot "
                         "be reproduced without supplying this. VIC1 mid-2026 is ~5,400.")
    ap.add_argument("--battery", default="bidirectional",
                    choices=["bidirectional", "directional", "all"],
                    help="OE stores each battery as 3 unit records. bidirectional (default) "
                         "keeps one and gives correct totals; directional keeps "
                         "charging+discharging; all keeps every record and TRIPLE-COUNTS.")
    ap.add_argument("--refresh", action="store_true", help="re-pull facilities, ignore cache")
    ap.add_argument("--audit", action="store_true", help="print the date-quality report and exit")
    ap.add_argument("--force-csv", action="store_true",
                    help=f"write the long CSV even past {CSV_ROW_WARN:,} rows "
                         f"(a 5m series is ~200 MB)")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(datetime.now().date())
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key()}",
                            "Accept": "application/json"})

    print(f"pulling facilities: {NETWORK}/{args.region}", flush=True)
    facilities = fetch_facilities(session, args.region, args.refresh)
    units = flatten_units(facilities)
    print(f"  {len(facilities)} facilities, {len(units)} units", flush=True)

    if args.status:
        keep = {s.strip() for s in args.status.split(",")}
        units = units[units["status"].isin(keep)]
        print(f"  status filter {sorted(keep)} -> {len(units)} units", flush=True)
    else:
        units = units[~units["status"].isin(FORWARD_LOOKING)]

    units = dedupe_batteries(units, args.battery)
    units = resolve_window(units)
    units = pick_capacity(units, args.capacity)

    unit_path = DATA_DIR / f"capacity_{args.region}_units.parquet"
    units.to_parquet(unit_path, index=False)
    units.to_csv(unit_path.with_suffix(".csv"), index=False)
    print(f"  wrote {unit_path.relative_to(ROOT)} (+ .csv)", flush=True)

    flagged = audit(units)
    audit_path = DATA_DIR / f"capacity_{args.region}_specificity.csv"
    flagged.to_csv(audit_path, index=False)
    print(f"\ndate-quality audit ({len(flagged)}/{len(units)} units flagged) "
          f"-> {audit_path.relative_to(ROOT)}", flush=True)
    print(f"  year-only commencement : {int(flagged['year_only_start'].sum())}")
    print(f"  no declared start      : {int(flagged['no_declared_start'].sum())}")
    print(f"  no capacity value      : {int(flagged['no_capacity'].sum())}")
    if args.audit:
        with pd.option_context("display.max_rows", 200, "display.width", 200):
            print(flagged.to_string(index=False))
        return

    if args.group == "fueltech_group":
        units["fueltech_group"] = units["fueltech"].map(FUELTECH_GROUP) \
            .fillna(units["fueltech"])

    series = build_series(units, args.group, args.freq, start, end)

    if args.rooftop_mw:
        # Flat band: OE gives no rooftop commissioning history, so this cannot be
        # ramped honestly. It exists to reconcile against the site's total, not to
        # model rooftop growth.
        idx = series["interval"].drop_duplicates().sort_values()
        series = pd.concat([series, pd.DataFrame({
            "interval": idx, args.group: "solar_rooftop",
            "capacity_mw": args.rooftop_mw, "capacity_mwh": 0.0, "n_units": 0,
        })], ignore_index=True).sort_values(["interval", args.group]).reset_index(drop=True)
        print(f"  added flat rooftop band: {args.rooftop_mw:,.0f} MW (not from the API)",
              flush=True)

    tag = args.freq.replace("/", "")
    bt = "" if args.battery == "bidirectional" else f"_bat-{args.battery}"
    cb = "" if args.capacity == "maximum" else f"_{args.capacity}"
    out = DATA_DIR / f"capacity_{args.region}_by_{args.group}_{tag}{cb}{bt}.parquet"
    series.to_parquet(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}  "
          f"({len(series):,} rows, {series['interval'].nunique():,} intervals, "
          f"{series[args.group].nunique()} groups)", flush=True)
    for p in write_csvs(series, args.group, out, args.force_csv):
        print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size/1024:,.0f} KB)", flush=True)

    latest = series[series["interval"] == series["interval"].max()]
    print(f"\ncapacity at {latest['interval'].iloc[0].date()}:")
    for _, r in latest.sort_values("capacity_mw", ascending=False).iterrows():
        if r["capacity_mw"] <= 0:
            continue
        mwh = f"  {r['capacity_mwh']:>9,.0f} MWh" if r["capacity_mwh"] > 0 else ""
        print(f"  {r[args.group]:<22} {r['capacity_mw']:>10,.1f} MW  "
              f"({r['n_units']:>3} units){mwh}")
    total = latest["capacity_mw"].sum()
    if args.battery == "all":
        print(f"  {'TOTAL':<22} {total:>10,.1f} MW  <-- TRIPLE-COUNTS BATTERIES "
              f"(--battery all); use bidirectional for a real total")
    else:
        print(f"  {'TOTAL':<22} {total:>10,.1f} MW")


if __name__ == "__main__":
    main()
