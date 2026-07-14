"""Export a small committable renewables extract for the stacked figures.

The all-energy stacked charts (stack_plots.py) need actual wind, solar_utility
and their curtailment — columns that live in the raw parquets, which are
gitignored (regenerable, hundreds of MB). Colab clones therefore can't draw the
renewable layers. This script runs ONCE on a machine that has the parquets and
writes data/renewables_extract_hist.parquet — 4 float32 columns over the hist
TEST window (~54k rows, <1 MB), which IS committed (see .gitignore exception).

Sources, best-first:
  generation:  vic_generation_20211001_20260706.parquet (hist)  else last365
  curtailment: vic_curtailment_last365.parquet (only source; rows after its
               coverage end are left NaN so plots can mark them, filled 0 there)

    python3 script/export_renewables_extract.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "lib"))

OUT = DATA / "renewables_extract_hist.parquet"
TEST_START = "2026-01-01"          # hist test split (pipeline.DATASETS['hist'])


def first_existing(*names: str) -> Path | None:
    for n in names:
        p = DATA / n
        if p.exists():
            return p
    return None


def main():
    gen_p = first_existing("vic_generation_20211001_20260706.parquet",
                           "vic_generation_last365.parquet")
    cur_p = first_existing("vic_curtailment_last365.parquet")
    if gen_p is None:
        sys.exit("no generation parquet found in data/ — run script/pull_data_hist.py "
                 "(or pull_data.py) first; this exporter needs the raw data once")

    gen = pd.read_parquet(gen_p)
    gen["interval"] = pd.to_datetime(gen["interval"])
    wide = gen.pivot_table(index="interval", columns="fueltech",
                           values="power_mw", aggfunc="mean").sort_index()
    idx = wide.index[wide.index >= pd.Timestamp(TEST_START, tz=wide.index.tz)]
    out = pd.DataFrame(index=idx)
    out["wind"] = wide["wind"].reindex(idx)
    out["solar"] = wide["solar_utility"].reindex(idx)

    if cur_p is not None:
        cur = pd.read_parquet(cur_p)
        cur["interval"] = pd.to_datetime(cur["interval"])
        cur = cur.set_index("interval").sort_index()
        out["wind_curt"] = cur["curtailment_wind_mw"].reindex(idx)
        out["solar_curt"] = cur["curtailment_solar_mw"].reindex(idx)
        cov = cur.index.max()
        print(f"curtailment coverage to {cov} (NaN after; plots fill 0)")
    else:
        out["wind_curt"] = np.nan
        out["solar_curt"] = np.nan
        print("no curtailment parquet — curtailment layers will be empty")

    out = out.astype(np.float32)
    out.to_parquet(OUT)
    n_nan = int(out[["wind", "solar"]].isna().any(axis=1).sum())
    print(f"wrote {OUT} rows={len(out)} span={idx.min()} → {idx.max()} "
          f"gen-NaN rows={n_nan} size={OUT.stat().st_size/1e6:.2f} MB")
    print("commit it: git add data/renewables_extract_hist.parquet && git commit && git push")


if __name__ == "__main__":
    main()
