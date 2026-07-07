"""Mine natural stress episodes from the hist dataset -> stress_episodes.json.

Each episode is a contiguous run of trigger days (padded with a seed day before
for the 24 h lookback), tagged with every criterion it fires and the split it
falls in. Primary evaluation episodes are the val/test ones (models never
trained on them); train-span episodes are kept but flagged in_sample=True
(valid for comparing inference-time mechanisms, optimistic for accuracy).
See constraint_research.md "Stages" / stress-test design.

    python3 constraints/mine_episodes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402

TOP_N = 12                      # per class, within val+test; train keeps top 3
MAX_EPISODE_DAYS = 5            # split mega-merges (June 2022 ran for weeks)
OUT = HERE / "stress_episodes.json"


def main():
    df = pipeline.build_table("5min", dataset="hist")
    cfg = pipeline.DATASETS["hist"]
    d = pd.DataFrame(index=df.resample("1D").size().index)

    d["peak_demand"] = df["demand_mw"].resample("1D").max()
    nd_1h = df["net_demand"].diff(12)                       # 1 h ramp at 5-min steps
    d["max_ramp_1h"] = nd_1h.abs().resample("1D").max()
    d["max_price"] = df["price_aud_per_mwh"].resample("1D").max()
    d["min_price"] = df["price_aud_per_mwh"].resample("1D").min()
    d["batt_mwh"] = (df["battery_discharging"].resample("1D").sum() * 5 / 60)
    d["min_nd"] = df["net_demand"].resample("1D").min()
    d = d.dropna()

    d["split"] = np.where(d.index <= cfg["train_end"], "train",
                          np.where(d.index <= cfg["val_end"], "val", "test"))
    eval_m = d["split"] != "train"

    def topd(col, ascending, n, mask):
        return set(d[mask].sort_values(col, ascending=ascending).head(n).index)

    triggers: dict[pd.Timestamp, set[str]] = {}

    def add(days, tag):
        for day in days:
            triggers.setdefault(day, set()).add(tag)

    for mask, n in ((eval_m, TOP_N), (~eval_m, 3)):
        add(topd("peak_demand", False, n, mask), "peak_demand")
        add(topd("max_ramp_1h", False, n, mask), "high_ramp")
        add(topd("batt_mwh", False, n, mask), "batt_cycle")
        add(topd("min_nd", True, n, mask), "min_nd")
        # threshold classes: all qualifying days in eval span, top 3 in train
        spike = d[mask & (d["max_price"] >= 1000)]
        add(set(spike.sort_values("max_price", ascending=False)
                .head(n if mask is eval_m else 3).index), "price_spike")
        neg = d[mask & (d["min_price"] <= -900)]
        add(set(neg.sort_values("min_price").head(n if mask is eval_m else 3).index),
            "neg_price")

    # merge consecutive trigger days into episodes, capped at MAX_EPISODE_DAYS
    days = sorted(triggers)
    episodes, cur = [], [days[0]]
    for day in days[1:]:
        if (day - cur[-1]).days <= 1 and len(cur) < MAX_EPISODE_DAYS:
            cur.append(day)
        else:
            episodes.append(cur); cur = [day]
    episodes.append(cur)

    out = []
    for ep in episodes:
        s, e = ep[0], ep[-1] + pd.Timedelta(days=1)
        span = df.loc[s:e - pd.Timedelta(minutes=5)]
        tags = sorted(set().union(*(triggers[day] for day in ep)))
        row_d = d.loc[ep]
        out.append({
            "id": f"{s:%Y%m%d}" + (f"_{len(ep)}d" if len(ep) > 1 else ""),
            "seed_day": str((s - pd.Timedelta(days=1)).date()),
            "start": str(s.date()), "end_excl": str(e.date()),
            "n_days": len(ep), "tags": tags,
            "split": d.loc[ep[0], "split"],
            "in_sample": bool(d.loc[ep[0], "split"] == "train"),
            "peak_demand_mw": round(float(row_d["peak_demand"].max()), 1),
            "max_ramp_1h_mw": round(float(row_d["max_ramp_1h"].max()), 1),
            "max_price": round(float(row_d["max_price"].max()), 1),
            "min_price": round(float(row_d["min_price"].min()), 1),
            "batt_mwh_day": round(float(row_d["batt_mwh"].max()), 1),
            "min_nd_mw": round(float(row_d["min_nd"].min()), 1),
        })

    OUT.write_text(json.dumps(out, indent=2))
    counts = pd.Series([t for ep in out for t in ep["tags"]]).value_counts()
    by_split = pd.Series([ep["split"] for ep in out]).value_counts()
    print(f"{len(out)} episodes ({dict(by_split)}) -> {OUT.name}")
    print("tag counts:", dict(counts))
    for ep in out:
        if not ep["in_sample"]:
            print(f"  {ep['id']:12s} {ep['split']:5s} {','.join(ep['tags']):45s} "
                  f"peak={ep['peak_demand_mw']:.0f} ramp1h={ep['max_ramp_1h_mw']:.0f} "
                  f"price=[{ep['min_price']:.0f},{ep['max_price']:.0f}] "
                  f"batt={ep['batt_mwh_day']:.0f}MWh nd_min={ep['min_nd_mw']:.0f}")


if __name__ == "__main__":
    main()
