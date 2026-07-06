"""
OpenElectricity-style stacked behaviour chart for VIC1.

Stacks actual generation (coal, wind, solar_utility) and overlays the curtailed
headroom for wind and solar as hatched bands sitting above each generation layer
(what could have been generated but was spilled).

Outputs to behaviour/figure/:
  - stacked_random_day.png   one random day from the overlap window
  - stacked_week.png         a representative 7-day window
  - stacked_full.png         the entire overlap window
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = Path(__file__).resolve().parent / "figure"
FIG.mkdir(parents=True, exist_ok=True)

COAL = "#7a4a23"   # brown coal
WIND = "#2e8b40"   # wind green
SOLAR = "#f4c20d"  # solar gold

SEED = 7


def load():
    gen = pd.read_parquet(DATA / "vic_generation_last365.parquet")
    gen["interval"] = pd.to_datetime(gen["interval"])
    wide = (gen.pivot_table(index="interval", columns="fueltech",
                            values="power_mw", aggfunc="first"))
    g = pd.DataFrame({
        "coal": wide["coal_brown"],
        "wind": wide["wind"],
        "solar": wide["solar_utility"],
    })

    cur = pd.read_parquet(DATA / "vic_curtailment_last365.parquet")
    cur["interval"] = pd.to_datetime(cur["interval"])
    cur = cur.set_index("interval")[["curtailment_wind_mw", "curtailment_solar_mw"]]
    cur.columns = ["wind_curt", "solar_curt"]

    df = g.join(cur, how="inner").sort_index()
    df = df.clip(lower=0)  # drop tiny negative measurement artifacts
    df.index = df.index.tz_localize(None)  # plot in local (+10:00) wall-clock
    return df


def stacked_plot(df, title, outpath, day_axis=False):
    t = df.index
    coal = df["coal"].values
    wind = df["wind"].values
    wcurt = df["wind_curt"].values
    solar = df["solar"].values
    scurt = df["solar_curt"].values

    # cumulative baselines, bottom -> top
    b_coal = coal
    b_wind = b_coal + wind
    b_wcurt = b_wind + wcurt
    b_solar = b_wcurt + solar
    b_scurt = b_solar + scurt

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(t, 0, b_coal, color=COAL, label="Brown coal")
    ax.fill_between(t, b_coal, b_wind, color=WIND, label="Wind")
    ax.fill_between(t, b_wind, b_wcurt, facecolor="none", edgecolor=WIND,
                    hatch="////", linewidth=0.0, label="Wind curtailment")
    ax.fill_between(t, b_wcurt, b_solar, color=SOLAR, label="Solar (utility)")
    ax.fill_between(t, b_solar, b_scurt, facecolor="none", edgecolor=SOLAR,
                    hatch="////", linewidth=0.0, label="Solar curtailment")

    ax.set_ylabel("MW")
    ax.set_title(title)
    ax.margins(x=0)
    ax.grid(True, axis="y", alpha=0.3)
    if day_axis:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper left", ncol=3, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved {outpath}")


def main():
    df = load()
    print(f"overlap {df.index.min()} -> {df.index.max()}  ({len(df):,} rows)")

    # full window
    stacked_plot(df, "VIC1 generation + curtailment — full window",
                 FIG / "stacked_full.png")

    # representative week
    rng = np.random.default_rng(SEED)
    days = pd.Index(df.index.normalize().unique())
    wk_start = days[rng.integers(0, len(days) - 7)]
    wk = df.loc[wk_start: wk_start + pd.Timedelta(days=7)]
    stacked_plot(wk, f"VIC1 generation + curtailment — week of {wk_start.date()}",
                 FIG / "stacked_week.png")

    # random day
    day = days[rng.integers(0, len(days))]
    d = df.loc[day: day + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)]
    stacked_plot(d, f"VIC1 generation + curtailment — {day.date()}",
                 FIG / "stacked_random_day.png", day_axis=True)


if __name__ == "__main__":
    main()
