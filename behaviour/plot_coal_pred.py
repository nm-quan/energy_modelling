"""
Coal predicted-vs-actual, stacked behaviour view.

Two PNGs (a test-set day and the week containing it). Each PNG has two panels
that are identical except for the brown-coal layer:
  top    = actual coal_brown
  bottom = predicted coal_brown (iTransformer-totdem)
Wind, solar and curtailment are the same real data in both panels, so the only
visible difference is how the coal forecast reshapes the dispatch stack.

Outputs to behaviour/figure/:
  - coal_pred_vs_actual_day.png
  - coal_pred_vs_actual_week.png
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = Path(__file__).resolve().parent / "figure"
FIG.mkdir(parents=True, exist_ok=True)

PRED = ROOT / "ml" / "itransformer_totdem" / "predictions.csv"

COAL = "#7a4a23"
WIND = "#2e8b40"
SOLAR = "#f4c20d"


def load():
    def naive(idx):
        # all sources are +10:00; drop tz to align on local wall-clock
        return idx.tz_localize(None) if idx.tz is not None else idx

    gen = pd.read_parquet(DATA / "vic_generation_last365.parquet")
    gen["interval"] = pd.to_datetime(gen["interval"])
    wide = gen.pivot_table(index="interval", columns="fueltech",
                           values="power_mw", aggfunc="first")
    ctx = pd.DataFrame({"wind": wide["wind"], "solar": wide["solar_utility"]})
    ctx.index = naive(ctx.index)

    cur = pd.read_parquet(DATA / "vic_curtailment_last365.parquet")
    cur["interval"] = pd.to_datetime(cur["interval"])
    cur = cur.set_index("interval")[["curtailment_wind_mw", "curtailment_solar_mw"]]
    cur.columns = ["wind_curt", "solar_curt"]
    cur.index = naive(cur.index)

    pred = pd.read_csv(PRED)
    pred["interval"] = pd.to_datetime(pred["interval"])
    pred = pred.set_index("interval")[["coal_brown_actual", "coal_brown_pred"]]
    pred.columns = ["coal_actual", "coal_pred"]
    pred.index = naive(pred.index)

    df = pred.join(ctx, how="left").join(cur, how="left").sort_index()
    df = df.clip(lower=0)
    return df


def draw_stack(ax, t, coal, df, title):
    wind, wcurt = df["wind"].values, df["wind_curt"].values
    solar, scurt = df["solar"].values, df["solar_curt"].values
    b_coal = coal
    b_wind = b_coal + wind
    b_wcurt = b_wind + wcurt
    b_solar = b_wcurt + solar
    b_scurt = b_solar + scurt

    ax.fill_between(t, 0, b_coal, color=COAL, label="Brown coal")
    ax.fill_between(t, b_coal, b_wind, color=WIND, label="Wind")
    ax.fill_between(t, b_wind, b_wcurt, facecolor="none", edgecolor=WIND,
                    hatch="////", linewidth=0.0, label="Wind curtailment")
    ax.fill_between(t, b_wcurt, b_solar, color=SOLAR, label="Solar (utility)")
    ax.fill_between(t, b_solar, b_scurt, facecolor="none", edgecolor=SOLAR,
                    hatch="////", linewidth=0.0, label="Solar curtailment")
    ax.set_ylabel("MW")
    ax.set_title(title, fontsize=11, loc="left")
    ax.margins(x=0)
    ax.grid(True, axis="y", alpha=0.3)


def make_fig(df, outpath, day_axis=False):
    t = df.index
    ymax = (df["coal_actual"].clip(lower=0) + df["wind"] + df["wind_curt"]
            + df["solar"] + df["solar_curt"]).max() * 1.05

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    draw_stack(axes[0], t, df["coal_actual"].values, df, "Actual coal")
    draw_stack(axes[1], t, df["coal_pred"].values, df, "Predicted coal (iTransformer)")
    for ax in axes:
        ax.set_ylim(0, ymax)
    if day_axis:
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].legend(loc="upper left", ncol=3, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved {outpath}")


def main():
    df = load()
    test = df.dropna(subset=["coal_actual"])
    print(f"test window {test.index.min()} -> {test.index.max()} ({len(test):,} rows)")

    # day with the most wind curtailment in the test set
    daily = test["wind_curt"].groupby(test.index.normalize()).sum()
    day = daily.idxmax()
    print(f"peak wind-curtailment day: {day.date()} ({daily.max():,.0f} MW-5min)")

    d = test.loc[day: day + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)]
    make_fig(d, FIG / "coal_pred_vs_actual_day.png", day_axis=True)

    wk_start = day - pd.Timedelta(days=3)
    wk = test.loc[wk_start: wk_start + pd.Timedelta(days=7)]
    make_fig(wk, FIG / "coal_pred_vs_actual_week.png")


if __name__ == "__main__":
    main()
