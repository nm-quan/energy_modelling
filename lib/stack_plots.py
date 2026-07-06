"""Shared all-energy dispatch-stack figures for the debugging models.

The forecast covers only the 6 dispatchable targets. To match the behaviour
reference (behaviour/figure/stacked_random_day.png) the stack also shows actual
wind, solar (utility) and their curtailment (hatched bands above each renewable
layer). Renewables are NOT predicted by the model, so those layers are identical
in the actual and predicted panels; only the dispatchable layers differ.

Stack, bottom -> top:
    coal_brown, gas_steam, gas_ocgt, hydro, battery_discharging,
    wind, wind curtailment (hatched), solar_utility, solar curtailment (hatched)
battery_charging is drawn below 0 (load).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]   # energy_modelling/
DATA = ROOT / "data"

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]

# dispatchable colours (match script/plotting.COLORS), renewables match behaviour/.
# gas/battery recoloured away from solar-gold / wind-green so the layers don't clash.
COLORS = {
    "coal_brown": "saddlebrown",
    "gas_steam": "#d62728",        # red (was orange, clashed with solar gold)
    "gas_ocgt": "#ff7f0e",         # orange (was gold, clashed with solar gold)
    "hydro": "royalblue",
    "battery_discharging": "#9467bd",  # purple (was seagreen, clashed with wind green)
    "battery_charging": "dimgray",
}
WIND = "#2e8b40"
SOLAR = "#f4c20d"

# (label, source, key, color, hatch) — source: "disp" target name or "ren" column
STACK_ORDER = [
    ("coal_brown", "disp", "coal_brown", COLORS["coal_brown"], None),
    ("gas_steam", "disp", "gas_steam", COLORS["gas_steam"], None),
    ("gas_ocgt", "disp", "gas_ocgt", COLORS["gas_ocgt"], None),
    ("hydro", "disp", "hydro", COLORS["hydro"], None),
    ("battery_discharging", "disp", "battery_discharging", COLORS["battery_discharging"], None),
    ("wind", "ren", "wind", WIND, None),
    ("wind curtailment", "ren", "wind_curt", WIND, "////"),
    ("solar_utility", "ren", "solar", SOLAR, None),
    ("solar curtailment", "ren", "solar_curt", SOLAR, "////"),
]


def load_renewables(test_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Actual wind, solar_utility and wind/solar curtailment aligned to test_index."""
    gen = pd.read_parquet(DATA / "vic_generation_last365.parquet")
    gen["interval"] = pd.to_datetime(gen["interval"])
    wide = gen.pivot_table(index="interval", columns="fueltech",
                           values="power_mw", aggfunc="mean").sort_index()
    cur = pd.read_parquet(DATA / "vic_curtailment_last365.parquet")
    cur["interval"] = pd.to_datetime(cur["interval"])
    cur = cur.set_index("interval").sort_index()

    df = pd.DataFrame(index=test_index)
    df["wind"] = wide["wind"].reindex(test_index)
    df["solar"] = wide["solar_utility"].reindex(test_index)
    df["wind_curt"] = cur["curtailment_wind_mw"].reindex(test_index)
    df["solar_curt"] = cur["curtailment_solar_mw"].reindex(test_index)
    return df.clip(lower=0).fillna(0.0)


def pick_full_days(test_index: pd.DatetimeIndex, n_days=4, seed=42):
    """Pick a random run of n_days consecutive complete (288-step) days.

    Returns (list_of_dates, positions) where positions index test_index in order.
    """
    dser = pd.Series(test_index.date, index=range(len(test_index)))
    counts = dser.value_counts()
    full = {d for d, c in counts.items() if c == 288}
    valid_starts = [d for d in sorted(full)
                    if all((d + dt.timedelta(days=k)) in full for k in range(n_days))]
    if not valid_starts:
        raise ValueError(f"no run of {n_days} consecutive full days in the test set")
    rng = np.random.default_rng(seed)
    start = valid_starts[int(rng.integers(len(valid_starts)))]
    days = [start + dt.timedelta(days=k) for k in range(n_days)]
    pos = np.where(np.isin(test_index.date, days))[0]
    return days, pos


def _draw_full_stack(ax, x, disp, ren, ti):
    """disp: (n,6) dispatchable MW in TARGETS order. ren: DataFrame (n rows)."""
    base = np.zeros(len(x))
    for label, src, key, color, hatch in STACK_ORDER:
        vals = disp[:, ti[key]] if src == "disp" else ren[key].to_numpy()
        top = base + vals
        if hatch:
            ax.fill_between(x, base, top, facecolor="none", edgecolor=color,
                            hatch=hatch, linewidth=0.0, label=label)
        else:
            ax.fill_between(x, base, top, color=color, alpha=0.9, label=label)
        base = top
    ax.fill_between(x, 0, -disp[:, ti["battery_charging"]],
                    color=COLORS["battery_charging"], alpha=0.6, label="battery_charging (load)")
    ax.axhline(0, color="k", lw=0.6)
    ax.margins(x=0)
    ax.grid(True, axis="y", alpha=0.3)


def fig_stacked_window(win_index, actual_win, pred_win, ren_win, out, tag):
    """Multi-day window: all-energy dispatch stack, actual (top) vs predicted (bottom)."""
    ti = {t: i for i, t in enumerate(TARGETS)}
    t = pd.DatetimeIndex(win_index).tz_localize(None)
    n_days = len(np.unique(t.date))
    f, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, sharey=True)
    for ax, (name, a) in zip(axes, [("actual", actual_win), ("predicted", pred_win)]):
        _draw_full_stack(ax, t, a, ren_win, ti)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_ylabel("MW")
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].xaxis.set_minor_locator(mdates.HourLocator(byhour=(6, 12, 18)))
    axes[1].set_xlabel("date")
    axes[0].legend(loc="upper left", ncol=5, fontsize=7, framealpha=0.9)
    f.suptitle(f"{tag} — all-energy dispatch stack (renewables + curtailment actual) — "
               f"{n_days} days {t[0].date()} to {t[-1].date()} (battery charging shown negative)")
    f.tight_layout(); f.savefig(out, dpi=130); plt.close(f)


def fig_stacked_3panel(day_index, actual, base, scen, ren_day, out, tag,
                       free_hours=(11, 13)):
    """One day, all-energy stack across 3 panels: actual / baseline / scenario.

    Only the free window (where demand is shifted up) is shaded gold.
    """
    ti = {t: i for i, t in enumerate(TARGETS)}
    x = np.asarray(day_index.hour) + np.asarray(day_index.minute) / 60.0
    f, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, sharey=True)
    panels = [("actual", actual), ("baseline (no shift)", base), ("scenario (shift)", scen)]
    for ax, (name, a) in zip(axes, panels):
        _draw_full_stack(ax, x, a, ren_day, ti)
        ax.axvspan(free_hours[0], free_hours[1], color="gold", alpha=0.18,
                   label="free window")
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_ylabel("MW")
    axes[2].set_xlim(0, 24)
    axes[2].set_xticks(range(0, 25, 3))
    axes[2].set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=8)
    axes[2].set_xlabel("time of day")
    axes[0].legend(loc="upper left", ncol=4, fontsize=7, framealpha=0.9)
    f.suptitle(f"{tag} — all-energy dispatch stack — {day_index[0].date()} "
               f"(battery charging shown negative)")
    f.tight_layout(); f.savefig(out, dpi=130); plt.close(f)


def fig_stacked_full(index, actual, pred, ren, out, tag):
    ti = {t: i for i, t in enumerate(TARGETS)}
    t = pd.DatetimeIndex(index).tz_localize(None)
    f, axes = plt.subplots(2, 1, figsize=(15, 9.5), sharex=True, sharey=True)
    for ax, (name, a) in zip(axes, [("actual", actual), ("predicted", pred)]):
        _draw_full_stack(ax, t, a, ren, ti)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_ylabel("MW")
    axes[0].legend(loc="upper left", ncol=5, fontsize=7, framealpha=0.9)
    f.suptitle(f"{tag} — all-energy dispatch stack (renewables + curtailment actual) — "
               f"full test set (battery charging shown negative)")
    f.tight_layout(); f.savefig(out, dpi=130); plt.close(f)
