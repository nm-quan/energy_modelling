"""4-day ALL-ENERGY stacked dispatch chart for a study scenario rollout.

Three panels (actual / baseline no-shift / scenario) over consecutive full test
days, matching the repo reference (stack_plots.py / hist_stack_shift_4day.py):
dispatchables stacked bottom-up, wind + solar with hatched curtailment bands on
top, battery charging below zero, the daily free window shaded gold. The
scenario/base net-demand input is drawn as a dashed line.

Renewable layers use the best available source:

  1. data/renewables_extract_hist.parquet — small committed extract; create it
     once on a machine with the raw parquets: script/export_renewables_extract.py
  2. the last365 raw parquets (stack_plots.load_renewables), local machines only
     (coverage ends 2026-05-18; days are picked inside coverage)
  3. derived single band: wind+solar ≈ max(demand_mw − net_demand, 0) from the
     model features (approximate — includes the ~197 MW rooftop/losses offset,
     no curtailment split); used when no parquet source exists (fresh clones)

Scenario + rollout are identical to study_shift.py (same flags incl. the SOC
fixes --fd-alloc / --soc-shield), so the figure matches the study tables.

    python3 demand_simulation/study_stack_4day.py --model itransformer_rayenfd
    python3 demand_simulation/study_stack_4day.py --fd-alloc share --soc-shield on
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import matplotlib.dates as mdates            # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(HERE))

import evaluate as ev                        # noqa: E402
import stack_plots as sp                     # noqa: E402
import study_shift as ss                     # noqa: E402

OUT = HERE / "sweep_eqnd" / "study" / "figure"
EXTRACT = ROOT / "data" / "renewables_extract_hist.parquet"


def load_renewables_any(test_index: pd.DatetimeIndex):
    """(ren DataFrame[wind, solar, wind_curt, solar_curt] or None, source tag).

    NaN rows (outside a parquet source's coverage) are kept as NaN so the day
    picker can avoid them; they are filled 0 just before drawing.
    """
    if EXTRACT.exists():
        df = pd.read_parquet(EXTRACT)
        if df.index.tz is None and test_index.tz is not None:
            df.index = df.index.tz_localize(test_index.tz)
        return df.reindex(test_index), "extract"
    try:
        return sp.load_renewables(test_index), "last365-parquets"
    except FileNotFoundError:
        return None, "derived"


def draw_derived_panel(ax, t, disp, ren_band, nd_line, ti):
    """Dispatch stack + one combined wind+solar band (no split/curtailment)."""
    base = np.zeros(len(t))
    for label, src, key, color, hatch in sp.STACK_ORDER[:5]:      # dispatchables
        top = base + disp[:, ti[key]]
        ax.fill_between(t, base, top, color=color, alpha=0.9, label=label)
        base = top
    ax.fill_between(t, base, base + ren_band, color=sp.WIND, alpha=0.55,
                    label="wind+solar (derived, approx)")
    ax.fill_between(t, 0, -disp[:, ti["battery_charging"]],
                    color=sp.COLORS["battery_charging"], alpha=0.6,
                    label="battery_charging (load)")
    if nd_line is not None:
        ax.plot(t, nd_line, color="k", lw=1.0, ls="--", label="net demand (input)")
    ax.axhline(0, color="k", lw=0.6)
    ax.margins(x=0)
    ax.grid(True, axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="itransformer_rayenfd",
                    help="substring selecting exactly one study_shift model")
    ap.add_argument("--scenario", choices=["reshape", "increase"], default="increase")
    ap.add_argument("--g", type=float, default=10.0)
    ap.add_argument("--demand-cap", type=float, default=ss.DEMAND_CAP_MW)
    ap.add_argument("--weights", default=str(ROOT / "weights"))
    ap.add_argument("--rayenfd-steam-pt", choices=["on", "off"], default="on")
    ap.add_argument("--fd-alloc", choices=["headroom", "invvar", "share"], default="headroom")
    ap.add_argument("--soc-shield", choices=["on", "off"], default="off")
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = ev.pick_device(args.device)

    frame = ss.load_frame()
    xs, ys, fc = frame["x_scaler"], frame["y_scaler"], frame["feat_cols"]
    lb, nd_idx = frame["lookback_steps"], fc.index("net_demand")
    fs_scen, info = ss.build_scenario(frame, args.scenario, args.g, args.demand_cap)

    entries = ss.load_entries(Path(args.weights), frame, device, [args.model],
                              steam_pt=args.rayenfd_steam_pt == "on",
                              fd_alloc=args.fd_alloc,
                              soc_shield=args.soc_shield == "on")
    if len(entries) != 1:
        raise SystemExit(f"--model {args.model!r} matches {[e[0] for e in entries]}; "
                         "narrow it to exactly one")
    name, n_out, d_ref, model = entries[0]

    ti_full = frame["test_index"]
    ren_full, ren_src = load_renewables_any(ti_full)
    pickable = ti_full if ren_full is None else \
        ti_full[~ren_full[["wind", "solar"]].isna().any(axis=1).to_numpy()]
    days, _ = sp.pick_full_days(pd.DatetimeIndex(pickable), n_days=args.days, seed=args.seed)
    pos = np.where(np.isin(ti_full.date, days))[0]
    end = int(pos.max()) + 1
    test_index = ti_full[:end]
    free_test = np.asarray((test_index.hour >= ss.FREE_HOURS[0])
                           & (test_index.hour < ss.FREE_HOURS[1]))
    print(f"model={name} scenario={args.scenario} g={info['g']:g}% renewables={ren_src} "
          f"days={[str(d) for d in days]} (rollout {end} steps)")

    t0 = time.time()
    pb, ps, _, _ = ss.rollout(model, n_out, d_ref, frame["fs_base"][:end + lb],
                              fs_scen[:end + lb], lb, free_test, nd_idx, xs, ys, device,
                              soc_shield=args.soc_shield == "on")
    print(f"  rollout done ({time.time() - t0:.0f}s)")
    actual = ys.inverse_transform(frame["Yte_flat"][lb:lb + end]).astype(np.float64)
    nd_base = (frame["fs_base"][lb:lb + end, nd_idx].astype(np.float64)
               * xs.scale_[nd_idx] + xs.mean_[nd_idx])
    nd_scen = info["nd_scen"][lb:lb + end]
    di = fc.index("demand_mw")
    dem = (frame["fs_base"][lb:lb + end, di].astype(np.float64)
           * xs.scale_[di] + xs.mean_[di])
    derived_band = np.clip(dem - nd_base, 0.0, None)

    ti = {t: i for i, t in enumerate(ss.TARGETS)}
    t = pd.DatetimeIndex(test_index[pos]).tz_localize(None)
    f, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, sharey=True)
    panels = [("actual", actual[pos], nd_base[pos]),
              ("baseline (no shift)", pb[pos], nd_base[pos]),
              (f"scenario ({args.scenario} g={info['g']:g}%)", ps[pos], nd_scen[pos])]
    if ren_full is not None:
        ren_win = ren_full.iloc[pos].fillna(0.0).clip(lower=0.0)
        for ax, (pname, disp, _nd) in zip(axes, panels):
            sp._draw_full_stack(ax, t, disp, ren_win, ti)
            ax.set_title(pname, loc="left", fontsize=11)
            ax.set_ylabel("MW")
    else:
        for ax, (pname, disp, nd_line) in zip(axes, panels):
            draw_derived_panel(ax, t, disp, derived_band[pos], nd_line, ti)
            ax.set_title(pname, loc="left", fontsize=11)
            ax.set_ylabel("MW")
    for ax in axes:
        for day in np.unique(t.normalize()):
            ax.axvspan(day + pd.Timedelta(hours=ss.FREE_HOURS[0]),
                       day + pd.Timedelta(hours=ss.FREE_HOURS[1]),
                       color="gold", alpha=0.15)
    axes[2].xaxis.set_major_locator(mdates.DayLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[2].xaxis.set_minor_locator(mdates.HourLocator(byhour=(6, 12, 18)))
    axes[2].set_xlabel("date")
    axes[0].legend(loc="upper left", ncol=5, fontsize=7, framealpha=0.9)
    ren_note = {"extract": "renewables + curtailment actual (extract)",
                "last365-parquets": "renewables + curtailment actual",
                "derived": "renewables derived from demand − net_demand (approx; "
                           "run script/export_renewables_extract.py for real layers)"}[ren_src]
    f.suptitle(f"{name} — all-energy dispatch stack, closed-loop free-window rollout — "
               f"{len(days)} days {t[0].date()} to {t[-1].date()}\n"
               f"({ren_note}; free window gold, battery charging negative)")
    f.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (f"stacked_4day_{name.replace('+', '_').replace('[', '_').replace(']', '')}"
                 f"_{args.scenario}_g{info['g']:g}.png")
    f.savefig(out, dpi=130)
    plt.close(f)
    print("wrote", out)


if __name__ == "__main__":
    main()
