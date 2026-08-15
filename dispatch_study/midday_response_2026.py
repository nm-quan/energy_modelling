"""Who serves an increase in DEMAND during the 11:00-14:00 midday trough, 2026?

Method is Study A from response.py, with three changes the question forces:

  1. The regressor is d(demand), not d(net_demand). net_demand IS the signed sum
     of the six dispatchable channels, so regressing on it closes to 1.000 by
     identity and says nothing about renewables. Demand is the exogenous thing
     the question perturbs.
  2. Both endpoints of a pair must lie inside the window, so the slope is the
     response *within* the midday trough, not the trough-to-evening ramp. The
     window is 3h wide, so the 3h horizon has no valid pairs and is dropped;
     the 4h evening window keeps it.
  3. Renewables and curtailment are carried as channels. At midday they are most
     of the answer, and dropping them would force the response onto thermal by
     construction.

TWO FRAMINGS, because lib/pipeline.py:27 sets demand_mw := demand_mw -
net_import_mw. The interconnector is already inside demand_mw.

  CLOSED  regress on demand_mw as stored. "Demand" then means VIC load net of
          imports, so a rise is either more load or fewer net imports, and only
          the local fleet can answer. This is the world the models are trained
          in, and the signed slopes close to ~1.00 because
          demand_mw == local generation excl rooftop to within 197 MW.
  OPEN    regress on demand_mw + net_import_mw, i.e. VIC consumer demand as the
          market reports it, and carry net import as a responding channel. This
          is the physical answer for a real demand increase in Victoria.

    python3 dispatch_study/midday_response_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TABLE, OUT, LABEL                                   # noqa: E402

MIDDAY = (11, 14)
EVENING = (17, 21)
HORIZONS = [(1, "5 min"), (6, "30 min"), (12, "1 hour"), (36, "3 hours")]

LOCAL = ["coal_brown", "gas_ocgt", "gas_steam", "hydro",
         "battery_discharging", "battery_charging", "wind", "solar_utility"]
SIGN = {c: +1 for c in LOCAL} | {"battery_charging": -1, "net_import_mw": +1}
MECH = ["wind_curtailment", "solar_curtailment"]

NAME = dict(LABEL)
NAME.update({"net_import_mw": "net import", "wind_curtailment": "wind curtailed",
             "solar_curtailment": "solar curtailed"})


def load() -> pd.DataFrame:
    t = pd.read_parquet(TABLE)
    ic = pd.read_parquet(ROOT / "data/vic_interconnector_20211001_20260706.parquet")
    t = t.join(ic.set_index("interval")["net_import_mw"], how="left")
    # undo the pipeline's netting to recover VIC consumer demand
    t["demand_open"] = t["demand_mw"] + t["net_import_mw"]
    return t


def window(t: pd.DataFrame, hours: tuple[int, int], year: int) -> pd.DataFrame:
    h = t.index.hour
    return t[(t.index.year == year) & (h >= hours[0]) & (h < hours[1])]


def slopes(t, cols, dcol="demand_mw", mask=None):
    """OLS slope of d(col) on d(demand) per horizon. Pairs must be exactly
    h*5min apart, which keeps both endpoints in the window and the same day."""
    ts = t.index.view("int64") // 10**9
    dem = t[dcol].to_numpy()
    out = {}
    for h, name in HORIZONS:
        ok = (ts[h:] - ts[:-h]) == h * 300
        if mask is not None:
            ok &= mask[:-h]
        dd = (dem[h:] - dem[:-h])[ok]
        fin = np.isfinite(dd)
        if ok.sum() < 200 or fin.sum() < 200 or np.nanvar(dd) == 0:
            out[name] = None
            continue
        row = {}
        for c in cols:
            v = t[c].to_numpy()
            dc = (v[h:] - v[:-h])[ok]
            g = np.isfinite(dc) & fin
            row[c] = float(np.cov(dc[g], dd[g])[0, 1] / dd[g].var())
        row["_n"] = int(fin.sum())
        row["_closure"] = float(sum(SIGN[c] * row[c] for c in cols if c in SIGN))
        out[name] = row
    return out


def table(res, cols, title, note=""):
    L = [title, ""]
    if note:
        L += [note, ""]
    L += ["| horizon | " + " | ".join(NAME[c] for c in cols) + " | Σ signed | n pairs |",
          "| --- |" + " ---: |" * (len(cols) + 2)]
    for _, name in HORIZONS:
        r = res.get(name)
        if r is None:
            continue
        L.append(f"| {name} | " +
                 " | ".join(f"{SIGN.get(c,1) * r[c]:+.3f}" for c in cols) +
                 f" | {r['_closure']:.3f} | {r['_n']:,} |")
    return L + [""]



# --------------------------------------------------------------------- figure
BUCKETS = [("coal", ["coal_brown"], "#eb6834"),
           ("battery\n(stops charging\n+ discharges)",
            ["battery_charging", "battery_discharging"], "#2a78d6"),
           ("renewables\n(spilled less)", ["wind", "solar_utility"], "#1baf7a"),
           ("hydro + gas", ["hydro", "gas_ocgt", "gas_steam"], "#eda100")]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1", "#fcfcfb"


def figure(bars, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "font.size": 9, "text.color": INK})
    fig, ax = plt.subplots(figsize=(10.4, 3.4))
    labels = [b[0] for b in bars]
    y = np.arange(len(bars))[::-1]
    left = np.zeros(len(bars))
    for name, _, col in BUCKETS:
        v = np.array([b[1][name] for b in bars]) * 100
        ax.barh(y, v, left=left, height=0.62, color=col, zorder=3,
                edgecolor=SURF, linewidth=2)
        for yi, (l, vi) in enumerate(zip(left, v)):
            if vi >= 6:
                ax.text(l + vi / 2, y[yi], f"{vi:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold", zorder=4)
        left += v
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 104); ax.set_xlabel("share of the demand increase (%)",
                                       fontsize=9, color=INK2)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    handles = [mpl.patches.Patch(facecolor=c, label=n.replace("\n", " "))
               for n, _, c in BUCKETS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncols=4, frameon=False, fontsize=8.5)
    fig.suptitle("Who serves the next MW of demand — VIC 2026, 1-hour horizon",
                 fontsize=12, fontweight="bold", x=0.012, ha="left", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main():
    t = load()
    mid = window(t, MIDDAY, 2026)
    eve = window(t, EVENING, 2026)
    OPEN = LOCAL + ["net_import_mw"]

    pr, curt = mid["price_aud_per_mwh"], mid["wind_curtailment"] + mid["solar_curtailment"]
    L = ["# Who serves extra demand at 11:00–14:00? — VIC, 2026", "",
         f"{len(mid):,} five-minute intervals, {mid.index[0].date()} to "
         f"{mid.index[-1].date()}. Descriptive, no model.", "",
         "Every cell is **MW of that channel per MW of demand increase** — the OLS "
         "slope of Δchannel on Δdemand at that horizon. Battery charging is signed "
         "negative, so a positive entry under *battery in* means charging **falls**, "
         "which is how a load supplies power.", "",
         "## The state of the window", "",
         f"- mean price **${pr.mean():,.0f}/MWh**, negative on **{(pr < 0).mean()*100:.0f}%** "
         f"of intervals and under $10 on {(pr < 10).mean()*100:.0f}%",
         f"- mean demand {mid.demand_mw.mean():,.0f} MW — the daily minimum",
         f"- renewables curtailed on **{(curt > 10).mean()*100:.0f}%** of intervals, "
         f"mean {curt.mean():,.0f} MW, max {curt.max():,.0f} MW",
         f"- battery charging {mid.battery_charging.mean():,.0f} MW mean against "
         f"{mid.battery_discharging.mean():,.0f} MW discharging",
         f"- VIC exports {-mid.net_import_mw.mean():,.0f} MW net on average", "",
         "The 3-hour horizon is absent at midday because the window is itself 3 hours "
         "wide, so no pair of endpoints 3 hours apart fits inside it.", ""]

    L += table(slopes(mid, LOCAL), LOCAL,
               "## A. Closed framing — the VIC fleet only",
               "`demand_mw` already has net imports subtracted "
               "(lib/pipeline.py:27), so this is the world the models see and the "
               "signed slopes close to ~1.")

    spill = (curt > 100).to_numpy()
    L += table(slopes(mid, LOCAL, mask=spill), LOCAL,
               f"### A1. When renewables are being spilled (curtailment > 100 MW, "
               f"{spill.mean()*100:.0f}% of the window)")
    L += table(slopes(mid, LOCAL, mask=~spill), LOCAL,
               f"### A2. When nothing is being spilled ({(~spill).mean()*100:.0f}% "
               f"of the window)")

    L += table(slopes(mid, OPEN, dcol="demand_open"), OPEN,
               "## B. Open framing — VIC consumer demand, interconnector free to move",
               "Regressor is `demand_mw + net_import_mw`, the market's VIC demand.")

    L += table(slopes(eve, LOCAL), LOCAL,
               "## C. Contrast — same closed regression at the 17:00–21:00 evening peak")

    L += ["## D. Mechanism — what happens to curtailment itself", "",
          "Slope of Δcurtailment on Δdemand, closed framing. Negative means the "
          "extra demand is served by spilling less.", "",
          "| condition | horizon | " + " | ".join(NAME[c] for c in MECH) + " | total |",
          "| --- | --- |" + " ---: |" * (len(MECH) + 1)]
    for lbl, msk in [("all midday", None), ("spilling", spill)]:
        rm = slopes(mid, MECH, mask=msk)
        for _, name in HORIZONS:
            r = rm.get(name)
            if r is None:
                continue
            tot = sum(r[c] for c in MECH)
            L.append(f"| {lbl} | {name} | " +
                     " | ".join(f"{r[c]:+.3f}" for c in MECH) + f" | {tot:+.3f} |")
    L.append("")

    bars = []
    for lbl, res, cols in [("11–14  all", slopes(mid, LOCAL), LOCAL),
                           ("11–14  spilling", slopes(mid, LOCAL, mask=spill), LOCAL),
                           ("11–14  no spill", slopes(mid, LOCAL, mask=~spill), LOCAL),
                           ("17–21  evening peak", slopes(eve, LOCAL), LOCAL)]:
        r = res["1 hour"]
        bars.append((lbl, {n: sum(SIGN[c] * r[c] for c in cs)
                           for n, cs, _ in BUCKETS}))
    figure(bars, OUT / "midday_response_2026.png")

    (OUT / "midday_response_2026.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT/'midday_response_2026.md'}")


if __name__ == "__main__":
    main()
