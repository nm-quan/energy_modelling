"""Deliverables 4 and 5 — two studies that say how the grid actually responds.

STUDY A — MARGINAL RESPONSE ("when net demand rises by 1 MW, who supplies it?")
  For horizons h = 5 min, 30 min, 1 h and 3 h, take every pair (t, t+h), form
  d_nd = nd(t+h) - nd(t) and d_f = fuel(t+h) - fuel(t), and fit the OLS slope
  b_f = cov(d_f, d_nd) / var(d_nd). b_f is the MW of fuel f that moves per MW of
  net-demand change -- the empirical merit order for CHANGES, as opposed to the
  merit order for levels. Sum(SIGN * b_f) should be near 1: the shortfall is the
  interconnector. The 3 h horizon is the one to compare a 3 h counterfactual to.

STUDY B — ACTIVATION ("what turns each fuel on?")
  For each fuel, P(active | net-demand decile) and P(active | price decile), plus
  the switch-on level: the net demand at which P(active) first crosses 50%. This
  is what "signs that gas gets used" means concretely -- a threshold in MW and in
  $/MWh, measured rather than assumed. Battery gets its own panel because it has
  two states, and the question is when it flips.

Writes response.md and response.png.

    python3 dispatch_study/response.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (load, in_hours, FUELS, ALL_DISPATCH, ACTIVE_MW, OUT,   # noqa: E402
                    COLORS, LABEL, INK, MUTED, PEAK_HOURS)

SIGN = {"coal_brown": 1, "hydro": 1, "gas_ocgt": 1, "gas_steam": 1,
        "battery_discharging": 1, "battery_charging": -1}
HORIZONS = [(1, "5 min"), (6, "30 min"), (12, "1 hour"), (36, "3 hours")]


def study_a(t: pd.DataFrame):
    """Marginal response slopes by horizon. Contiguity is enforced by requiring
    the timestamp difference to be exactly h*5 min, so day/data gaps are dropped."""
    out = {}
    ts = t.index.view("int64") // 10**9
    for h, name in HORIZONS:
        ok = (ts[h:] - ts[:-h]) == h * 300
        dnd = (t["net_demand"].values[h:] - t["net_demand"].values[:-h])[ok]
        v = dnd.var()
        row = {}
        for f in ALL_DISPATCH:
            df = (t[f].values[h:] - t[f].values[:-h])[ok]
            row[f] = float(np.cov(df, dnd)[0, 1] / v)
        row["_n"] = int(ok.sum())
        row["_closure"] = float(sum(SIGN[f] * row[f] for f in ALL_DISPATCH))
        out[name] = row
    return out


def study_b(t: pd.DataFrame, nbins: int = 10):
    """P(active | net-demand decile) and | price decile, plus switch-on levels."""
    nd, pr = t["net_demand"].values, t["price_aud_per_mwh"].values
    qn = np.quantile(nd, np.linspace(0, 1, nbins + 1))
    qp = np.quantile(pr, np.linspace(0, 1, nbins + 1))
    bn = np.clip(np.digitize(nd, qn[1:-1]), 0, nbins - 1)
    bp = np.clip(np.digitize(pr, qp[1:-1]), 0, nbins - 1)
    res = {"nd_edges": qn, "price_edges": qp, "by_nd": {}, "by_price": {}, "switch": {}}
    for f in ALL_DISPATCH:
        a = t[f].values > ACTIVE_MW
        pn = np.array([a[bn == i].mean() for i in range(nbins)])
        pp = np.array([a[bp == i].mean() for i in range(nbins)])
        res["by_nd"][f] = pn
        res["by_price"][f] = pp
        # switch-on level: first decile whose midpoint crosses 50% active
        hit = np.where(pn >= 0.5)[0]
        res["switch"][f] = (float((qn[hit[0]] + qn[hit[0] + 1]) / 2) if len(hit) else np.nan,
                            float(np.median(pr[a])) if a.any() else np.nan,
                            float(np.median(nd[a])) if a.any() else np.nan)
    return res


def battery_state(t: pd.DataFrame):
    """When does the battery flip? Share of steps charging / discharging / idle,
    by hour and by price quintile."""
    chg = t["battery_charging"].values > ACTIVE_MW
    dis = t["battery_discharging"].values > ACTIVE_MW
    hr = t.index.hour
    by_hour = pd.DataFrame({"charging": [chg[hr == h].mean() for h in range(24)],
                            "discharging": [dis[hr == h].mean() for h in range(24)]},
                           index=range(24))
    pr = t["price_aud_per_mwh"].values
    q = np.quantile(pr, np.linspace(0, 1, 6))
    b = np.clip(np.digitize(pr, q[1:-1]), 0, 4)
    by_price = pd.DataFrame({"charging": [chg[b == i].mean() for i in range(5)],
                             "discharging": [dis[b == i].mean() for i in range(5)],
                             "price_lo": q[:-1], "price_hi": q[1:]}, index=range(5))
    return by_hour, by_price


def main():
    t = load()
    A = study_a(t)
    B = study_b(t)
    bh, bp = battery_state(t)

    L = ["# How the Victorian grid actually responds — two studies", "",
         f"Source: {len(t):,} five-minute intervals, {t.index[0].date()} to "
         f"{t.index[-1].date()}, local (AEST) time. Descriptive only, no model.", "",
         "## Study A — marginal response: who supplies the next MW?", "",
         "OLS slope of Δfuel on Δnet-demand over each horizon: **MW of that fuel per "
         "MW of net-demand change**. This is the merit order for *changes*, which is "
         "the thing a demand-shift study needs — as opposed to the merit order for "
         "*levels*, which just says coal is big.", "",
         "| horizon | " + " | ".join(LABEL[f] for f in ALL_DISPATCH) +
         " | Σ (signed) | n pairs |",
         "| --- |" + " ---: |" * (len(ALL_DISPATCH) + 2)]
    for _, name in HORIZONS:
        r = A[name]
        L.append(f"| {name} | " + " | ".join(f"{r[f]:+.3f}" for f in ALL_DISPATCH) +
                 f" | {r['_closure']:.3f} | {r['_n']:,} |")
    L += ["", "_Battery charging carries a negative sign in the balance, so a negative "
          "slope there means charging FALLS as net demand rises — which supplies power. "
          "Σ (signed) below 1 is the interconnector: the rest of the change is imported "
          "or exported rather than met by the local fleet._", ""]

    L += ["## Study B — activation: what turns each fuel on?", "",
          f"Active = output above {ACTIVE_MW:.0f} MW. Switch-on level = the net demand at "
          "which the fuel is active more than half the time.", "",
          "| fuel | switch-on net demand (MW) | median net demand when active (MW) | "
          "median price when active ($/MWh) | overall active |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for f in ALL_DISPATCH:
        sw, mp, mn = B["switch"][f]
        act = 100 * (t[f].values > ACTIVE_MW).mean()
        swv = "always" if sw <= B["nd_edges"][1] else (f"{sw:,.0f}" if np.isfinite(sw) else "never >50%")
        L.append(f"| {LABEL[f]} | {swv} | {mn:,.0f} | {mp:,.0f} | {act:.1f}% |")
    L.append("")

    L += ["### P(active) by net-demand decile", "",
          "| decile (net demand MW) | " + " | ".join(LABEL[f] for f in ALL_DISPATCH) + " |",
          "| --- |" + " ---: |" * len(ALL_DISPATCH)]
    for i in range(10):
        lo, hi = B["nd_edges"][i], B["nd_edges"][i + 1]
        L.append(f"| D{i+1} ({lo:,.0f}–{hi:,.0f}) | " +
                 " | ".join(f"{100*B['by_nd'][f][i]:.0f}%" for f in ALL_DISPATCH) + " |")
    L.append("")

    L += ["### Battery state by hour (% of steps)", "",
          "| hour | charging | discharging |", "| --- | ---: | ---: |"]
    for h in range(24):
        L.append(f"| {h:02d}:00 | {100*bh.loc[h,'charging']:.0f}% | "
                 f"{100*bh.loc[h,'discharging']:.0f}% |")
    L += ["", "### Battery state by price quintile (% of steps)", "",
          "| price quintile ($/MWh) | charging | discharging |", "| --- | ---: | ---: |"]
    for i in range(5):
        L.append(f"| Q{i+1} ({bp.loc[i,'price_lo']:,.0f} to {bp.loc[i,'price_hi']:,.0f}) | "
                 f"{100*bp.loc[i,'charging']:.0f}% | {100*bp.loc[i,'discharging']:.0f}% |")
    L.append("")

    (OUT / "response.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # A: marginal slope by horizon
    xs = np.arange(len(HORIZONS))
    ends = sorted(((A[HORIZONS[-1][1]][f] * SIGN[f], f) for f in ALL_DISPATCH),
                  reverse=True)
    # stagger direct labels so near-equal end values do not collide
    span = ends[0][0] - ends[-1][0]
    place, last = {}, None
    for v, f in ends:
        y = v if last is None else min(v, last - 0.035 * span)
        place[f] = y; last = y
    for f in ALL_DISPATCH:
        v = [A[n][f] * SIGN[f] for _, n in HORIZONS]
        ax[0].plot(xs, v, color=COLORS[f], lw=2, marker="o", ms=5, label=LABEL[f])
        ax[0].text(xs[-1] + 0.08, place[f], LABEL[f], fontsize=7.5,
                   color=COLORS[f], va="center", ha="left")
    ax[0].set_xticks(xs); ax[0].set_xticklabels([n for _, n in HORIZONS], fontsize=8)
    ax[0].set_title("A. MW supplied per MW of net-demand change",
                    loc="left", fontsize=10, color=INK)
    ax[0].set_ylabel("signed slope", fontsize=8, color=MUTED)
    ax[0].set_xlim(-0.15, len(HORIZONS) + 0.35)

    # B1: P(active) by nd decile
    for f in ALL_DISPATCH:
        ax[1].plot(np.arange(1, 11), 100 * B["by_nd"][f], color=COLORS[f], lw=2,
                   marker="o", ms=4, label=LABEL[f])
    ax[1].set_xticks(range(1, 11)); ax[1].set_xlabel("net-demand decile", fontsize=8, color=MUTED)
    ax[1].set_ylabel("% of steps active", fontsize=8, color=MUTED)
    ax[1].set_title("B. Activation vs net demand", loc="left", fontsize=10, color=INK)
    ax[1].legend(fontsize=7, frameon=False, ncol=2, loc="center left")

    # B2: battery by hour
    ax[2].fill_between(range(24), 0, 100 * bh["charging"].values,
                       color=COLORS["battery_charging"], alpha=0.85, label="charging")
    ax[2].fill_between(range(24), 0, -100 * bh["discharging"].values,
                       color=COLORS["battery_discharging"], alpha=0.85, label="discharging")
    ax[2].axhline(0, color=MUTED, lw=0.8)
    ax[2].axvspan(PEAK_HOURS[0], PEAK_HOURS[1], color="#000000", alpha=0.05)
    ax[2].set_xticks([0, 6, 12, 18, 23]); ax[2].set_xlabel("hour", fontsize=8, color=MUTED)
    ax[2].set_ylabel("% of steps  (down = discharging)", fontsize=8, color=MUTED)
    ax[2].set_title("B. Battery state by hour", loc="left", fontsize=10, color=INK)
    ax[2].legend(fontsize=7.5, frameon=False, loc="upper right")
    for a in ax:
        a.grid(True, alpha=0.18, lw=0.6)
        a.tick_params(labelsize=8, colors=MUTED, length=0)
        for s in ("top", "right", "left", "bottom"):
            a.spines[s].set_visible(False)
    fig.suptitle("Victoria 2022–2026 — how dispatch responds", fontsize=12,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "response.png", dpi=140, facecolor="white"); plt.close(fig)
    print(f"\nwrote {OUT/'response.md'} and {OUT/'response.png'}")


if __name__ == "__main__":
    main()
