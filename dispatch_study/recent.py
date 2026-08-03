"""Peak-filling patterns and curtailment-release rules for a RECENT window only.

The rest of dispatch_study/ pools 2022-2026. That is the wrong window for judging
a model whose test split is 2026: battery went from 4.8% to 33.1% of the evening
turn-up across those five years, and gas OCGT fell 17.1% to 4.8%. A five-year
average describes a fleet that no longer exists.

This recomputes everything on --from onward (default 2025) and prints the pooled
figure beside it so the drift is visible rather than hidden.

Writes patterns_<from>_<to>.md.

    python3 dispatch_study/recent.py            # 2025-2026
    python3 dispatch_study/recent.py --from 2026
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (load, in_hours, GEN, ALL_DISPATCH, ACTIVE_MW, LABEL, OUT,   # noqa: E402
                    DT, NIGHT_HOURS, MORNING_HOURS, TROUGH_HOURS, PEAK_HOURS,
                    COLORS, INK, MUTED)

SIGN = {"coal_brown": 1, "hydro": 1, "gas_ocgt": 1, "gas_steam": 1,
        "battery_discharging": 1, "battery_charging": -1}
HORIZONS = [(1, "5 min"), (6, "30 min"), (12, "1 hour"), (36, "3 hours")]
CURT = ["wind_curtailment", "solar_curtailment"]


def turnup(t, peak, base):
    """Peak mean minus the trough it rises out of, per fuel, and the shares."""
    mp = t.loc[in_hours(t.index, *peak), ALL_DISPATCH].mean()
    mb = t.loc[in_hours(t.index, *base), ALL_DISPATCH].mean()
    d = mp - mb
    d["battery_charging"] *= -1                 # a reduction supplies power
    return d, 100 * d / d.clip(lower=0).sum()


def slopes(t):
    """MW of each fuel per MW of net-demand change, by horizon."""
    ts = t.index.view("int64") // 10 ** 9
    nd = t["net_demand"].values
    out = {}
    for h, name in HORIZONS:
        ok = (ts[h:] - ts[:-h]) == h * 300
        dn = (nd[h:] - nd[:-h])[ok]
        v = dn.var()
        r = {f: float(np.cov((t[f].values[h:] - t[f].values[:-h])[ok], dn)[0, 1] / v)
             for f in ALL_DISPATCH}
        r["_sum"] = sum(SIGN[f] * r[f] for f in ALL_DISPATCH)
        out[name] = r
    return out


def curt_rules(t):
    """Release rules: the price split, the marginal rate, and the floor."""
    c = (t[CURT[0]] + t[CURT[1]]).values
    nd, p = t["net_demand"].values, t["price_aud_per_mwh"].values
    ts = t.index.view("int64") // 10 ** 9
    tot = c.sum() * DT
    neg = 100 * c[p < 0].sum() / max(c.sum(), 1e-9)
    k = {}
    for h, name in HORIZONS:
        ok = (ts[h:] - ts[:-h]) == h * 300
        dn = (nd[h:] - nd[:-h])[ok]
        dc = (c[h:] - c[:-h])[ok]
        base, pr = c[:-h][ok], p[:-h][ok]
        k[name] = {lab: float(np.cov(dc[m], dn[m])[0, 1] / dn[m].var())
                   for lab, m in (("all", np.ones(len(dn), bool)),
                                  ("active", base > 10),
                                  ("large", base > 200),
                                  ("negprice", pr < 0))}
    q = np.quantile(nd, np.linspace(0, 1, 11))
    dec = [(q[i], q[i + 1], c[(nd >= q[i]) & (nd <= q[i + 1])].mean(),
            100 * (c[(nd >= q[i]) & (nd <= q[i + 1])] > ACTIVE_MW).mean())
           for i in range(10)]
    return dict(total=tot, neg_share=neg, k=k, dec=dec)


def _axis(a, title, ylab=None):
    a.set_title(title, loc="left", fontsize=9.5, color=INK, pad=4)
    if ylab:
        a.set_ylabel(ylab, fontsize=8, color=MUTED)
    a.grid(True, axis="y", alpha=0.18, lw=0.6)
    a.tick_params(labelsize=8, colors=MUTED, length=0)
    for sp in ("top", "right", "left", "bottom"):
        a.spines[sp].set_visible(False)


def figures(rec, full, y0, y1, tag):
    """Two figures. Panels are grouped bars or lines on a single MW/% axis --
    never two scales -- with the pooled figure drawn pale beside the recent one so
    the drift is the visual, not a number you have to hold in your head."""
    cols = [COLORS[f] for f in ALL_DISPATCH]
    lab = [LABEL[f] for f in ALL_DISPATCH]
    x = np.arange(len(ALL_DISPATCH)); w = 0.38

    # ---------------- figure 1: peak composition ----------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for k, (nm, pk, bs) in enumerate((("EVENING", PEAK_HOURS, TROUGH_HOURS),
                                      ("MORNING", MORNING_HOURS, NIGHT_HOURS))):
        _, sr = turnup(rec, pk, bs)
        _, sf = turnup(full, pk, bs)
        ax[k].bar(x - w / 2, [sf[f] for f in ALL_DISPATCH], w, color=cols,
                  alpha=0.45, edgecolor="white", lw=1.2)
        ax[k].bar(x + w / 2, [sr[f] for f in ALL_DISPATCH], w, color=cols,
                  edgecolor="white", lw=1.2)
        ax[k].axhline(0, color=MUTED, lw=0.8)
        ax[k].set_xticks(x); ax[k].set_xticklabels(lab, fontsize=7.5,
                                                   rotation=20, ha="right")
        _axis(ax[k], f"{nm} peak turn-up share\n(pale = pooled 2022-{y1}, "
                     f"solid = {y0}-{y1})", "% of turn-up")
    yrs = sorted(full.index.year.unique())
    for f in ALL_DISPATCH:
        v = [turnup(full[full.index.year == y], PEAK_HOURS, TROUGH_HOURS)[1][f]
             for y in yrs]
        ax[2].plot(yrs, v, color=COLORS[f], lw=2, marker="o", ms=4)
        ax[2].annotate(LABEL[f], (yrs[-1], v[-1]), textcoords="offset points",
                       xytext=(6, 0), fontsize=7.5, color=COLORS[f], va="center")
    ax[2].set_xticks(yrs); ax[2].set_xlim(yrs[0] - 0.2, yrs[-1] + 1.4)
    _axis(ax[2], "Evening turn-up share by year\n(why pooling misleads)",
          "% of turn-up")
    fig.suptitle(f"Victoria {y0}-{y1} — who fills the peak, against the "
                 f"2022-{y1} average", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / f"patterns_{tag}_peaks.png", dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT/f'patterns_{tag}_peaks.png'}")

    # ---------------- figure 2: response + curtailment ----------------
    sr = slopes(rec); cr = curt_rules(rec)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    hn = [n for _, n in HORIZONS]; xs = np.arange(len(hn))
    for f in ALL_DISPATCH:
        v = [sr[n][f] * SIGN[f] for n in hn]
        ax[0].plot(xs, v, color=COLORS[f], lw=2, marker="o", ms=5)
        ax[0].annotate(LABEL[f], (xs[-1], v[-1]), textcoords="offset points",
                       xytext=(6, 0), fontsize=7.5, color=COLORS[f], va="center")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(hn, fontsize=8)
    ax[0].set_xlim(-0.15, len(hn) + 0.55)
    _axis(ax[0], "MW supplied per MW of net-demand change", "signed slope")

    for lab2, c_, st in (("price < 0", "#B03A2E", "-"), ("> 200 MW", "#A0661F", "-"),
                         ("active", "#2277BB", "--"), ("all steps", MUTED, ":")):
        key = {"price < 0": "negprice", "> 200 MW": "large",
               "active": "active", "all steps": "all"}[lab2]
        ax[1].plot(xs, [-cr["k"][n][key] for n in hn], color=c_, lw=2, ls=st,
                   marker="o", ms=5, label=lab2)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(hn, fontsize=8)
    ax[1].legend(fontsize=7.5, frameon=False, title="conditioned on",
                 title_fontsize=7.5)
    _axis(ax[1], "Curtailment RELEASED per MW of net-demand rise", "k")

    d = cr["dec"]
    ax[2].bar(range(1, 11), [r[2] for r in d], color=COLORS["wind"],
              edgecolor="white", lw=1.2)
    ax[2].set_yscale("log")
    ax[2].set_xticks(range(1, 11)); ax[2].set_xlabel("net-demand decile",
                                                     fontsize=8, color=MUTED)
    for i, r in enumerate(d):
        ax[2].annotate(f"{r[2]:,.0f}", (i + 1, r[2]), ha="center", va="bottom",
                       fontsize=6.5, color=MUTED)
    _axis(ax[2], "Curtailment available to release, by net demand\n"
                 "(log scale — 165x from D1 to D10)", "mean MW")
    fig.suptitle(f"Victoria {y0}-{y1} — marginal response and the curtailment "
                 f"release rule", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / f"patterns_{tag}_response.png", dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT/f'patterns_{tag}_response.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="y0", type=int, default=2025)
    args = ap.parse_args()
    full = load()
    rec = full[full.index.year >= args.y0]
    y1 = int(rec.index.year.max())
    tag = f"{args.y0}_{y1}"
    print(f"recent window {args.y0}-{y1}: {len(rec):,} intervals "
          f"({len(rec)//288} days)  vs pooled {len(full):,}")

    L = [f"# Peak-filling patterns and curtailment rules — {args.y0}–{y1}", "",
         f"{len(rec):,} five-minute intervals ({len(rec)//288} days), "
         f"{rec.index[0].date()} → {rec.index[-1].date()}, local AEST. "
         f"Every table shows the **pooled 2022–{y1}** figure alongside, because the "
         "fleet changed enough over that span that the pooled number describes a "
         "system that no longer exists.", "",
         "Windows, all measured from the mean hourly profile: overnight trough "
         f"{NIGHT_HOURS[0]:02d}–{NIGHT_HOURS[1]:02d}, morning peak "
         f"{MORNING_HOURS[0]:02d}–{MORNING_HOURS[1]:02d}, midday trough "
         f"{TROUGH_HOURS[0]:02d}–{TROUGH_HOURS[1]:02d}, evening peak "
         f"{PEAK_HOURS[0]:02d}–{PEAK_HOURS[1]:02d}.", ""]

    # ---------- 1. peak turn-up ----------
    for nm, pk, bs in (("EVENING", PEAK_HOURS, TROUGH_HOURS),
                       ("MORNING", MORNING_HOURS, NIGHT_HOURS)):
        dr, sr = turnup(rec, pk, bs)
        df, sf = turnup(full, pk, bs)
        L += [f"## {nm} peak — who is turned up", "",
              f"| fuel | {args.y0}–{y1} MW | share | pooled share | change |",
              "| --- | ---: | ---: | ---: | ---: |"]
        for f_ in ALL_DISPATCH:
            L.append(f"| {LABEL[f_]} | {dr[f_]:+,.0f} | **{sr[f_]:.1f}%** | "
                     f"{sf[f_]:.1f}% | {sr[f_]-sf[f_]:+.1f} |")
        L += [f"| **total turn-up** | **{dr.clip(lower=0).sum():,.0f} MW** | | "
              f"{df.clip(lower=0).sum():,.0f} MW | |", ""]

    # ---------- 2. marginal response ----------
    sr, sf = slopes(rec), slopes(full)
    L += ["## Marginal response — MW of each fuel per MW of net-demand change", "",
          "This is the number a demand-shift counterfactual should reproduce. Read "
          "the 3-hour row for a 3-hour window.", "",
          "| horizon | " + " | ".join(LABEL[f] for f in ALL_DISPATCH) + " | Σ signed |",
          "| --- |" + " ---: |" * (len(ALL_DISPATCH) + 1)]
    for _, n in HORIZONS:
        L.append(f"| **{n}** | " + " | ".join(f"{sr[n][f]:+.3f}" for f in ALL_DISPATCH)
                 + f" | {sr[n]['_sum']:.3f} |")
    L += ["", f"Pooled 2022–{y1} for comparison:", "",
          "| horizon | " + " | ".join(LABEL[f] for f in ALL_DISPATCH) + " | Σ signed |",
          "| --- |" + " ---: |" * (len(ALL_DISPATCH) + 1)]
    for _, n in HORIZONS:
        L.append(f"| {n} | " + " | ".join(f"{sf[n][f]:+.3f}" for f in ALL_DISPATCH)
                 + f" | {sf[n]['_sum']:.3f} |")
    n3 = "3 hours"
    tot_r = sum(abs(sr[n3][f]) for f in ALL_DISPATCH)
    L += ["", f"**Normalised 3-hour shares, {args.y0}–{y1}** — the benchmark column "
          "for a counterfactual table:", "",
          "| " + " | ".join(LABEL[f] for f in ALL_DISPATCH) + " |",
          "|" + " ---: |" * len(ALL_DISPATCH),
          "| " + " | ".join(f"**{100*abs(sr[n3][f])/tot_r:.1f}%**"
                            for f in ALL_DISPATCH) + " |", ""]

    # ---------- 3. curtailment ----------
    cr, cf = curt_rules(rec), curt_rules(full)
    L += ["## Curtailment release rules", "",
          f"Total curtailment {args.y0}–{y1}: **{cr['total']:,.0f} MWh**. "
          f"Share at negative price (economic, releasable): **{cr['neg_share']:.1f}%** "
          f"(pooled {cf['neg_share']:.1f}%). The remainder is network-bound — "
          "congestion and system strength — and does not go away when demand rises.",
          "", "### Marginal release rate (MW released per MW of net-demand rise)", "",
          f"| horizon | all steps | curtailment active | > 200 MW | price < 0 |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for _, n in HORIZONS:
        r = cr["k"][n]
        L.append(f"| {n} | {r['all']:+.3f} | {r['active']:+.3f} | "
                 f"{r['large']:+.3f} | {r['negprice']:+.3f} |")
    k3 = cr["k"][n3]
    L += ["", "### The rule", "",
          "For a counterfactual raising net demand by Δ at a step with curtailment C "
          "and price p:", "", "```",
          "released  =  min( C,  k · Δ )", "",
          f"k = {abs(k3['large']):.2f}   if C > 200 MW and p < 0",
          f"k = {abs(k3['active']):.2f}   if C >  10 MW",
          f"k = {abs(k3['all']):.2f}   otherwise", "",
          f"retained  >=  {100-cr['neg_share']:.3f} % of C      the network-bound floor",
          "```", "",
          f"(pooled 2022–{y1} gave {abs(cf['k'][n3]['large']):.2f} / "
          f"{abs(cf['k'][n3]['active']):.2f} / {abs(cf['k'][n3]['all']):.2f} and a "
          f"{100-cf['neg_share']:.1f}% floor.)", "",
          "### The floor is steep in net demand", "",
          f"| net-demand decile | mean curtailment | active |",
          "| --- | ---: | ---: |"]
    for i, (lo, hi, mu, act) in enumerate(cr["dec"]):
        L.append(f"| D{i+1} ({lo:,.0f}–{hi:,.0f} MW) | {mu:,.0f} MW | {act:.0f}% |")
    L += ["", "When net demand is already high there is almost nothing left to "
          "release, which is why raising demand at the evening peak frees very "
          "little.", ""]

    figures(rec, full, args.y0, y1, tag)
    L += ["## Figures", "",
          f"- `patterns_{tag}_peaks.png` — turn-up share, morning and evening, "
          f"{args.y0}-{y1} against the pooled average, plus the year-by-year drift",
          f"- `patterns_{tag}_response.png` — marginal response by horizon, the "
          "curtailment release rate, and how little is left to release at high "
          "net demand", ""]
    (OUT / f"patterns_{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT/f'patterns_{tag}.md'}")


if __name__ == "__main__":
    main()
