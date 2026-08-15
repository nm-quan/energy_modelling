"""Per-DAY accuracy for the evening-peak fill — no averaging across days.

    python3 planB/peak_days.py --arms hardnet_2025_alloc_sup --hour 18 --days 6

peak_eval.py reports statistics pooled over every test day and draws a
day-averaged stack. Averaging hides the thing that matters: whether the fill is
right on the days it is hard. This reports each day on its own, and draws the
same days with their own numbers on them.

MAE  mean |pred - actual| over the 36 window steps x 6 dispatch channels, MW.
MRE  sum|err| / sum|actual| per channel, then meaned -- the same definition
     peak_eval uses, so the numbers are comparable. It is unstable on channels
     that are mostly zero (gas steam is <=1 MW in 94% of window cells), so the
     per-channel table names the offenders rather than hiding them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "imputation"))
sys.path.insert(0, str(ROOT / "dispatch_study"))
sys.path.insert(0, str(HERE))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "prepared.npz"
from gap_data import load_flats, TARGETS                               # noqa: E402
from common import COLORS, LABEL, INK, MUTED, FUELS                    # noqa: E402
from peak_eval import peak_starts, run_arm                             # noqa: E402
from fit import build, CTX, GAP                                        # noqa: E402

OUT = ROOT / "planB" / "results"
SHORT = {"hydro": "hydro", "coal_brown": "coal", "gas_steam": "steam",
         "gas_ocgt": "ocgt", "battery_charging": "bat_chg",
         "battery_discharging": "bat_dis"}


def draw_day(ax, disp, rows, curt, nd, hour):
    x = np.arange(len(disp)) * 5.0 / 60.0 + hour
    ti = {t: i for i, t in enumerate(TARGETS)}
    base = np.zeros(len(x))
    for f_ in FUELS:
        top = base + disp[:, ti[f_]]
        ax.fill_between(x, base, top, color=COLORS[f_], lw=0.2, edgecolor="white",
                        zorder=2, label=LABEL[f_])
        base = top
    for k, col in enumerate(("wind", "solar_utility")):
        top = base + rows[:, k]
        ax.fill_between(x, base, top, color=COLORS[col], lw=0.2, edgecolor="white",
                        zorder=2, label=LABEL[col])
        base = top
    for k, col in enumerate(("wind", "solar_utility")):
        top = base + curt[:, k]
        ax.fill_between(x, base, top, color=COLORS[col], alpha=0.33, lw=0,
                        hatch="///", zorder=2, label=f"{LABEL[col]} curtailed")
        base = top
    ax.fill_between(x, 0, -disp[:, ti["battery_charging"]],
                    color=COLORS["battery_charging"], lw=0.2, edgecolor="white", zorder=2,
                    label=LABEL["battery_charging"])
    ax.plot(x, nd, color="#B3261E", lw=1.1, zorder=4, label="net demand")
    ax.margins(x=0); ax.grid(True, axis="y", alpha=0.15, lw=0.5, zorder=0)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["hardnet_2025_alloc_sup"])
    ap.add_argument("--hour", type=int, default=18)
    ap.add_argument("--days", type=int, default=6, help="how many days to DRAW")
    ap.add_argument("--from-day", default=None)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    f = load_flats(); nfeat = len(f.feat_cols)
    starts, days = peak_starts(f, args.hour)
    b = build(f, starts, "test")
    n = len(starts)
    end_h = args.hour + GAP * 5 // 60
    print(f"{n} test days, gap {args.hour:02d}:00-{end_h:02d}:00 "
          f"({days[0].date()} .. {days[-1].date()})", flush=True)

    truth = np.concatenate([b["Y"] * f.y_scale + f.y_mean, b["C"]], -1)   # (n,G,8)
    nd = b["nd"]
    tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
    cm, cs = b["c_mean"], b["c_scale"]
    cL = f.Xte[starts + CTX - 1][:, [19, 20]] * cs + cm
    cR = f.Xte[starts + CTX + GAP][:, [19, 20]] * cs + cm
    preds = {"interp": np.concatenate(
        [b["pL"][:, None] + tt * (b["pR"] - b["pL"])[:, None],
         cL[:, None] + tt * (cR - cL)[:, None]], -1)}
    for a in args.arms:
        if (OUT / f"{a}.pt").exists():
            preds[a], _ = run_arm(a, f, b, nfeat)
        else:
            print(f"  [skip] {a}")

    rows = ["interp"] + [a for a in args.arms if a in preds]
    # per-day MAE (dispatch) and per-day MRE (dispatch), no pooling
    mae_d = {r: np.abs(preds[r][..., :6] - truth[..., :6]).mean((1, 2)) for r in rows}
    mre_d = {r: 100 * (np.abs(preds[r][..., :6] - truth[..., :6]).sum(1)
                       / np.maximum(np.abs(truth[..., :6]).sum(1), 1.0)).mean(1)
             for r in rows}
    mae_c = {r: np.abs(preds[r][..., :6] - truth[..., :6]).mean(1) for r in rows}  # (n,6)
    # WAPE pools numerator and denominator over all six channels, so a channel
    # that was OFF all day cannot divide by ~zero. MRE above averages per-channel
    # ratios and DOES blow up: on 2026-01-06 gas steam ran 0.0 MW for the whole
    # window, giving that one channel 5,633% and dragging the mean to 967%.
    wape_d = {r: 100 * np.abs(preds[r][..., :6] - truth[..., :6]).sum((1, 2))
              / np.maximum(np.abs(truth[..., :6]).sum((1, 2)), 1.0) for r in rows}
    dead = (np.abs(truth[..., :6]).sum(1) < 1.0)                      # (n,6) off all window

    off = 0
    if args.from_day:
        hit = np.where(np.array([str(d.date()) for d in days]) == args.from_day)[0]
        off = int(hit[0]) if len(hit) else 0
    sl = slice(off, off + args.days)
    show = np.arange(n)[sl]

    L = [f"# Evening peak {args.hour:02d}:00-{end_h:02d}:00 — per-day accuracy, no averaging",
         "",
         f"{n} test days, {days[0].date()} .. {days[-1].date()}. MAE is over the "
         f"36 window steps x 6 dispatch channels of that day alone.", ""]
    main_arm = rows[-1]

    L += [f"## The {len(show)} days drawn", "",
          "| day | " + " | ".join(f"{r} MAE" for r in rows) + " | "
          + " | ".join(f"{r} WAPE" for r in rows) + " | "
          + " | ".join(f"{r} MRE" for r in rows) + " | winner |",
          "| --- |" + " ---: |" * (3 * len(rows)) + " --- |"]
    for i in show:
        best = min(rows, key=lambda r: mae_d[r][i])
        flag = " *" if dead[i].any() else ""
        L.append(f"| {days[i].date()} | "
                 + " | ".join(f"{mae_d[r][i]:,.1f}" for r in rows) + " | "
                 + " | ".join(f"{wape_d[r][i]:.1f}%" for r in rows) + " | "
                 + " | ".join(f"{mre_d[r][i]:.1f}%{flag}" for r in rows)
                 + f" | {best} |")
    L += ["", "_MAE and WAPE are the numbers to read. WAPE pools numerator and "
          "denominator over all six channels; MRE averages per-channel ratios and "
          "is marked `*` on days where at least one channel was off for the whole "
          "window, which makes its denominator ~0 -- on 2026-01-06 gas steam ran "
          "0.0 MW and alone scored 5,633%._"]
    L += ["", f"## Distribution over all {n} days (MAE, MW)", "",
          "| arm | min | p10 | median | mean | p90 | max | WAPE (all days) | "
          "days it beats interp |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        v = mae_d[r]
        pooled = 100 * np.abs(preds[r][..., :6] - truth[..., :6]).sum() / np.abs(truth[..., :6]).sum()
        wins = "" if r == "interp" else \
            f"{int((mae_d[r] < mae_d['interp']).sum())} / {n}  " \
            f"({100 * (mae_d[r] < mae_d['interp']).mean():.0f}%)"
        L.append(f"| {r} | {v.min():,.1f} | {np.percentile(v,10):,.1f} | "
                 f"{np.median(v):,.1f} | {v.mean():,.1f} | {np.percentile(v,90):,.1f} | "
                 f"{v.max():,.1f} | {pooled:.1f}% | {wins} |")

    L += ["", f"## Hardest and easiest days for `{main_arm}`", "",
          "| | day | " + " | ".join(f"{r} MAE" for r in rows) + " |",
          "| --- | --- |" + " ---: |" * len(rows)]
    order = np.argsort(mae_d[main_arm])
    for tag, i in ([("best", order[k]) for k in range(3)]
                   + [("worst", order[-1 - k]) for k in range(3)]):
        L.append(f"| {tag} | {days[i].date()} | "
                 + " | ".join(f"{mae_d[r][i]:,.1f}" for r in rows) + " |")

    L += ["", "## Per-channel MAE on the drawn days (MW)", "",
          "| day | arm | " + " | ".join(SHORT[t] for t in TARGETS) + " |",
          "| --- | --- |" + " ---: |" * 6]
    for i in show:
        for r in rows:
            L.append(f"| {days[i].date()} | {r} | "
                     + " | ".join(f"{v:,.1f}" for v in mae_c[r][i]) + " |")
    L += ["", "_MRE is unreliable where the truth is mostly zero. Share of window "
          "cells at or below 1 MW: "
          + ", ".join(f"{SHORT[t]} {100*(np.abs(truth[...,k])<=1).mean():.0f}%"
                      for k, t in enumerate(TARGETS)) + "._"]

    (OUT / f"peak_{args.hour:02d}_{end_h:02d}_perday{args.suffix}.md").write_text(
        "\n".join(L) + "\n")
    print("\n".join(L))

    # ---------------- figure: the exact days, each with its own numbers -------
    ren = np.stack([f.Xte[s_ + CTX:s_ + CTX + GAP][:, [17, 18]] * f.x_scale[[17, 18]]
                    + f.x_mean[[17, 18]] for s_ in starts])
    panels = [("actual", truth)] + [(r, preds[r]) for r in rows]
    nc, nr = len(show), len(panels)
    fig, axes = plt.subplots(nr, nc, figsize=(2.05 * nc + 1.0, 2.15 * nr),
                             sharey=True, sharex=True, squeeze=False)
    for c, i in enumerate(show):
        for r_, (name, arr) in enumerate(panels):
            ax = axes[r_][c]
            draw_day(ax, arr[i, :, :6], ren[i], arr[i, :, 6:], nd[i], args.hour)
            ax.set_xlim(args.hour, args.hour + (GAP - 1) * 5.0 / 60.0)
            if r_ == 0:
                ax.set_title(str(days[i].date()), fontsize=10.5, color=INK, pad=6)
            else:
                ax.text(0.03, 0.965,
                        f"MAE {mae_d[name][i]:,.1f} MW\nWAPE {wape_d[name][i]:.1f}%",
                        transform=ax.transAxes, fontsize=8, color=INK, va="top",
                        ha="left", zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.6))
            if c == 0:
                ax.set_ylabel(name.replace("hardnet_2025_", ""), fontsize=9.5,
                              color=INK)
    h, lb = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=6, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Evening peak {args.hour:02d}:00-{end_h:02d}:00 — {nc} exact test "
                 f"days, each panel scored on that day alone", fontsize=12.5,
                 color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0.055, 1, 0.965])
    p_ = OUT / f"peak_{args.hour:02d}_{end_h:02d}_perday{args.suffix}.png"
    fig.savefig(p_, dpi=145, facecolor="white", bbox_inches="tight")
    print(f"\nwrote {p_}")


if __name__ == "__main__":
    main()
