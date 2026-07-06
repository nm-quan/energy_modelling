"""When does RevIN cancel a demand change? Localized bump vs whole-window lift.

Same 24h lookback window, two perturbations, each shown raw and RevIN-normalized:

  LEFT  - localized free-window bump (our 11:00-14:00 shift): only a few hours move,
          so the window mu/sigma barely move -> RevIN RETAINS the bump.
  RIGHT - uniform whole-window lift (+x% on every step): mu and sigma move to exactly
          cancel it (proven: (a d + b - (a mu + b))/(a sigma) = (d-mu)/sigma) ->
          RevIN ERASES it. The two normalized curves lie on top of each other.

This is the point: RevIN is invariant to the *overall demand level*, not to a
localized shape change. A long lookback is what makes our bump localized, so the
one-step input keeps the signal -- the iTransformer's real failure is the
closed-loop collapse (behaviour/report.md), not this one-step normalization.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

OUT = HERE / "sweep_eqnd" / "figure"
FREE = (11, 14)
LB = 288


def revin(w):
    return (w - w.mean()) / (w.std() + 1e-5)


def retained(base, scen):
    """% of the free-window perturbation that survives RevIN normalization."""
    fw_raw = scen.mean() - base.mean()
    rb, rs = revin(base), revin(scen)
    return 100.0 * (rs.mean() - rb.mean()) / (revin(base + (fw_raw)).mean() - rb.mean() + 1e-9)


def main():
    df = pipeline.build_table("5min")
    end = pd.Timestamp("2026-04-16 14:00", tz=df.index.tz)
    p = df.index.get_indexer([end])[0]
    sl = slice(p - LB, p)
    idx = df.index[sl].tz_localize(None)
    base = df["demand_mw"].to_numpy()[sl]
    fw = (idx.hour >= FREE[0]) & (idx.hour < FREE[1])

    # localized: the real reb20/red10 shift
    loc = FixedPercentageShift(20, 10, free_hours=FREE).transform(df)["demand_mw"].to_numpy()[sl]
    # uniform: lift the WHOLE window by the same average % the localized bump adds in-window
    lift = (loc[fw].mean() - base[fw].mean()) / base.mean()
    uni = base * (1 + lift)

    def frac(b, s):  # fraction of the in-window raw gap retained after RevIN
        raw = s[fw].mean() - b[fw].mean()
        rev = revin(s)[fw].mean() - revin(b)[fw].mean()
        # express RevIN delta as % of the fixed-scale delta (raw/sd_window of base)
        return 100.0 * rev / (raw / (b.std() + 1e-5))

    r_loc, r_uni = frac(base, loc), frac(base, uni)
    print(f"localized free-window bump: RevIN retains {r_loc:.0f}% of the raw bump")
    print(f"uniform whole-window +{100*lift:.0f}%: RevIN retains {r_uni:.0f}% of the raw lift")

    f, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    cols = [("LOCALIZED free-window bump (reb20/red10)", loc, r_loc),
            (f"UNIFORM whole-window lift (+{100*lift:.0f}%)", uni, r_uni)]
    for c, (title, scen, ret) in enumerate(cols):
        for row, (space, fn, ylab) in enumerate(
                [("raw MW", lambda a: a, "MW"),
                 ("RevIN (x-mu)/sigma", revin, "z")]):
            ax = axes[row][c]
            ax.plot(idx, fn(base), color="#1f77b4", lw=1.3, label="base")
            ax.plot(idx, fn(scen), color="#d62728", lw=1.3, label="perturbed")
            for day in np.unique(idx.normalize()):
                ax.axvspan(day + pd.Timedelta(hours=FREE[0]),
                           day + pd.Timedelta(hours=FREE[1]), color="gold", alpha=0.18)
            ax.grid(True, alpha=0.3); ax.margins(x=0); ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(f"{title}\n{space}", fontsize=10)
            else:
                ax.set_title(f"{space} — RevIN retains {ret:.0f}% of the bump", fontsize=10)
    axes[0][0].legend(loc="upper left", fontsize=9)
    for ax in axes[1]:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    f.suptitle("RevIN cancels a whole-window level change, NOT a localized bump "
               "(why a 24h lookback keeps the one-step signal)")
    f.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "revin_cancel_demo.png"
    f.savefig(out, dpi=130); plt.close(f)
    print("wrote", out)


if __name__ == "__main__":
    main()
