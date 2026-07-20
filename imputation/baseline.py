"""Zero-learning baselines for gap imputation — the bar the bi-LSTM must beat.

Reconstruction task: mask the real 11:00-14:00 window on each test day, fill the
6-source breakdown, score against the true (measured) dispatch. Because we hid
data we possess, the answer key exists (unlike the +demand counterfactual, which
has none — see constraints/lit_review.md theme 8).

Baselines (imputation-track analogue of forecasting-track persistence):
  interp        per-source linear interpolation between the two boundaries
  interp+proj   the above, then project onto SIGN.P=net_demand + ramp tube + box
                (constraints.project_gap) — constraint-clean, still zero-learning

Metric = reconstruction WAPE per source + macro, over all gap cells of all days.

    python3 imputation/baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap_data import load_flats, test_gap_windows, TARGETS, SIGN     # noqa: E402
import constraints as C                                             # noqa: E402

OUT = HERE / "results"


def linear_fill(pL, pR, N):
    t = np.arange(1, N + 1)[:, None] / (N + 1)
    return pL[None, :] + t * (pR - pL)[None, :]            # (N,6)


def wape_per_channel(pred, truth):
    num = np.abs(pred - truth).sum(axis=0)
    den = np.abs(truth).sum(axis=0)
    return np.where(den > 1e-6, num / den, np.nan)


def evaluate(context: int = 48):     # 48 = the pipeline-wide window setting
    f = load_flats()
    gws = test_gap_windows(f, context=context)
    N = len(gws[0].gap_idx)
    methods = {"interp": [], "interp+proj": []}
    ramp_bad = {"interp": 0, "interp+proj": 0}
    neg = {"interp": 0, "interp+proj": 0}
    bal = {"interp": [], "interp+proj": []}
    num = {m: np.zeros(6) for m in methods}
    den = np.zeros(6)

    for gw in gws:
        pL, pR, truth, nd = gw.pL_mw, gw.pR_mw, gw.truth_mw, gw.nd_mw
        lin = linear_fill(pL, pR, N)
        proj, resid = C.project_gap(lin, pL, pR, nd)
        for m, P in (("interp", lin), ("interp+proj", proj)):
            num[m] += np.abs(P - truth).sum(axis=0)
            d = np.diff(np.vstack([pL, P, pR]), axis=0)
            ramp_bad[m] += int(((d > C.R_UP + 0.6) | (d < -(C.R_DN + 0.6))).sum())
            neg[m] += int((P < -0.1).sum())
        den += np.abs(truth).sum(axis=0)
        bal["interp"].append(np.abs(nd - lin @ SIGN).max())
        bal["interp+proj"].append(resid.max())

    OUT.mkdir(exist_ok=True)
    rows = []
    for m in methods:
        per = num[m] / np.clip(den, 1e-6, None)
        row = {"method": m, "n_days": len(gws), "gap_steps": N,
               "macro_WAPE": float(np.mean(per)),
               "per_channel_WAPE": {t: float(per[i]) for i, t in enumerate(TARGETS)},
               "ramp_violations": ramp_bad[m], "n_neg": neg[m],
               "balance_resid_max_mw": float(np.max(bal[m]))}
        rows.append(row)
        (OUT / f"baseline_{m.replace('+', '_')}.json").write_text(json.dumps(row, indent=2))
        print(f"\n[{m}]  macro reconstruction WAPE = {row['macro_WAPE']:.4f}   "
              f"ramp_viol={ramp_bad[m]}  n_neg={neg[m]}  bal_max={row['balance_resid_max_mw']:.1f} MW")
        for i, t in enumerate(TARGETS):
            print(f"    {t:20s} {per[i]:.4f}")
    hdr = "| method | macro | " + " | ".join(TARGETS) + " | ramp | n_neg | bal_max_mw |"
    lines = ["# Gap-imputation baselines — reconstruction WAPE on real test 11:00-14:00 windows\n",
             f"{len(gws)} test days, {N}-step gap, {context}-step context each side. "
             "Reconstruction WAPE = fill vs measured truth (answer key exists — we masked "
             "real data). This is the bar the bi-LSTM must beat.\n", hdr,
             "| " + " | ".join("---" for _ in range(len(TARGETS) + 5)) + " |"]
    for m in methods:
        r = next(x for x in rows if x["method"] == m)
        lines.append(f"| {m} | {r['macro_WAPE']:.4f} | "
                     + " | ".join(f"{r['per_channel_WAPE'][t]:.4f}" for t in TARGETS)
                     + f" | {r['ramp_violations']} | {r['n_neg']} | {r['balance_resid_max_mw']:.1f} |")
    (OUT / "baselines.md").write_text("\n".join(lines) + "\n")
    print("\nwrote", OUT / "baselines.md")
    return rows


if __name__ == "__main__":
    evaluate()
