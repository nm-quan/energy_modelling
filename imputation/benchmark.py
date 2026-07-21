"""Head-to-head benchmark of the constraint modes on ONE identical general eval,
vs the interpolation bar. Reads the per-mode reconstruction jsons that train.py
writes (bilstm_<mode>_recon.json) and computes the interp+projection baseline on the
SAME general test windows (seed 123, n 800 -- matched to train.py's eval), so every
row is apples-to-apples: same windows, same posthoc projection, same metrics.

    # after training each mode:
    #   python imputation/train.py --constraint-mode posthoc    --out results/bilstm_posthoc.pt
    #   python imputation/train.py --constraint-mode unrolled    --out results/bilstm_unrolled.pt
    #   python imputation/train.py --constraint-mode rayen_traj  --out results/bilstm_rayen_traj.pt
    python imputation/benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap_data import load_flats, sample_recon_windows                    # noqa: E402
import constraints as C                                                  # noqa: E402

OUT = HERE / "results"
# "none" = the pure bi-LSTM baseline (raw fill, NOT projected). Its ramp_overshoot /
# balance_resid / n_neg columns are non-zero on purpose -- they show the violations the
# projected modes fix, so the row is NOT constraint-clean like the others.
MODES = ["none", "posthoc", "unrolled", "rayen_traj"]


def interp_fill(gw):
    """Zero-learning bar: linear interpolation between the pinned boundaries."""
    N = len(gw.gap_idx)
    tt = (np.arange(1, N + 1) / (N + 1))[:, None]
    return gw.pL_mw[None, :] + tt * (gw.pR_mw - gw.pL_mw)[None, :]


def score_fill(gws, fill_fn):
    """Score a raw fill function on `gws` through the posthoc eval projection."""
    num = np.zeros(6); den = np.zeros(6)
    mnum = np.zeros(6); mden = np.zeros(6)       # midday(11-14) slice
    ramp_over = 0.0; neg = 0; bal = 0.0
    for gw in gws:
        P, resid = C.project_gap(fill_fn(gw), gw.pL_mw, gw.pR_mw, gw.nd_mw)
        e = np.abs(P - gw.truth_mw).sum(0); t = np.abs(gw.truth_mw).sum(0)
        num += e; den += t
        if 11 <= gw.hour < 14:
            mnum += e; mden += t
        ramp_over = max(ramp_over, C._ramp_overshoot_mw(np.vstack([gw.pL_mw, P, gw.pR_mw])))
        neg += int((P < -0.1).sum()); bal = max(bal, float(resid.max()))
    per = num / np.clip(den, 1e-6, None)
    return {"macro_WAPE": float(per.mean()), "micro_WAPE": float(num.sum() / max(den.sum(), 1e-6)),
            "midday_micro_WAPE": float(mnum.sum() / max(mden.sum(), 1e-6)),
            "ramp_overshoot_mw": ramp_over, "balance_resid_max_mw": bal, "n_neg": neg}


EVAL_SETTINGS = {"context": 48, "n_eval": 800, "eval_seed": 123}   # the interp row's windows


def row_from_model_json(d):
    """Pull the benchmark columns out of a train.py recon json. Rows are only
    strictly comparable to the interp row if the run used the SAME eval windows
    (context/n_eval/eval_seed determine them); mismatches are flagged, not hidden."""
    row = {"macro_WAPE": d.get("macro_WAPE"), "micro_WAPE": d.get("micro_WAPE"),
           "midday_micro_WAPE": d.get("per_hour_WAPE", {}).get("midday(11-14)"),
           "ramp_overshoot_mw": d.get("ramp_overshoot_mw"),
           "balance_resid_max_mw": d.get("balance_resid_max_mw"), "n_neg": d.get("n_neg")}
    off = [k for k, v in EVAL_SETTINGS.items() if d.get(k) is not None and d.get(k) != v]
    row["_flag"] = ("" if not off else
                    " ⚠ different eval windows (" + ", ".join(f"{k}={d.get(k)}" for k in off) + ")")
    return row


def main():
    f = load_flats()
    gws = sample_recon_windows(f, "test", n=EVAL_SETTINGS["n_eval"],    # same windows as train.py
                               context=EVAL_SETTINGS["context"], seed=EVAL_SETTINGS["eval_seed"])
    rows = [("interp+projection", score_fill(gws, interp_fill))]
    # classical imputers (classical_baselines.py) — same eval windows, same Π, driver-only.
    # Their jsons already store score_fill's exact schema, so they render like the interp row.
    for m in ["mean", "knn", "mice", "mf"]:
        p = OUT / f"baseline_{m}.json"
        rows.append((f"classical/{m}", json.loads(p.read_text()) if p.exists() else None))
    for mode in MODES:
        p = OUT / f"bilstm_{mode}_recon.json"
        if p.exists():
            rows.append((f"bilstm/{mode}", row_from_model_json(json.loads(p.read_text()))))
        else:
            rows.append((f"bilstm/{mode}", None))

    cols = [("macro_WAPE", "{:.4f}"), ("micro_WAPE", "{:.4f}"), ("midday_micro_WAPE", "{:.4f}"),
            ("ramp_overshoot_mw", "{:.1e}"), ("balance_resid_max_mw", "{:.1e}"), ("n_neg", "{:d}")]
    hdr = "| method | " + " | ".join(c for c, _ in cols) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cols) + " |"
    lines = [f"# Imputation benchmark (general eval, {len(gws)} windows, seed 123)", "",
             "Same windows, same projection Π. macro=mean per-channel WAPE; micro=Σerr/Σtruth "
             "(stable); midday=the 11-14 deployment slice. Bar = interpolation + projection. "
             "`classical/*` (mean/knn/mice/mf) impute the 6 sources from the known drivers only "
             "(no boundary/temporal structure); `bilstm/*` are the learned modes.", "",
             hdr, sep]
    for name, r in rows:
        if r is None:
            lines.append(f"| {name} | _not run yet_ | | | | | |")
            continue
        cells = []
        for c, fmt in cols:
            v = r.get(c)
            cells.append("—" if v is None else fmt.format(v))
        lines.append(f"| {name}{r.get('_flag', '')} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    OUT.mkdir(exist_ok=True)
    (OUT / "benchmark.md").write_text(table + "\n")
    print(table)
    print(f"\nwrote {OUT / 'benchmark.md'}")


if __name__ == "__main__":
    main()
