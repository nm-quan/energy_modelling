"""Shared constraint-evaluation harness (see constraint_research.md).

constraint_report() turns a (N,6) MW prediction array into one metrics row:
WAPE/R2, box violations (n_neg / n_cap), ramp violations on the predicted
trajectory, demand-balance violations + mismatch against both demand references
(nd given to the model, actual nd), and battery SOC feasibility (feasible SOC0
window as % of nameplate, or None). Constants and the SOC swing logic are
imported from ml/check_caps.py so CONSTRAINT.md stays the single source of
truth. Markdown tables are rebuilt from per-run JSONs so re-runs never clobber
other rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # constraints/
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "ml"))
import check_caps as cc          # noqa: E402  CAPS/RAMPS/BATT_CAP_MWH/_soc_swing
import evaluate as ev            # noqa: E402

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]
SIGN = np.array([1, 1, 1, 1, -1, 1], dtype=np.float64)
RESULTS = HERE / "results"

_eta_cache: dict[str, float | None] = {}


def _eta(res: str) -> float | None:
    if res not in _eta_cache:
        _eta_cache[res] = cc.calibrate_round_trip_eta(res)
    return _eta_cache[res]


def _consec(index, res: str) -> np.ndarray:
    """True where row t is exactly one interval after row t-1 (ramp/SOC pairing)."""
    step = cc.RES_HOURS[res] * 3600
    dt = pd.DatetimeIndex(index).to_series().diff().dt.total_seconds().to_numpy()
    m = dt == step
    m[0] = False
    return m


def constraint_report(pred: np.ndarray, actual: np.ndarray, index,
                      nd_input: np.ndarray | None = None, res: str = "5min",
                      demand_thresh: float = 10.0) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    met = ev.compute_metrics(actual, pred, TARGETS)["average"]
    out = {"WAPE": met["WAPE"], "R2": met["R2"], "n": len(pred)}

    # box: negativity + per-target data-max caps
    out["n_neg"] = int((pred < 0).sum())
    out["n_cap"] = int(sum((pred[:, i] > cc.CAPS[t]).sum() for i, t in enumerate(TARGETS)))

    # ramps on the predicted trajectory, consecutive pairs only
    consec = _consec(index, res)
    n_ramp = 0
    for i, t in enumerate(TARGETS):
        rdn, rup = cc.RAMPS[t]
        delta = np.diff(pred[:, i], prepend=pred[0, i])
        n_ramp += int((((delta > rup) | (delta < rdn)) & consec).sum())
    out["n_ramp"] = n_ramp

    # demand balance vs both references
    nd_pred = pred @ SIGN
    for tag, ref in (("in", nd_input), ("act", actual @ SIGN)):
        if ref is None:
            out[f"n_demand_{tag}"] = None
            out[f"mismatch_{tag}_pct"] = None
            continue
        resid = nd_pred - np.asarray(ref, dtype=np.float64)
        out[f"n_demand_{tag}"] = int((np.abs(resid) > demand_thresh).sum())
        out[f"mismatch_{tag}_pct"] = float(100 * np.abs(resid).sum() / np.abs(ref).sum())

    # battery SOC feasibility: does any SOC0 in [0, cap] keep the reservoir in
    # bounds? Gaps in the index start a new reservoir segment (as in check_caps).
    eta_rt = _eta(res)
    if eta_rt is None:
        out.update(soc_feasible=None, soc0_window_pct=None, soc_swing_pct=None)
    else:
        seg = np.cumsum(~consec)
        chg = np.clip(pred[:, TARGETS.index("battery_charging")], 0, None)
        dis = np.clip(pred[:, TARGETS.index("battery_discharging")], 0, None)
        swing, cmin, cmax, _ = cc._soc_swing(chg, dis, seg, np.sqrt(eta_rt),
                                             cc.RES_HOURS[res])
        cap = cc.BATT_CAP_MWH
        out["soc_swing_pct"] = float(100 * swing / cap)
        if swing <= cap + 1e-6:
            out["soc_feasible"] = True
            out["soc0_window_pct"] = [float(100 * -cmin / cap), float(100 * (cap - cmax) / cap)]
        else:
            out["soc_feasible"] = False
            out["soc0_window_pct"] = None
    return out


# ----------------------------- reporting -----------------------------

COLS = ["model", "WAPE", "R2", "n_neg", "n_cap", "n_ramp",
        "n_demand_in", "mismatch_in_pct", "n_demand_act", "mismatch_act_pct", "SOC0"]


def _fmt(row: dict) -> str:
    def num(v, spec):
        return "—" if v is None else format(v, spec)
    if row.get("soc_feasible") is None:
        soc = "—"
    elif row["soc_feasible"]:
        lo, hi = row["soc0_window_pct"]
        soc = f"[{lo:.0f}%, {hi:.0f}%]"
    else:
        soc = f"None (swing {row['soc_swing_pct']:.0f}%)"
    return (f"| {row['model']} | {row['WAPE']:.4f} | {row['R2']:.4f} | "
            f"{row['n_neg']} | {row['n_cap']} | {row['n_ramp']} | "
            f"{num(row['n_demand_in'], 'd')} | {num(row['mismatch_in_pct'], '.2f')} | "
            f"{num(row['n_demand_act'], 'd')} | {num(row['mismatch_act_pct'], '.2f')} | {soc} |")


def save_row(stage: str, row: dict) -> None:
    """One JSON per (stage, model); the stage md table is rebuilt from all JSONs."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    safe = row["model"].replace("/", "_").replace(" ", "_").replace("+", "_")
    (RESULTS / f"{stage}_{safe}.json").write_text(json.dumps(row, indent=2))


def write_table(stage: str, title: str, order: list[str] | None = None) -> str:
    rows = []
    for p in sorted(RESULTS.glob(f"{stage}_*.json")):
        rows.append(json.loads(p.read_text()))
    if order:
        pos = {m: i for i, m in enumerate(order)}
        rows.sort(key=lambda r: pos.get(r["model"], len(pos)))
    lines = [f"# {title}\n",
             "demand refs: `in` = nd(t-1) given to the model, `act` = actual nd(t); "
             f"violation threshold 10 MW. SOC0 = feasible starting-charge window, % of "
             f"{cc.BATT_CAP_MWH:.0f} MWh.\n",
             "| " + " | ".join(COLS) + " |",
             "| " + " | ".join("---" for _ in COLS) + " |"]
    lines += [_fmt(r) for r in rows]
    text = "\n".join(lines) + "\n"
    (RESULTS / f"{stage}.md").write_text(text)
    return text
