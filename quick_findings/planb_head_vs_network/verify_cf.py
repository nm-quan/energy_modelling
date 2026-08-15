"""Constraint verification of the counterfactual fills.

Wraps every head call so the EXACT tensors the head saw and produced are audited,
rather than re-deriving them. For each arm, on the 10 counterfactual days:

    balance   |SIGN.P - nd_cf|                        head guarantees exact
    box       max(-P, 0) and max(P - CAP, 0)          head guarantees exact
    ramp      inside the window AND at both seams     head guarantees exact
              (pL -> P[0] and P[-1] -> pR)
    SOC       battery energy swing over the window    NOT enforced by any head

Reported separately for the base case and the shifted counterfactual, because
the shifted one is where the head does the most work.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/nguyenminhquan/Downloads/energy_modelling")
sys.path.insert(0, str(ROOT / "imputation"))
sys.path.insert(0, str(ROOT / "planB"))

import heads                                                           # noqa: E402
import counterfactual as CF                                            # noqa: E402
import constraints as IC                                               # noqa: E402
from gap_data import TARGETS, SIGN                                     # noqa: E402
from limits import R_UP, R_DN, CAP                                     # noqa: E402

ARMS = sys.argv[1:] or ["hardnet_2025_base"]
CF.OUT = ROOT / "planB" / "results"
REC = []


def _wrap(fn):
    def rec(raw, pL, pR, nd, return_debug=False):
        out = fn(raw, pL, pR, nd, return_debug=return_debug)
        P = out[0] if return_debug else out
        REC.append(tuple(t.detach().numpy().copy() for t in (P, pL, pR, nd)))
        return out
    return rec


CF.HEADS = dict(CF.HEADS)
for _n, _k in (("hardnet", 6), ("hardnet_alloc", 12)):
    _w = _wrap(heads.HEADS[_n][0])
    setattr(heads, _n + "_head", _w)
    CF.HEADS[_n] = (_w, _k)

CAP_EFF = IC.BATT_CAP_MWH - 2.0 * 100.0          # same margin cyclic_project uses

print(f"{'arm':26s} {'case':6s} {'balance':>10s} {'below 0':>9s} {'above cap':>10s} "
      f"{'ramp in-win':>12s} {'ramp seams':>11s} {'SOC swing':>11s}")
print("-" * 92)
for arm in ARMS:
    REC.clear()
    sys.argv = ["cf", "--days", "10", "--arms", arm, "--suffix", "_verify"]
    with contextlib.redirect_stdout(io.StringIO()):
        CF.main()
    half = len(REC) // 2                          # 2nd half = masked mode
    masked = REC[half:]
    for lbl, off in (("base", 0), ("cf", 1)):
        bal = boc = cap_ = rin = rsm = soc = 0.0
        for k in range(off, len(masked), 2):
            P, pL, pR, nd = (a[0] for a in masked[k])
            bal = max(bal, float(np.abs(P @ SIGN - nd).max()))
            boc = max(boc, float(np.maximum(-P, 0).max()))
            cap_ = max(cap_, float(np.maximum(P - CAP, 0).max()))
            d = np.diff(P, axis=0)
            rin = max(rin, float((np.maximum(d - R_UP, 0)
                                  + np.maximum(-d - R_DN, 0)).max()))
            for dd in (P[0] - pL, pR - P[-1]):
                rsm = max(rsm, float((np.maximum(dd - R_UP, 0)
                                      + np.maximum(-dd - R_DN, 0)).max()))
            dE = (P[:, IC.CHG_IDX] * IC.ETA - P[:, IC.DIS_IDX] / IC.ETA) * IC.DT
            E = np.concatenate([[0.0], np.cumsum(dE)])
            soc = max(soc, float(E.max() - E.min()))
        print(f"{arm:26s} {lbl:6s} {bal:10.2e} {boc:9.2e} {cap_:10.2e} "
              f"{rin:12.2e} {rsm:11.2e} {soc:8.1f}MWh")

print(f"\nSOC reference: battery reservoir {IC.BATT_CAP_MWH:,.0f} MWh, "
      f"cyclic_project's effective cap {CAP_EFF:,.0f} MWh (200 MWh margin).")
print("The window is 3h, so its swing is only part of a day -- this is a lower")
print("bound on the daily swing, not the daily constraint. No planB head enforces")
print("SOC at all (heads.py says so); imputation/constraints.py:_soc_project does.")
for f in ("counterfactual_verify.md", "counterfactual_10day_verify.png"):
    (CF.OUT / f).unlink(missing_ok=True)
