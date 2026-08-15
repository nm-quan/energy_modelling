"""Where does hardnet's gas-steam response come from: the BiLSTM or the projection?

Wrap heads.hardnet_head so every call records (F_in, P_out), then run the normal
counterfactual for the hardnet arm. The response splits exactly:

    resp        = E_c - E_b                      (what the report table shows)
    resp_net    = F_c - F_b                      (the raw network, pre-head)
    resp_proj   = (P_c - F_c) - (P_b - F_b)      (what the projection added)

Call order inside counterfactual.run() is, per mode, per day: fill(base) then
fill(shifted) -- so even calls are base, odd are counterfactual.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/nguyenminhquan/Downloads/energy_modelling")
sys.path.insert(0, str(ROOT / "planB"))

import heads                                                           # noqa: E402
import counterfactual as CF                                            # noqa: E402
from gap_data import TARGETS, SIGN                                     # noqa: E402

REC = []
_orig = heads.hardnet_head


def recording_head(raw, pL, pR, nd, return_debug=False):
    out = _orig(raw, pL, pR, nd, return_debug=return_debug)
    P = out[0] if return_debug else out
    REC.append((raw[..., :6].detach().numpy().copy(), P.detach().numpy().copy()))
    return out


heads.hardnet_head = recording_head
CF.HEADS = dict(CF.HEADS)
CF.HEADS["hardnet"] = (recording_head, 6)
CF.OUT = ROOT / "colab" / "planB_results"

sys.argv = ["cf", "--days", "10", "--arms", "hardnet", "--suffix", "_decomp"]
import contextlib, io                                                  # noqa: E402
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    CF.main()

DT = 5.0 / 60.0
n = len(REC)
print(f"recorded {n} head calls (2 modes x 10 days x [base, counterfactual])\n")
# second half = masked mode, matching the report's masked table
half = n // 2
masked = REC[half:]
F_b = np.stack([masked[i][0][0] for i in range(0, len(masked), 2)])
P_b = np.stack([masked[i][1][0] for i in range(0, len(masked), 2)])
F_c = np.stack([masked[i][0][0] for i in range(1, len(masked), 2)])
P_c = np.stack([masked[i][1][0] for i in range(1, len(masked), 2)])

E = lambda A: A.sum((0, 1)) * DT                                       # MWh per channel
resp = E(P_c) - E(P_b)
resp_net = E(F_c) - E(F_b)
resp_proj = resp - resp_net
tot = abs(float(resp @ SIGN))

EMP = {"coal_brown": 43.4, "hydro": 29.3, "gas_ocgt": 9.7, "gas_steam": 3.7,
       "battery_discharging": 7.4, "battery_charging": 6.6}
print(f"{'channel':22s} {'RESPONSE MWh':>13s} {'share':>7s} | {'from BiLSTM':>12s} "
      f"{'from proj':>11s} | {'real':>6s}")
print("-" * 82)
for i, t in enumerate(TARGETS):
    print(f"{t:22s} {resp[i]:+13,.0f} {100*abs(resp[i])/tot:6.1f}% | "
          f"{resp_net[i]:+12,.0f} {resp_proj[i]:+11,.0f} | {EMP[t]:5.1f}%")
print(f"\n{'|correction| MWh':22s} base {np.abs(P_b-F_b).sum((0,1)).sum()*DT:,.0f}   "
      f"cf {np.abs(P_c-F_c).sum((0,1)).sum()*DT:,.0f}")

print("\nper-channel MEAN LEVEL in the window (MW), counterfactual case")
print(f"{'channel':22s} {'raw F':>10s} {'after head P':>13s} {'moved':>9s}")
for i, t in enumerate(TARGETS):
    print(f"{t:22s} {F_c[..., i].mean():10.1f} {P_c[..., i].mean():13.1f} "
          f"{(P_c-F_c)[..., i].mean():+9.1f}")
