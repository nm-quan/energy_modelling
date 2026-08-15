"""One-day (288-step) imputation with the trained `blackout` arm.

Two modes, and the difference between them is the whole point.

  TF    teacher-forced. 4h of real dispatch on each side of the masked day, which
        is how `blackout` was trained and what eval.py measures.

  REC   recursive, and the mode the counterfactual needs. Day 1 gets NO dispatch
        context at all -- both sides zeroed -- only the subspace that is always
        observable: net_demand, demand_mw, wind, solar_utility, price, calendar.
        Day 2 takes its left context from day 1's OWN IMPUTED output, and its
        right context stays unknown. So errors compound across days, which is the
        thing a one-day counterfactual has to survive.

THE RIGHT SEAM IN REC MODE. hardnet pins the trajectory to pR at the last step.
In REC there is no observed pR: the first step of the next day is exactly as
unknown as the day being filled. Using the real one would leak the answer. So pR
is taken from the model's own raw output at that step, which makes the pin
self-consistent and adds no information.

    python3 planB/oneday_impute.py --days 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "imputation")); sys.path.insert(0, str(HERE))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
from gap_data import load_flats, TARGETS, SIGN, TARGET_FEAT_IDX        # noqa: E402
import constraints as C                                                # noqa: E402
from heads import HEADS                                                # noqa: E402
from fit import curt_activation, CURT_COLS, CTX, apply_residual, BACKBONES  # noqa: E402

DAY = 288
OUT = HERE / "results"
ND_COL, DEM_COL, PRICE_COL = 6, 7, 8
SUBSPACE = [6, 7, 8] + list(range(9, 17)) + [17, 18]     # nd, demand, price, cal, wind, solar
SHORT = {"hydro": "hydro", "coal_brown": "coal", "gas_steam": "steam",
         "gas_ocgt": "ocgt", "battery_charging": "bat_chg",
         "battery_discharging": "bat_dis"}


def load_model(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    f = load_flats()
    m = BACKBONES[ck.get("backbone", "bilstm")](len(f.feat_cols), 6, hidden=ck["hidden"])
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck, f


def day_starts(f, n_days, offset=0):
    """Start indices of whole test days that have CTX steps of room on both sides."""
    idx = np.asarray(GD.np.load(GD.NPZ, allow_pickle=True)["test_index"], dtype=str)
    first = next(i for i in range(len(idx)) if idx[i][11:16] == "00:00")
    s0 = first + offset * DAY
    while s0 - CTX < 0:
        s0 += DAY
    out = [s0 + k * DAY for k in range(n_days)]
    assert out[-1] + DAY + CTX <= len(f.Xte), "not enough test data for that many days"
    return out, idx


FREE_HOURS = (11, 14)          # midday free window, matching planB/counterfactual.py


def apply_shift(X, f, nd_gap, reduction_pct, rebound_pct, free_hours=FREE_HOURS,
                free_price=0.0):
    """The repo's counterfactual: an energy-conserving LOAD SHIFT into midday,
    not a flat rise. Mirrors lib/shift_model.py::FixedPercentageShift.

    Per calendar day, over the 288 gap steps:

        outside the window   d_new = d0 * (1 - reduction)
        inside  the window   d_new = d0 * (1 + rebound) + (MWh removed) / n_free

    The reduction term is MOVED energy, so the day very nearly conserves; the
    rebound term is genuinely induced demand, and is the only reason the day's
    total changes at all. Price inside the window is forced to `free_price`,
    which is what makes the window attractive to charge in.

    Net demand takes exactly the same change, per the scenario definition:

        nd_cf = nd_base + (demand_new - demand_old)

    Renewables, curtailment and the interconnector are held fixed, so the whole
    change lands on the six dispatchable channels. `dispatch_study/
    midday_response_2026.md` measures the real grid serving about 30% of a midday
    rise by spilling less, so the dispatchable shares come out inflated.
    """
    gs, ge = CTX, CTX + DAY
    red, reb = reduction_pct / 100.0, rebound_pct / 100.0
    # clock inside the gap: the day starts at 00:00 by construction (day_starts)
    hour = (np.arange(DAY) * 5) // 60
    free = (hour >= free_hours[0]) & (hour < free_hours[1])            # (288,)
    n_free = int(free.sum())

    dem = X[:, gs:ge, DEM_COL] * f.x_scale[DEM_COL] + f.x_mean[DEM_COL]   # (B,288) MW
    new = dem.copy()
    new[:, ~free] = dem[:, ~free] * (1.0 - red)
    removed = (dem[:, ~free] * red).sum(axis=1, keepdims=True)          # MW-steps
    new[:, free] = dem[:, free] * (1.0 + reb) + removed / max(n_free, 1)
    delta = new - dem

    X[:, gs:ge, DEM_COL] = ((new - f.x_mean[DEM_COL]) / f.x_scale[DEM_COL]).astype(np.float32)
    nd_feat = X[:, gs:ge, ND_COL] * f.x_scale[ND_COL] + f.x_mean[ND_COL]
    X[:, gs:ge, ND_COL] = (((nd_feat + delta) - f.x_mean[ND_COL])
                           / f.x_scale[ND_COL]).astype(np.float32)
    pf = np.zeros_like(X[:, gs:ge, PRICE_COL]) + free_price
    X[:, gs:ge, PRICE_COL] = np.where(
        free, (pf - f.x_mean[PRICE_COL]) / f.x_scale[PRICE_COL],
        X[:, gs:ge, PRICE_COL]).astype(np.float32)
    return X, (nd_gap + delta).astype(np.float32), free


def build_window(f, s, mode, left_ctx=None):
    """One (1,W,F) window for the day starting at flat index s.

    mode='tf'   real dispatch in both context blocks
    mode='rec'  context dispatch zeroed; if left_ctx is given (CTX,6) in MW it is
                written into the left block in z-space
    """
    W = 2 * CTX + DAY
    X = f.Xte[s - CTX:s - CTX + W].copy().astype(np.float32)[None]
    tfi = np.asarray(TARGET_FEAT_IDX)
    # nd feature = supply-side SIGN.dispatch, computed BEFORE masking (as in fit.py)
    disp = X[:, :, tfi] * f.x_scale[tfi] + f.x_mean[tfi]
    nd_all = disp @ SIGN.astype(np.float32)
    X[:, :, ND_COL] = ((nd_all - f.x_mean[ND_COL]) / f.x_scale[ND_COL]).astype(np.float32)

    gs, ge = CTX, CTX + DAY
    mask = np.ones((1, W, 1), np.float32); mask[:, gs:ge] = 0.0
    X[:, gs:ge, tfi] = 0.0
    X[:, gs:ge, CURT_COLS] = 0.0
    if mode == "rec":
        X[:, :gs, tfi] = 0.0; X[:, ge:, tfi] = 0.0
        X[:, :gs, CURT_COLS] = 0.0; X[:, ge:, CURT_COLS] = 0.0
        mask[:, :gs] = 0.0; mask[:, ge:] = 0.0
        if left_ctx is not None:
            z = (left_ctx - f.x_mean[tfi]) / f.x_scale[tfi]
            # advanced index in the last axis + a slice => result is (6, CTX)
            X[0, :gs, tfi] = z.astype(np.float32).T
            mask[0, :gs] = 1.0
    y_true = f.y_to_mw(f.Yte[s:s + DAY]).astype(np.float32)
    nd_gap = (y_true @ SIGN).astype(np.float32)[None]
    tt = (np.arange(1, DAY + 1) / (DAY + 1))[None, :, None].astype(np.float32)
    pLz, pRz = f.Yte[s - 1][None], f.Yte[s + DAY][None]
    interp = (pLz[:, None] + tt * (pRz - pLz)[:, None]).astype(np.float32)
    return X, mask, y_true, nd_gap, interp


def run_day(m, ck, f, X, mask, nd_gap, interp, pL, pR_obs, mode,
            soc_carry=None, soc=True, energy_target=None):
    """One day. Returns (P_mw (288,6), curtailment (288,2), new soc carry).

    SOC is applied as a POST-IMPUTATION projection, carrying the reservoir state
    from the previous day. Per-day accounting would restart the swing at zero
    every 288 steps and could never see a multi-day drift.
    """
    head, _ = HEADS[ck["head"]]
    ys_m = torch.tensor(f.y_mean, dtype=torch.float32)
    ys_s = torch.tensor(f.y_scale, dtype=torch.float32)
    with torch.no_grad():
        d_raw, c_raw = m(torch.from_numpy(X), torch.from_numpy(mask), None)
        d_raw = d_raw[:, CTX:CTX + DAY]
        if ck.get("residual", False):
            d_raw = apply_residual(d_raw, torch.from_numpy(interp))
        F_ = d_raw * ys_s + ys_m
        # REC has no observable seam on either side. Day 1 has no left context
        # either, so pL comes from the model's own first step; every later day
        # inherits pL from its predecessor's imputed output. Using the real
        # neighbouring steps would inject exactly the dispatch being imputed.
        pR = torch.tensor(pR_obs)[None] if mode == "tf" else F_[:, -1].clone()
        pLt = F_[:, 0].clone() if pL is None else torch.tensor(pL)[None]
        P = head(F_, pLt, pR, torch.from_numpy(nd_gap))
        if soc:
            carry = None if soc_carry is None else tuple(
                torch.tensor([v], dtype=torch.float64) for v in soc_carry)
            et = (None if energy_target is None else
                  torch.tensor([energy_target], dtype=torch.float64))
            P = C.cyclic_project(P.double(), pLt.double(), pR.double(),
                                 torch.from_numpy(nd_gap).double(), iters=200,
                                 soc=True, soc_carry=carry, energy_target=et).float()
        E, lo, hi = C.soc_state(P.double(), None if soc_carry is None else tuple(
            torch.tensor([v], dtype=torch.float64) for v in soc_carry))
        curt = curt_activation(c_raw[:, CTX:CTX + DAY],
                               torch.tensor(f.x_mean[CURT_COLS], dtype=torch.float32),
                               torch.tensor(f.x_scale[CURT_COLS], dtype=torch.float32))
    return (P[0].numpy(), curt[0].numpy(),
            (float(E.item()), float(lo.item()), float(hi.item())))


def impute(mode, n_days, offset=0, ckpt=None, soc=True, soc_window=7,
           reduction_pct=0.0, rebound_pct=0.0, energy_fix=False):
    """energy_fix=True forces the battery's total net energy over the whole run to
    equal the baseline's, so the counterfactual cannot manufacture or destroy
    stored energy. The residual is spread over the remaining days rather than
    dumped on the last one:

        target_d = baseline_d - D / r        D = drift so far, r = days left

    which drives D to exactly 0 at the end. Without it the +5% run drains
    191 MWh/day, i.e. a full pack every 25 days.
    """
    """Carry the reservoir state across days, resetting every `soc_window` days.

    WHY IT RESETS. The bound is only valid over a horizon where a single initial
    charge can explain the whole trajectory, and that horizon is finite: the
    ACTUAL dispatch needs a 4,891 MWh swing over 60 days against a 4,536 MWh
    target, so 41% of 60-day chains of real data would be declared infeasible.
    At 7 days the real maximum is 4,062 MWh and there is clear margin. Chaining
    further does not make the constraint stronger, it makes it wrong.
    """
    m, ck, f = load_model(ckpt or ROOT / "colab/planB_results/blackout.pt")
    starts, idx = day_starts(f, n_days, offset)
    preds, truths, stamps, socs = [], [], [], []
    left_ctx, carry, drift = None, None, 0.0
    for k, s in enumerate(starts):
        if soc_window and k % soc_window == 0:
            carry = None                       # new reservoir chain
        X, mask, y_true, nd_gap, interp = build_window(f, s, mode, left_ctx)
        if reduction_pct or rebound_pct:
            X, nd_gap, _ = apply_shift(X, f, nd_gap, reduction_pct, rebound_pct)
        if mode == "tf":
            pL = f.y_to_mw(f.Yte[s - 1]).astype(np.float32)     # observed seam
        elif left_ctx is not None:
            pL = left_ctx[-1].astype(np.float32)                # own previous output
        else:
            pL = None                                            # day 1: model's own
        pR_obs = f.y_to_mw(f.Yte[s + DAY]).astype(np.float32)
        et = None
        if energy_fix:
            base = float(((y_true[:, 4] * C.ETA - y_true[:, 5] / C.ETA) * C.DT).sum())
            et = base - drift / max(1, len(starts) - k)
        P, _, st = run_day(m, ck, f, X, mask, nd_gap, interp, pL, pR_obs, mode,
                           soc_carry=carry, soc=soc, energy_target=et)
        if energy_fix:
            got = float(((P[:, 4] * C.ETA - P[:, 5] / C.ETA) * C.DT).sum())
            drift += got - base
        preds.append(P); truths.append(y_true); stamps.append(idx[s][:10]); socs.append(st)
        left_ctx = P[-CTX:].copy()          # REC: next day sees its own output
        carry = st
    return np.stack(preds), np.stack(truths), stamps, socs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--no-soc", action="store_true")
    args = ap.parse_args()
    for mode in ("tf", "rec"):
        P, Y, st, socs = impute(mode, args.days, args.offset, soc=not args.no_soc)
        e = np.abs(P - Y).reshape(-1, 6)
        print(f"\n{mode.upper()}  days {st[0]}..{st[-1]}")
        print("  " + "  ".join(f"{SHORT[c]} {v:7.1f}" for c, v in zip(TARGETS, e.mean(0))))
        print(f"  aggregate MAE {e.mean():.1f} MW")
        print(f"  chained SOC swing {socs[-1][2] - socs[-1][1]:,.0f} MWh "
              f"over {len(st)} days (cap {C.BATT_CAP_MWH:,.0f})")


if __name__ == "__main__":
    main()
