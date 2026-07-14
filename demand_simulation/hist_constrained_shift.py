"""Demand-shift simulation for hist constrained models.

Runs the four Colab hist constrained checkpoints:

  - lstm_rayen
  - itransformer_rayen
  - lstm_task7+DR
  - itransformer_task7+DR

The scenario is a FixedPercentageShift on hist demand_mw. By default the script
computes the largest equal percentage q such that rebound_pct = reduction_pct =
q keeps reshaped demand_mw non-negative and below the demand cap from
CONSTRAINT.md over the exact rollout frame (val tail + test).

Only a markdown results table is written.

    python3 demand_simulation/hist_constrained_shift.py
    python3 demand_simulation/hist_constrained_shift.py --device cpu --max-steps 288
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "ml"))

import decision_rule as dr       # noqa: E402
import evaluate as ev            # noqa: E402
import models as M               # noqa: E402
import pipeline                  # noqa: E402
from check_caps import BATT_CAP_MWH, CAPS, RAMPS, RES_HOURS  # noqa: E402
from shift_model import FixedPercentageShift  # noqa: E402

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]
SIGN = np.array([1, 1, 1, 1, -1, 1], dtype=np.float64)
FREE_HOURS = (11, 14)
DEMAND_CAP_MW = 10783.7
RAMP_TOL = 0.6
DEMAND_TOL = 10.0
ETA_RT = 0.834
OUT = HERE / "sweep_eqnd" / "hist_constrained_shift.md"


def demand_side_nd_hist(frame: pd.DataFrame) -> pd.Series:
    """Hist demand_mw is already VIC-island adjusted; do not subtract imports."""
    return frame["demand_mw"] - frame["wind"] - frame["solar_utility"]


def max_equal_shift_pct(frame: pd.DataFrame, cap_mw: float,
                        free_hours=FREE_HOURS) -> float:
    """Largest q with rebound_pct = reduction_pct = q that respects demand cap.

    FixedPercentageShift gives, for a free-window interval j on a day:
      d'_j = d_j * (1 + q) + q * daily_nonfree_sum / n_free
    with q in fraction units. Outside the free window, d' = d * (1 - q), so
    non-negativity adds q <= 1.
    """
    idx = frame.index
    free = (idx.hour >= free_hours[0]) & (idx.hour < free_hours[1])
    if not np.any(free):
        raise ValueError("no free-window intervals in frame")
    d = frame["demand_mw"].to_numpy(dtype=np.float64)
    days = idx.normalize()
    removed_at_100 = np.where(~free, d, 0.0)
    tmp = pd.DataFrame({"day": days, "removed": removed_at_100,
                        "free": free.astype(float)}, index=idx)
    shifted = tmp.groupby("day")["removed"].transform("sum").to_numpy()
    n_free = tmp.groupby("day")["free"].transform("sum").to_numpy()
    add_at_100 = np.where(free & (n_free > 0),
                          shifted / np.where(n_free > 0, n_free, 1.0), 0.0)
    denom = d[free] + add_at_100[free]
    q = np.min((cap_mw - d[free]) / denom)
    return float(100.0 * max(0.0, min(1.0, q)))


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_entries(weights: Path, data: dict, device: str, model_filter=None,
                 rayenfd_steam_pt: bool = False):
    xs, ys, fc = data["x_scaler"], data["y_scaler"], data["feat_cols"]
    nd_idx = fc.index("net_demand")
    ramp_dn = [abs(RAMPS[t][0]) for t in TARGETS]
    ramp_up = [RAMPS[t][1] for t in TARGETS]

    specs = [
        ("lstm_rayen", 7, weights / "lstm_rayen_hist_s0.pt", None),
        ("itransformer_rayen", 7, weights / "itransformer_rayen_hist_s0.pt", None),
        ("itransformer_rayenfd", 6, weights / "itransformer_rayenfd_hist_s0.pt", None),
        ("lstm_task7+DR", 7, weights / "lstm_task7_hist_s0.pt",
         weights / "lstm_task7_hist_s0_safeF.npz"),
        ("itransformer_task7+DR", 7, weights / "itransformer_task7_hist_s0.pt",
         weights / "itransformer_task7_hist_s0_safeF.npz"),
    ]
    if model_filter:
        specs = [s for s in specs if any(f in s[0] for f in model_filter)]
        if not specs:
            raise ValueError(f"no checkpoints match --models {model_filter}")
    missing = [str(p) for _, _, p, f in specs if not p.exists()] \
        + [str(f) for _, _, _, f in specs if f is not None and not f.exists()]
    if missing:
        raise FileNotFoundError("missing required weight files:\n  " + "\n  ".join(missing))

    entries = []
    for name, n_out, pt, safe_f in specs:
        base_arch = "itransformer" if name.startswith("itransformer") else "lstm"
        if "rayen" in name:
            model = M.make_rayen(base_arch, xs, ys, ramp_up, ramp_dn,
                                 nd_feat_idx=nd_idx, n_features=len(fc),
                                 fix_demand="rayenfd" in name,
                                 passthrough_idx=(2,) if ("rayenfd" in name and rayenfd_steam_pt)
                                 else None)
            model.load_state_dict(_torch_load(pt), strict=False)
            if "rayenfd" in name:
                free = torch.ones(6)
                if rayenfd_steam_pt:
                    free[2] = 0.0
                model._free.copy_(free)
                model._sign_free.copy_(torch.tensor(SIGN, dtype=torch.float32) * free)
        else:
            task = M.make_task7(base_arch, n_features=len(fc), nd_feat_idx=nd_idx)
            task.load_state_dict(_torch_load(pt))
            z = np.load(safe_f)
            model = dr.DecisionRuleHead(task, z["F"], xs.mean_, xs.scale_,
                                        ys.mean_, ys.scale_, ramp_up, ramp_dn,
                                        nd_feat_idx=nd_idx)
        entries.append((name, n_out, model.to(device).eval()))
    return entries


def rollout(model, n_out: int, fs_base: np.ndarray, fs_scen: np.ndarray,
            lb: int, free_test: np.ndarray, xs, ys, device: str):
    """Closed-loop free-window rollout for 7-output constrained hist models."""
    fs = np.stack([fs_base, fs_scen])
    fs_t = torch.from_numpy(fs).to(device)
    ymean = torch.tensor(ys.mean_, dtype=torch.float32, device=device)
    yscale = torch.tensor(ys.scale_, dtype=torch.float32, device=device)
    xmean = torch.tensor(xs.mean_, dtype=torch.float32, device=device)
    xscale = torch.tensor(xs.scale_, dtype=torch.float32, device=device)
    tfi = torch.tensor(M.TARGET_FEAT_IDX, dtype=torch.long, device=device)
    nd_idx = int(np.where(np.asarray(data_feat_cols) == "net_demand")[0][0])
    sign = torch.tensor(SIGN, dtype=torch.float32, device=device)

    n = len(free_test)
    preds = torch.zeros((2, n, 6), dtype=torch.float32, device=device)
    d_own = torch.zeros((2, n), dtype=torch.float32, device=device)
    alpha_scen = []
    window, free_prev = None, False
    with torch.no_grad():
        for i in range(n):
            free_now = bool(free_test[i])
            if not (free_now and free_prev):
                window = fs_t[:, i:i + lb, :].clone()
            out = model(window)
            p_scaled = out[:, 1:] if n_out == 7 else out
            pred_mw = p_scaled * yscale + ymean
            preds[:, i, :] = pred_mw
            if n_out == 7:
                d_own[:, i] = out[:, 0] * xscale[nd_idx] + xmean[nd_idx]
            else:
                d_own[:, i] = pred_mw @ sign
            if getattr(model, "last_alpha", None) is not None:
                alpha_scen.append(float(model.last_alpha.detach().cpu().numpy()[1]))
            if free_now:
                newrow = fs_t[:, i + lb, :].clone()
                newrow[:, tfi] = (pred_mw - xmean.index_select(0, tfi)) / xscale.index_select(0, tfi)
                # Keep scenario/base net_demand, demand_mw and price teacher-forced from fs_t.
                window = torch.cat([window[:, 1:, :], newrow[:, None, :]], dim=1)
            free_prev = free_now
            if device == "mps" and i % 2000 == 0:
                torch.mps.empty_cache()   # allocator cache grows over the 53k-step loop -> OOM
    return (preds[0].cpu().numpy(), preds[1].cpu().numpy(),
            d_own[0].cpu().numpy(), d_own[1].cpu().numpy(),
            np.asarray(alpha_scen, dtype=np.float64))


def wape_r2_response(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    per = ev.compute_metrics(actual, pred, TARGETS)["per_target"]
    wapes, r2s = [], []
    for t in TARGETS:
        if np.isfinite(per[t]["WAPE"]):
            wapes.append(per[t]["WAPE"])
        if np.isfinite(per[t]["R2"]):
            r2s.append(per[t]["R2"])
    return float(np.mean(wapes)), float(np.mean(r2s))


def ramp_counts(pred: np.ndarray, prev_first: np.ndarray) -> dict:
    """Ramp-violation stats on the predicted trajectory.

    A violation is a (step, target) cell whose 5-min change |P_t - P_{t-1}|
    exceeds the asymmetric data ramp (+RAMP_TOL). `ex` is the per-cell excess in
    MW; multiplying by the interval length RES_HOURS['5min'] gives the excess
    *energy* delivered over that 5-min step in MWh. Means are over the violating
    cells only (i.e. the typical severity of a violation, not diluted by the
    ~53k*6 compliant cells)."""
    ramp_up = np.array([RAMPS[t][1] for t in TARGETS])
    ramp_dn = np.array([abs(RAMPS[t][0]) for t in TARGETS])
    delta = np.diff(np.concatenate([prev_first[None], pred]), axis=0)
    up_ex = np.clip(delta - (ramp_up + RAMP_TOL), 0, None)
    dn_ex = np.clip(-delta - (ramp_dn + RAMP_TOL), 0, None)
    ex = up_ex + dn_ex
    viol = ex > 0
    n = int(viol.sum())
    ex_sum_mw = float(ex.sum())
    h = RES_HOURS["5min"]
    return {
        "n": n,
        "excess_sum_mw": ex_sum_mw,
        "mean_mw": ex_sum_mw / n if n else 0.0,        # mean MW over ramp per violation
        "max_mw": float(ex.max()) if n else 0.0,
        "mean_mwh": (ex_sum_mw / n if n else 0.0) * h,  # mean excess energy per violation
        "total_mwh": ex_sum_mw * h,                     # total excess energy over rollout
    }


def soc_swing_pct(pred: np.ndarray) -> tuple[bool, float]:
    eta = np.sqrt(ETA_RT)
    chg = np.clip(pred[:, TARGETS.index("battery_charging")], 0, None)
    dis = np.clip(pred[:, TARGETS.index("battery_discharging")], 0, None)
    d_e = (chg * eta - dis / eta) * RES_HOURS["5min"]
    cum = np.concatenate([[0.0], np.cumsum(d_e)])
    swing = float(cum.max() - cum.min())
    return swing <= BATT_CAP_MWH + 1e-6, float(100 * swing / BATT_CAP_MWH)


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(v)
    if abs(float(v)) >= 100:
        return f"{float(v):.1f}"
    return f"{float(v):.4f}" if abs(float(v)) < 10 else f"{float(v):.1f}"


def write_table(path: Path, note: str, rows: list[dict]):
    cols = ["model", "base_WAPE", "base_R2", "demand_in_pct", "nd_resp_pct",
            "coal_resp_pct", "hydro_resp_pct", "ocgt_resp_pct", "batt_dis_resp_pct",
            "track_p50_mw", "track_p95_mw", "bal_own_max_mw", "n_demand_in",
            "mismatch_in_pct", "n_ramp", "ramp_mean_mw", "ramp_mean_mwh",
            "ramp_max_mw", "ramp_total_mwh", "ramp_excess_pct", "n_neg",
            "soc_feasible", "soc_swing_pct", "alpha_active_pct"]
    lines = ["# Hist constrained demand-shift simulation\n", note + "\n",
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--rebound", type=float, default=None)
    ap.add_argument("--reduction", type=float, default=None)
    ap.add_argument("--demand-cap", type=float, default=DEMAND_CAP_MW)
    ap.add_argument("--models", default=None,
                    help="comma-separated substrings selecting which checkpoints "
                         "to run, e.g. 'itransformer_rayen' (default: all 4)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="smoke-test cap on test rollout length")
    ap.add_argument("--out", default=str(OUT))
    return ap.parse_args()


data_feat_cols: list[str] = []


def main():
    global data_feat_cols
    args = parse_args()
    device = ev.pick_device(args.device)
    print(f"device={device}")

    data = pipeline.load_prepared("5min", 24, 1, "net_dispatch_totdem", "hist")
    xs, ys, fc = data["x_scaler"], data["y_scaler"], data["feat_cols"]
    data_feat_cols = fc
    lb = data["lookback_steps"]
    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    entries = load_entries(Path(args.weights), data, device, model_filter)

    df = pipeline.build_table("5min", "hist")
    val_end = pipeline.DATASETS["hist"]["val_end"]
    val_df = df[(df.index > pipeline.DATASETS["hist"]["train_end"]) & (df.index <= val_end)]
    test_df = df[df.index > val_end]
    if args.max_steps is not None:
        test_df = test_df.iloc[:args.max_steps]
    full = pd.concat([val_df.tail(lb), test_df])
    test_index = test_df.index
    free_test = np.asarray((test_index.hour >= FREE_HOURS[0]) & (test_index.hour < FREE_HOURS[1]))
    mask = np.asarray((test_index - pd.Timedelta(minutes=5)).hour >= FREE_HOURS[0]) \
        & np.asarray((test_index - pd.Timedelta(minutes=5)).hour < FREE_HOURS[1])

    qmax = max_equal_shift_pct(full, args.demand_cap, FREE_HOURS)
    rebound = qmax if args.rebound is None else args.rebound
    reduction = qmax if args.reduction is None else args.reduction
    shift = FixedPercentageShift(rebound_pct=rebound, reduction_pct=reduction,
                                 free_hours=FREE_HOURS)

    base = df.copy()
    base["net_demand"] = demand_side_nd_hist(base)
    scen = shift.transform(df)
    scen["net_demand"] = demand_side_nd_hist(scen)
    scen_full = scen.loc[full.index, "demand_mw"]
    if (scen_full < -1e-6).any() or (scen_full > args.demand_cap + 1e-6).any():
        raise ValueError("scenario violates demand cap/non-negativity; lower rebound/reduction")

    fs_base = xs.transform(base.loc[full.index, fc].values).astype(np.float32)
    fs_scen = xs.transform(scen.loc[full.index, fc].values).astype(np.float32)
    actual = test_df[TARGETS].to_numpy(dtype=np.float64)
    actual_resp = actual[mask]
    nd_input_scen = scen.loc[test_index, "net_demand"].to_numpy(dtype=np.float64)
    dem_b = base.loc[test_index, "demand_mw"].to_numpy(dtype=np.float64)
    dem_s = scen.loc[test_index, "demand_mw"].to_numpy(dtype=np.float64)
    prev0 = (fs_scen[lb - 1, M.TARGET_FEAT_IDX] * xs.scale_[M.TARGET_FEAT_IDX]
             + xs.mean_[M.TARGET_FEAT_IDX]).astype(np.float64)

    rows = []
    for name, n_out, model in entries:
        t0 = time.time()
        print(f"rolling out {name} ...", flush=True)
        pb, ps, db, ds, alpha = rollout(model, n_out, fs_base, fs_scen, lb,
                                        free_test, xs, ys, device)
        b_resp, s_resp = pb[mask], ps[mask]
        base_wape, base_r2 = wape_r2_response(actual_resp, b_resp)
        nd_b, nd_s = b_resp @ SIGN, s_resp @ SIGN
        track = np.abs(ps @ SIGN - nd_input_scen)
        resid_in = ps @ SIGN - nd_input_scen
        bal = np.abs(ps @ SIGN - ds)
        rs = ramp_counts(ps, prev0)
        soc_ok, soc_pct = soc_swing_pct(ps)
        resp = {}
        for i, t in enumerate(TARGETS):
            denom = b_resp[:, i].mean()
            resp[t] = 100.0 * (s_resp[:, i].mean() - denom) / denom if abs(denom) > 1e-9 else np.nan
        row = {
            "model": name,
            "base_WAPE": base_wape,
            "base_R2": base_r2,
            "demand_in_pct": 100.0 * (dem_s[mask].mean() - dem_b[mask].mean()) / dem_b[mask].mean(),
            "nd_resp_pct": 100.0 * (nd_s.mean() - nd_b.mean()) / nd_b.mean(),
            "coal_resp_pct": resp["coal_brown"],
            "hydro_resp_pct": resp["hydro"],
            "ocgt_resp_pct": resp["gas_ocgt"],
            "batt_dis_resp_pct": resp["battery_discharging"],
            "track_p50_mw": float(np.percentile(track, 50)),
            "track_p95_mw": float(np.percentile(track, 95)),
            "bal_own_max_mw": float(bal.max()),
            "n_demand_in": int((np.abs(resid_in) > DEMAND_TOL).sum()),
            "mismatch_in_pct": float(100 * np.abs(resid_in).sum() / np.abs(nd_input_scen).sum()),
            "n_ramp": rs["n"],
            "ramp_mean_mw": rs["mean_mw"],
            "ramp_mean_mwh": rs["mean_mwh"],
            "ramp_max_mw": rs["max_mw"],
            "ramp_total_mwh": rs["total_mwh"],
            "ramp_excess_pct": float(100 * rs["excess_sum_mw"] / np.abs(ps).sum()),
            "n_neg": int((ps < -0.1).sum()),
            "soc_feasible": "yes" if soc_ok else "no",
            "soc_swing_pct": soc_pct,
            "alpha_active_pct": float(100 * (alpha > 1e-6).mean()) if len(alpha) else None,
        }
        rows.append(row)
        print(f"  {name}: base_WAPE={base_wape:.4f} nd_resp={row['nd_resp_pct']:+.1f}% "
              f"ramp={rs['n']} (mean {rs['mean_mwh']:.3f} MWh) neg={row['n_neg']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    note = (
        f"Scenario: FixedPercentageShift rebound={rebound:.4f}%, reduction={reduction:.4f}%, "
        f"free window {FREE_HOURS[0]:02d}:00-{FREE_HOURS[1]:02d}:00. The default value is the "
        f"maximum equal rebound/reduction over val-tail+test under demand cap "
        f"{args.demand_cap:.1f} MW: q_max={qmax:.4f}%. Hist net_demand input is "
        "`demand_mw - wind - solar_utility` because hist demand is already VIC-island adjusted. "
        "Rows use closed-loop free-window rollout; base_WAPE/base_R2 score the no-shift rollout "
        "over the response region, while violation metrics are on the full scenario test rollout."
    )
    write_table(Path(args.out), note, rows)


if __name__ == "__main__":
    main()
