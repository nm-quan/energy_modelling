"""Bottleneck-study demand-scenario rollout — parquet-free (runs on Colab from
the repo clone alone; see colab/study_bottlenecks.ipynb).

Problem 2 of the study: does the model respond to a simulated increased-load
period? Two scenarios on the daily 11:00-14:00 free window, over the full hist
test span (Jan-Jul 2026):

  reshape   FixedPercentageShift(rebound=g, reduction=g): energy-neutral load
            shift + g% induced free-window demand, price zeroed in-window
            (identical semantics to hist_constrained_shift.py).
  increase  demand_mw *= (1+g/100) inside the free window only — a pure load
            increase (heat wave / EV surge). Price untouched, nothing removed.

Everything is built from data/preprocessed/hist/5min/.../prepared.npz — no raw
parquets. The scenario net_demand input is nd_base + (demand_scen − demand_base)
("supply-delta" mode): the base nd is the supply-side series the models were
TRAINED on, so the input stays in-distribution and the shift still propagates
(this removes the ~197 MW demand-side/supply-side offset that confounded the
legacy runs — results.md note 1).

Rollout protocol (same as hist_constrained_shift.py): closed-loop inside the
free window (model feeds back its own dispatch; net_demand/demand_mw/price stay
teacher-forced from the scenario frame), teacher-forced outside it.

Rows: persistence, anchor(persistence) [the +response reference], *_rayen,
*_rayenfd (gas_steam passthrough on by default), *_task7+DR — whatever
checkpoints exist under --weights.

    python3 demand_simulation/study_shift.py --scenario increase --g 10
    python3 demand_simulation/study_shift.py --scenario reshape            # g = q_max
    python3 demand_simulation/study_shift.py --max-steps 288 --device cpu  # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

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
from check_caps import BATT_CAP_MWH, RAMPS, RES_HOURS  # noqa: E402

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]
SIGN = np.array([1, 1, 1, 1, -1, 1], dtype=np.float64)
FREE_HOURS = (11, 14)
DEMAND_CAP_MW = 10783.7          # CONSTRAINT.md historical demand max
NOSTEAM = [0, 1, 3, 5]           # anchor rescale set (findings.md)
RAMP_TOL = 0.6
DEMAND_TOL = 10.0
ETA_RT = 0.834
OUT_DIR = HERE / "sweep_eqnd" / "study"


# ----------------------------- data (npz only) -----------------------------

def load_frame():
    """Rollout frame from prepared.npz: scaled feature flats (test + lb context),
    scalers, feature names, timestamps. No parquets touched."""
    z = np.load(pipeline.PREPROCESSED_DIR / "hist" / "5min" / "net_dispatch_totdem"
                / "prepared.npz", allow_pickle=False)
    fc = [str(c) for c in z["feat_cols"]]
    lb = int(z["lookback_steps"])
    test_index = pd.DatetimeIndex(z["test_index"])
    # context stamps: lb rows immediately before the first test stamp. The flats
    # are the rows build_table produced; a gap here would only mislabel context
    # rows (never predicted) for the day-grouping, so 5-min spacing is safe.
    ctx = test_index[0] - pd.to_timedelta(np.arange(lb, 0, -1) * 5, unit="m")
    full_index = ctx.append(test_index)
    from sklearn.preprocessing import StandardScaler

    def scaler(mean, scale):
        s = StandardScaler()
        s.mean_, s.scale_ = mean, scale
        s.var_, s.n_features_in_ = scale ** 2, len(mean)
        return s

    return {"fs_base": z["Xte_flat"].copy(), "Yte_flat": z["Yte_flat"],
            "x_scaler": scaler(z["x_mean"], z["x_scale"]),
            "y_scaler": scaler(z["y_mean"], z["y_scale"]),
            "feat_cols": fc, "lookback_steps": lb,
            "test_index": test_index, "full_index": full_index}


# ----------------------------- scenario -----------------------------

def max_equal_shift_pct(d: np.ndarray, free: np.ndarray, index, cap_mw: float) -> float:
    """Largest q with rebound = reduction = q %% under the demand cap (the
    hist_constrained_shift.py default): d'_free = d*(1+q) + q*daily_nonfree/n_free."""
    days = pd.DatetimeIndex(index).normalize()
    agg = pd.DataFrame({"day": days, "removed": np.where(~free, d, 0.0),
                        "free": free.astype(float)})
    shifted = agg.groupby("day")["removed"].transform("sum").to_numpy()
    n_free = agg.groupby("day")["free"].transform("sum").to_numpy()
    add = np.where(free & (n_free > 0), shifted / np.where(n_free > 0, n_free, 1.0), 0.0)
    q = np.min((cap_mw - d[free]) / (d[free] + add[free]))
    return float(100.0 * max(0.0, min(1.0, q)))


def build_scenario(frame, scenario: str, g: float, demand_cap: float):
    """Return (fs_scen, info). Operates on the SCALED flats via the x_scaler."""
    xs, fc = frame["x_scaler"], frame["feat_cols"]
    fs = frame["fs_base"]
    full_index = frame["full_index"]
    di, ni = fc.index("demand_mw"), fc.index("net_demand")
    pi = fc.index("price_aud_per_mwh")

    def col_mw(i):
        return fs[:, i].astype(np.float64) * xs.scale_[i] + xs.mean_[i]

    d_base = col_mw(di)
    free_full = np.asarray((full_index.hour >= FREE_HOURS[0])
                           & (full_index.hour < FREE_HOURS[1]))

    if scenario == "increase":
        if g is None:
            g = 10.0
        d_scen = d_base.copy()
        d_scen[free_full] = d_base[free_full] * (1.0 + g / 100.0)
        price_scen = None                                  # price untouched
        g_max = 100.0 * (demand_cap / d_base[free_full].max() - 1.0)
    elif scenario == "reshape":
        from shift_model import FixedPercentageShift
        g_max = max_equal_shift_pct(d_base, free_full, full_index, demand_cap)
        if g is None:                                      # legacy default: q_max
            g = g_max
            print(f"  reshape g defaulting to q_max = {g:.4f}%")
        mini = pd.DataFrame({"demand_mw": d_base, "price_aud_per_mwh": col_mw(pi)},
                            index=full_index)
        shifted = FixedPercentageShift(rebound_pct=g, reduction_pct=g,
                                       free_hours=FREE_HOURS).transform(mini)
        d_scen = shifted["demand_mw"].to_numpy()
        price_scen = shifted["price_aud_per_mwh"].to_numpy()
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    if (d_scen < -1e-6).any() or (d_scen > demand_cap + 1e-6).any():
        raise ValueError(f"scenario demand violates [0, {demand_cap}] MW "
                         f"(max {d_scen.max():.1f}); lower --g (g_max={g_max:.2f}%)")

    fs_scen = fs.copy()
    fs_scen[:, di] = (d_scen - xs.mean_[di]) / xs.scale_[di]
    # supply-delta nd: trained-on base nd + the exogenous demand change
    nd_scen = col_mw(ni) + (d_scen - d_base)
    fs_scen[:, ni] = (nd_scen - xs.mean_[ni]) / xs.scale_[ni]
    if price_scen is not None:
        fs_scen[:, pi] = (price_scen - xs.mean_[pi]) / xs.scale_[pi]
    return fs_scen.astype(np.float32), {"d_base": d_base, "d_scen": d_scen,
                                        "nd_scen": nd_scen, "free_full": free_full,
                                        "g": float(g)}


# ----------------------------- models -----------------------------

def load_entries(weights: Path, frame, device: str, model_filter=None,
                 steam_pt: bool = True):
    """[(name, n_out, d_ref, module)]; d_ref in {own7, nd_window, sign_pred}."""
    xs, ys, fc = frame["x_scaler"], frame["y_scaler"], frame["feat_cols"]
    nd_idx = fc.index("net_demand")
    ramp_dn = [abs(RAMPS[t][0]) for t in TARGETS]
    ramp_up = [RAMPS[t][1] for t in TARGETS]

    pers = M.PersistenceForecaster(xs.mean_, xs.scale_, ys.mean_, ys.scale_)
    anchor = M.DemandAnchoredHead(
        M.PersistenceForecaster(xs.mean_, xs.scale_, ys.mean_, ys.scale_),
        xs.mean_[nd_idx], xs.scale_[nd_idx], ys.mean_, ys.scale_,
        nd_feat_idx=nd_idx, rescale_idx=NOSTEAM, pos_fn="relu")
    entries = [("persistence", 6, "sign_pred", pers.to(device).eval()),
               ("anchor_persistence", 6, "nd_window", anchor.to(device).eval())]

    def _load(pt):
        return torch.load(pt, map_location="cpu", weights_only=True)

    for base_arch in ("lstm", "itransformer"):
        pt = weights / f"{base_arch}_rayen_hist_s0.pt"
        if pt.exists():
            m = M.make_rayen(base_arch, xs, ys, ramp_up, ramp_dn,
                             nd_feat_idx=nd_idx, n_features=len(fc))
            m.load_state_dict(_load(pt))
            entries.append((f"{base_arch}_rayen", 7, "own7", m.to(device).eval()))

        pt = weights / f"{base_arch}_rayenfd_hist_s0.pt"
        if pt.exists():
            m = M.make_rayen(base_arch, xs, ys, ramp_up, ramp_dn,
                             nd_feat_idx=nd_idx, n_features=len(fc), fix_demand=True,
                             passthrough_idx=(2,) if steam_pt else None)
            missing, _ = m.load_state_dict(_load(pt), strict=False)  # legacy ckpts lack _free
            free = torch.ones(6)
            if steam_pt:
                free[2] = 0.0
            m._free.copy_(free)
            m._sign_free.copy_(torch.tensor([1., 1., 1., 1., -1., 1.]) * free)
            name = f"{base_arch}_rayenfd" + ("+spt" if steam_pt else "")
            entries.append((name, 6, "nd_window", m.to(device).eval()))

        pt = weights / f"{base_arch}_task7_hist_s0.pt"
        fz = weights / f"{base_arch}_task7_hist_s0_safeF.npz"
        if pt.exists() and fz.exists():
            task = M.make_task7(base_arch, n_features=len(fc), nd_feat_idx=nd_idx)
            task.load_state_dict(_load(pt))
            z = np.load(fz)
            m = dr.DecisionRuleHead(task, z["F"], xs.mean_, xs.scale_,
                                    ys.mean_, ys.scale_, ramp_up, ramp_dn,
                                    nd_feat_idx=nd_idx)
            entries.append((f"{base_arch}_task7+DR", 7, "own7", m.to(device).eval()))

    if model_filter:
        entries = [e for e in entries if any(f in e[0] for f in model_filter)]
        if not entries:
            raise ValueError(f"no loaded model matches --models {model_filter}")
    return entries


# ----------------------------- rollout -----------------------------

def rollout(model, n_out: int, d_ref: str, fs_base, fs_scen, lb: int,
            free_test: np.ndarray, nd_idx: int, xs, ys, device: str):
    """Closed-loop free-window rollout, base + scenario stacked (batch 2)."""
    fs_t = torch.from_numpy(np.stack([fs_base, fs_scen])).to(device)
    ymean = torch.tensor(ys.mean_, dtype=torch.float32, device=device)
    yscale = torch.tensor(ys.scale_, dtype=torch.float32, device=device)
    nd_mean, nd_scale = float(xs.mean_[nd_idx]), float(xs.scale_[nd_idx])
    xmean_t = torch.tensor(xs.mean_[M.TARGET_FEAT_IDX], dtype=torch.float32, device=device)
    xscale_t = torch.tensor(xs.scale_[M.TARGET_FEAT_IDX], dtype=torch.float32, device=device)
    tfi = torch.tensor(M.TARGET_FEAT_IDX, dtype=torch.long, device=device)
    sign = torch.tensor(SIGN, dtype=torch.float32, device=device)

    n = len(free_test)
    preds = torch.zeros((2, n, 6), dtype=torch.float32, device=device)
    d_own = torch.zeros((2, n), dtype=torch.float32, device=device)
    window, free_prev = None, False
    with torch.no_grad():
        for i in range(n):
            free_now = bool(free_test[i])
            if not (free_now and free_prev):
                window = fs_t[:, i:i + lb, :].clone()      # teacher-forced re-read
            out = model(window)
            p_scaled = out[:, 1:] if n_out == 7 else out
            pred_mw = p_scaled * yscale + ymean
            preds[:, i, :] = pred_mw
            if n_out == 7:
                d_own[:, i] = out[:, 0] * nd_scale + nd_mean
            elif d_ref == "nd_window":
                d_own[:, i] = window[:, -1, nd_idx] * nd_scale + nd_mean
            else:
                d_own[:, i] = pred_mw @ sign
            if free_now:
                newrow = fs_t[:, i + lb, :].clone()        # exogenous stays scenario
                newrow[:, tfi] = (pred_mw - xmean_t) / xscale_t
                window = torch.cat([window[:, 1:, :], newrow[:, None, :]], dim=1)
            free_prev = free_now
    return (preds[0].cpu().numpy().astype(np.float64),
            preds[1].cpu().numpy().astype(np.float64),
            d_own[0].cpu().numpy().astype(np.float64),
            d_own[1].cpu().numpy().astype(np.float64))


# ----------------------------- metrics -----------------------------

def ramp_split(pred, prev_first, consec, free_test):
    """Ramp violations on the predicted trajectory, split by rollout position:
    free (closed-loop window incl. first step), seam (first teacher-forced step
    after a window — the frozen-model reconnection snap), tf (everything else)."""
    ramp_up = np.array([RAMPS[t][1] for t in TARGETS])
    ramp_dn = np.array([abs(RAMPS[t][0]) for t in TARGETS])
    delta = np.diff(np.concatenate([prev_first[None], pred]), axis=0)
    viol = ((delta > ramp_up + RAMP_TOL) | (delta < -(ramp_dn + RAMP_TOL))) & consec[:, None]
    ex = (np.clip(delta - (ramp_up + RAMP_TOL), 0, None)
          + np.clip(-delta - (ramp_dn + RAMP_TOL), 0, None)) * consec[:, None]
    seam = np.zeros(len(pred), dtype=bool)
    seam[1:] = free_test[:-1] & ~free_test[1:]
    n_free = int(viol[free_test].sum())
    n_seam = int(viol[seam].sum())
    n_tf = int(viol.sum()) - n_free - n_seam
    return {"n_ramp": int(viol.sum()), "n_ramp_free": n_free, "n_ramp_seam": n_seam,
            "n_ramp_tf": n_tf, "ramp_excess_sum_mw": float(ex.sum()),
            "ramp_max_mw": float(ex.max())}


def soc_report(pred, index, consec):
    """Per-day SOC feasibility (the physical unit — batteries cycle daily) plus
    the legacy whole-window swing for transparency (drift artifact, see
    constraint_research.md finding 2)."""
    eta = np.sqrt(ETA_RT)
    chg = np.clip(pred[:, TARGETS.index("battery_charging")], 0, None)
    dis = np.clip(pred[:, TARGETS.index("battery_discharging")], 0, None)
    dE = (chg * eta - dis / eta) * RES_HOURS["5min"]
    day = pd.DatetimeIndex(index).normalize().to_numpy()
    brk = ~consec
    brk[0] = True
    brk[1:] |= day[1:] != day[:-1]
    seg = np.cumsum(brk)
    swings = []
    for s in np.unique(seg):
        cum = np.concatenate([[0.0], np.cumsum(dE[seg == s])])
        swings.append(cum.max() - cum.min())
    swings = np.array(swings)
    cum_all = np.concatenate([[0.0], np.cumsum(dE)])
    return {"soc_day_feasible_pct": float(100 * (swings <= BATT_CAP_MWH).mean()),
            "soc_worst_day_pct": float(100 * swings.max() / BATT_CAP_MWH),
            "soc_window_swing_pct": float(100 * (cum_all.max() - cum_all.min()) / BATT_CAP_MWH)}


# ----------------------------- reporting -----------------------------

COLS = ["model", "base_WAPE", "base_R2", "demand_in_pct", "nd_resp_pct",
        "response_capture", "coal_resp_pct", "hydro_resp_pct", "ocgt_resp_pct",
        "batt_dis_resp_pct", "track_free_p50_mw", "track_free_p95_mw",
        "bal_own_max_mw", "mismatch_in_pct", "n_ramp", "n_ramp_free",
        "n_ramp_seam", "n_ramp_tf", "ramp_max_mw", "n_neg",
        "soc_day_feasible_pct", "soc_worst_day_pct", "soc_window_swing_pct", "secs"]


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{float(v):.4f}" if abs(float(v)) < 10 else f"{float(v):.1f}"


def rebuild_md(tag: str, note: str):
    rows = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob(f"{tag}_*.json"))]
    lines = [f"# Study demand-scenario rollout — {tag}\n", note + "\n",
             "| " + " | ".join(COLS) + " |",
             "| " + " | ".join("---" for _ in COLS) + " |"]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in COLS) + " |" for r in rows]
    path = OUT_DIR / f"{tag}.md"
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["reshape", "increase"], default="increase")
    ap.add_argument("--g", type=float, default=None,
                    help="reshape: rebound=reduction=g %% (default: q_max under the cap); "
                         "increase: +g%% free-window demand (default 10)")
    ap.add_argument("--demand-cap", type=float, default=DEMAND_CAP_MW)
    ap.add_argument("--weights", default=str(ROOT / "weights"))
    ap.add_argument("--models", default=None, help="comma-separated substrings")
    ap.add_argument("--rayenfd-steam-pt", choices=["on", "off"], default="on")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke-test cap")
    args = ap.parse_args()
    device = ev.pick_device(args.device)
    print(f"device={device} scenario={args.scenario} g={args.g}")

    frame = load_frame()
    xs, ys, fc = frame["x_scaler"], frame["y_scaler"], frame["feat_cols"]
    lb, nd_idx = frame["lookback_steps"], fc.index("net_demand")
    if args.max_steps is not None:
        n = args.max_steps
        frame["fs_base"] = frame["fs_base"][:n + lb]
        frame["Yte_flat"] = frame["Yte_flat"][:n + lb]
        frame["test_index"] = frame["test_index"][:n]
        frame["full_index"] = frame["full_index"][:n + lb]
    ti = frame["test_index"]
    fs_scen, info = build_scenario(frame, args.scenario, args.g, args.demand_cap)
    g = info["g"]

    free_test = np.asarray((ti.hour >= FREE_HOURS[0]) & (ti.hour < FREE_HOURS[1]))
    tprev = ti - pd.Timedelta(minutes=5)
    mask = np.asarray((tprev.hour >= FREE_HOURS[0]) & (tprev.hour < FREE_HOURS[1]))
    consec = np.ones(len(ti), dtype=bool)
    consec[0] = False
    consec[1:] = (np.diff(ti.values) == np.timedelta64(5, "m"))
    if not consec[1:].all():
        print(f"  note: {int((~consec[1:]).sum())} index gaps; ramps/SOC segmented")

    actual = ys.inverse_transform(frame["Yte_flat"][lb:]).astype(np.float64)
    nd_in_base = (frame["fs_base"][lb:, nd_idx].astype(np.float64)
                  * xs.scale_[nd_idx] + xs.mean_[nd_idx])
    nd_in_scen = info["nd_scen"][lb:]
    d_base_t, d_scen_t = info["d_base"][lb:], info["d_scen"][lb:]
    prev0 = (frame["fs_base"][lb - 1, M.TARGET_FEAT_IDX].astype(np.float64)
             * xs.scale_[M.TARGET_FEAT_IDX] + xs.mean_[M.TARGET_FEAT_IDX])

    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    entries = load_entries(Path(args.weights), frame, device, model_filter,
                           steam_pt=args.rayenfd_steam_pt == "on")
    print("models:", [e[0] for e in entries])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.scenario}_g{g:g}"

    for name, n_out, d_ref, model in entries:
        t0 = time.time()
        pb, ps, db, ds = rollout(model, n_out, d_ref, frame["fs_base"], fs_scen,
                                 lb, free_test, nd_idx, xs, ys, device)
        b_resp, s_resp = pb[mask], ps[mask]
        # per-target WAPE with the eval_hist_models denominator floor: require
        # mean |actual| >= 1 MW in the region, else the channel's near-zero
        # denominator (gas_steam midday, batteries on quiet days) explodes it
        act_r = actual[mask]
        per = ev.compute_metrics(act_r, b_resp, TARGETS)["per_target"]
        keep = [t for i, t in enumerate(TARGETS)
                if np.abs(act_r[:, i]).sum() >= len(act_r)]
        base_wape = float(np.mean([per[t]["WAPE"] for t in keep]))
        base_r2 = float(np.mean([per[t]["R2"] for t in keep]))
        nd_b, nd_s = b_resp @ SIGN, s_resp @ SIGN
        in_delta = float(nd_in_scen[mask].mean() - nd_in_base[mask].mean())
        capture = float((nd_s.mean() - nd_b.mean()) / in_delta) if abs(in_delta) > 1e-9 else None
        resp = {}
        for i, t in enumerate(TARGETS):
            den = b_resp[:, i].mean()
            resp[t] = float(100 * (s_resp[:, i].mean() - den) / den) if abs(den) > 1e-9 else None
        track = np.abs(ps @ SIGN - nd_in_scen)
        resid_in = ps @ SIGN - nd_in_scen
        rs = ramp_split(ps, prev0, consec, free_test)
        row = {"model": name, "scenario": args.scenario, "g": g, "n": len(ti),
               "base_WAPE": base_wape, "base_R2": base_r2,
               "demand_in_pct": float(100 * (d_scen_t[mask].mean() - d_base_t[mask].mean())
                                      / d_base_t[mask].mean()),
               "nd_resp_pct": float(100 * (nd_s.mean() - nd_b.mean()) / nd_b.mean()),
               "response_capture": capture,
               "coal_resp_pct": resp["coal_brown"], "hydro_resp_pct": resp["hydro"],
               "ocgt_resp_pct": resp["gas_ocgt"],
               "batt_dis_resp_pct": resp["battery_discharging"],
               "track_free_p50_mw": float(np.percentile(track[mask], 50)),
               "track_free_p95_mw": float(np.percentile(track[mask], 95)),
               "bal_own_max_mw": float(np.abs(ps @ SIGN - ds).max()),
               "n_demand_in": int((np.abs(resid_in) > DEMAND_TOL).sum()),
               "mismatch_in_pct": float(100 * np.abs(resid_in).sum() / np.abs(nd_in_scen).sum()),
               **rs, "n_neg": int((ps < -0.1).sum()),
               **soc_report(ps, ti, consec), "secs": round(time.time() - t0, 1)}
        safe = name.replace("+", "_")
        (OUT_DIR / f"{tag}_{safe}.json").write_text(json.dumps(row, indent=2))
        print(f"  {name:24s} base_WAPE={base_wape:.4f} capture="
              f"{'—' if capture is None else f'{capture:+.3f}'} "
              f"track_p50={row['track_free_p50_mw']:.0f} MW ramp={rs['n_ramp']} "
              f"(seam {rs['n_ramp_seam']}) soc_day={row['soc_day_feasible_pct']:.0f}% "
              f"({row['secs']}s)", flush=True)

    note = (f"Scenario **{args.scenario}**, g={g:g}%, free window "
            f"{FREE_HOURS[0]:02d}:00-{FREE_HOURS[1]:02d}:00, demand cap {args.demand_cap:.1f} MW, "
            f"n={len(ti)} steps. nd input = supply-delta (base nd + demand change; "
            "in-distribution). response_capture = (nd_scen - nd_base rollout mean delta) / "
            "(input nd mean delta) over the response region — 1.0 = fleet delivers the full "
            "simulated increase. track_free = |SIGN·pred − nd_scen_input| in the response "
            "region. Ramp counts split: free window / seam (first TF step after the window; "
            "a frozen model snaps here) / rest. SOC per calendar day + legacy whole-window "
            "swing (drift artifact, transparency only).")
    rebuild_md(tag, note)


if __name__ == "__main__":
    main()
