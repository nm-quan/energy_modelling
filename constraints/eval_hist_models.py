"""Unified constraint evaluation for the hist-trained models (plan.md).

Two tables, matching the evaluation protocol:

  TF  (demand balance -- regular test set, teacher-forced): WAPE/R2, D-forecast
      quality (nd_WAPE), balance vs own D (@own: max residual + steps > 10 MW)
      and vs actual nd(t) (@act: steps > 10 MW + mismatch %), negativity, ramp
      vs previous actuals, per-day battery SOC feasibility.
  CL  (ramp rate -- the simulated test set): closed-loop autoregressive rollout
      over the TEST-split stress episodes (constraints/stress_episodes.json,
      never seen in training): each model feeds back its own dispatch into the
      input window; exogenous features (net_demand, demand, price, calendar)
      stay actual. Ramp violations are counted on the model's own trajectory,
      which is what ramp feasibility means in deployment.

Percentage-violation conventions ("raw MWh -> %"):
  ramp_excess_pct   100 * sum(MW beyond the ramp limit) / sum(|predicted MW|)
  mismatch_act_pct  100 * sum(|SIGN.pred - nd_act|) / sum(|nd_act|)

Rows: persistence benchmark + every checkpoint found among
{lstm,itransformer}_{rayen,task7} (task7 also gets a +DR row = post-hoc
decision-rules blend, lib/decision_rule.py). Works with both checkpoint
layouts: one flat dir (Colab --out) or ml/<arch>_hist/ subdirs.

  python3 constraints/eval_hist_models.py                      # local layout
  python3 constraints/eval_hist_models.py --ckpt-dir $OUT      # Colab
  python3 constraints/eval_hist_models.py --max-steps 48 --tf-stride 100  # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent          # constraints/
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "ml"))
import pipeline                  # noqa: E402
import models as M               # noqa: E402
import evaluate as ev            # noqa: E402
import decision_rule as dr       # noqa: E402
from check_caps import RAMPS, CAPS, BATT_CAP_MWH, RES_HOURS  # noqa: E402

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]
SIGN = np.array([1, 1, 1, 1, -1, 1], dtype=np.float64)
ARCHS = ["lstm_rayen", "itransformer_rayen", "lstm_rayenfd", "itransformer_rayenfd",
         "lstm_task7", "itransformer_task7"]
RAMP_TOL = 0.6                   # eps-anchor slack + float32 noise floor (MW)
DEMAND_TOL = 10.0                # balance violation threshold (MW)
RESULTS = HERE / "results"


# ----------------------------- model loading -----------------------------

def _find(ckpt_dir: Path, arch: str, seed: int, ext: str = ".pt") -> Path | None:
    name = f"{arch}_hist_s{seed}{ext}"
    for p in (ckpt_dir / name, ROOT / "ml" / f"{arch}_hist" / name):
        if p.exists():
            return p
    return None


def load_entries(ckpt_dir: Path, data: dict, seed: int, device: str,
                 rayenfd_steam_pt: bool = True) -> list:
    """[(name, n_out, module)] for persistence + every checkpoint on disk.

    rayenfd_steam_pt: force the gas_steam passthrough on the RayenHeadFixedD
    rows (freeze the peaker at persistence). The buffers are overwritten AFTER
    load_state_dict, so the flag wins over whatever the checkpoint was trained
    with -- this is the zero-retrain "A-prime" retrofit from the study ladder.
    Legacy rayenfd checkpoints predate the _free/_sign_free buffers, hence
    strict=False (missing buffers keep the constructor's values).
    """
    xs, ys, fc = data["x_scaler"], data["y_scaler"], data["feat_cols"]
    nd_idx = fc.index("net_demand")
    ramp_dn = [abs(RAMPS[t][0]) for t in TARGETS]
    ramp_up = [RAMPS[t][1] for t in TARGETS]
    entries = [("persistence", 6,
                M.PersistenceForecaster(xs.mean_, xs.scale_, ys.mean_, ys.scale_).to(device).eval())]
    for arch in ARCHS:
        pt = _find(ckpt_dir, arch, seed)
        if pt is None:
            print(f"  [skip] no checkpoint for {arch}")
            continue
        base_arch = arch.rsplit("_", 1)[0]
        name, n_out = arch, 7
        if arch.endswith("_rayenfd"):
            m = M.make_rayen(base_arch, xs, ys, ramp_up, ramp_dn,
                             nd_feat_idx=nd_idx, n_features=len(fc), fix_demand=True,
                             passthrough_idx=(2,) if rayenfd_steam_pt else None)
            missing, _ = m.load_state_dict(
                torch.load(pt, map_location="cpu", weights_only=True), strict=False)
            if missing:
                print(f"  {arch}: legacy checkpoint, constructor buffers kept for {missing}")
            free = torch.ones(6)
            if rayenfd_steam_pt:
                free[2] = 0.0
                name = arch + "+spt"
            m._free.copy_(free)
            m._sign_free.copy_(torch.tensor([1., 1., 1., 1., -1., 1.]) * free)
            n_out = 6
        elif arch.endswith("_rayen"):
            m = M.make_rayen(base_arch, xs, ys, ramp_up, ramp_dn,
                             nd_feat_idx=nd_idx, n_features=len(fc))
            m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
        else:
            m = M.make_task7(base_arch, n_features=len(fc), nd_feat_idx=nd_idx)
            m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
        m = m.to(device).eval()
        entries.append((name, n_out, m))
        if arch.endswith("_task7"):
            fz = _find(ckpt_dir, arch, seed, "_safeF.npz")
            if fz is not None:
                z = np.load(fz)
                F, t_star = z["F"], float(z["t"])
            else:                                   # LP is seconds -- fit fresh
                nd_col = data["Xtr"][:, -1, nd_idx] * xs.scale_[nd_idx] + xs.mean_[nd_idx]
                span = nd_col.max() - nd_col.min()
                fit = dr.fit_safe_F(ramp_up, ramp_dn, [1.3 * CAPS[t] for t in TARGETS],
                                    float(nd_col.min() - 0.25 * span),
                                    float(nd_col.max() + 0.25 * span))
                F, t_star = fit["F"], fit["t"]
            wrapped = dr.DecisionRuleHead(m, F, xs.mean_, xs.scale_, ys.mean_, ys.scale_,
                                          ramp_up, ramp_dn, nd_feat_idx=nd_idx).to(device).eval()
            print(f"  {arch}+DR: safe LP t*={t_star:.2f} MW")
            entries.append((arch + "+DR", 7, wrapped))
    return entries


# ----------------------------- shared metrics -----------------------------

def _predict(model, X, device, batch=256):
    outs, alphas = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(device)
            outs.append(model(xb).cpu().numpy())
            if getattr(model, "last_alpha", None) is not None:
                alphas.append(model.last_alpha.cpu().numpy())
    return np.concatenate(outs), (np.concatenate(alphas) if alphas else None)


def _split7(pred_raw, n_out, xs, ys, nd_idx):
    """scaled model output -> (pred6 MW, own-D MW or None)"""
    if n_out == 7:
        d_mw = pred_raw[:, 0] * xs.scale_[nd_idx] + xs.mean_[nd_idx]
        return ys.inverse_transform(pred_raw[:, 1:]).astype(np.float64), d_mw.astype(np.float64)
    return ys.inverse_transform(pred_raw).astype(np.float64), None


def _ramp_counts(pred, prev_first, ramp_up, ramp_dn):
    """Ramp violations on a contiguous trajectory (first delta vs prev_first)."""
    delta = np.diff(np.concatenate([prev_first[None], pred]), axis=0)
    up_ex = np.clip(delta - (ramp_up + RAMP_TOL), 0, None)
    dn_ex = np.clip(-delta - (ramp_dn + RAMP_TOL), 0, None)
    ex = up_ex + dn_ex
    return int((ex > 0).sum()), float(ex.sum())


def _soc_swings(pred, segs, eta_rt, dt):
    """Per-segment reservoir swing (MWh) of predicted battery operation."""
    eta = np.sqrt(eta_rt)
    chg = np.clip(pred[:, TARGETS.index("battery_charging")], 0, None)
    dis = np.clip(pred[:, TARGETS.index("battery_discharging")], 0, None)
    dE = (chg * eta - dis / eta) * dt
    out = []
    for s in np.unique(segs):
        cum = np.concatenate([[0.0], np.cumsum(dE[segs == s])])
        out.append(float(cum.max() - cum.min()))
    return np.array(out)


# ----------------------------- TF evaluation -----------------------------

def tf_eval(entries, data, device, stride, eta_rt):
    xs, ys = data["x_scaler"], data["y_scaler"]
    nd_idx = data["feat_cols"].index("net_demand")
    Xte, Yte = data["Xte"][::stride], data["Yte"][::stride]
    idx = data["test_index"][::stride]
    true = ys.inverse_transform(Yte).astype(np.float64)
    nd_act = true @ SIGN
    prev = (Xte[:, -1, M.TARGET_FEAT_IDX] * xs.scale_[M.TARGET_FEAT_IDX]
            + xs.mean_[M.TARGET_FEAT_IDX]).astype(np.float64)
    nd_in = (Xte[:, -1, nd_idx] * xs.scale_[nd_idx] + xs.mean_[nd_idx]).astype(np.float64)
    ramp_up = np.array([RAMPS[t][1] for t in TARGETS])
    ramp_dn = np.array([abs(RAMPS[t][0]) for t in TARGETS])
    days = pd.DatetimeIndex(idx).normalize()
    segs = np.unique(days, return_inverse=True)[1]
    dt = RES_HOURS["5min"] * stride            # stride>1 stretches SOC steps; smoke only

    rows = []
    for name, n_out, model in entries:
        pred_raw, alpha = _predict(model, Xte, device)
        pred, d_own = _split7(pred_raw, n_out, xs, ys, nd_idx)
        if d_own is None:
            # 6-output rows never predict D. persistence: own D = SIGN.prev =
            # nd(t-1) by the hist identity; rayenfd: the plane is PINNED to
            # nd(t-1) read off the window -- same reference for both.
            d_own = nd_in
        met_all = ev.compute_metrics(true, pred, TARGETS)
        met = met_all["average"]
        bal = np.abs(pred @ SIGN - d_own)
        resid_act = pred @ SIGN - nd_act
        delta = pred - prev
        n_ramp = int(((delta > ramp_up + RAMP_TOL) | (delta < -(ramp_dn + RAMP_TOL))).sum())
        swings = _soc_swings(pred, segs, eta_rt, dt)
        row = {"model": name, "n": len(pred),
               "WAPE": met["WAPE"], "R2": met["R2"],
               "per_target_WAPE": {t: met_all["per_target"][t]["WAPE"] for t in TARGETS},
               "nd_WAPE": float(np.abs(d_own - nd_act).sum() / np.abs(nd_act).sum()),
               "bal_own_max_mw": float(bal.max()),
               "n_demand_own": int((bal > DEMAND_TOL).sum()),
               "n_demand_act": int((np.abs(resid_act) > DEMAND_TOL).sum()),
               "mismatch_act_pct": float(100 * np.abs(resid_act).sum() / np.abs(nd_act).sum()),
               "n_neg": int((pred < -0.1).sum()),
               "n_ramp_vs_prev": n_ramp,
               "soc_day_feasible_pct": float(100 * (swings <= BATT_CAP_MWH).mean()),
               "soc_worst_day_pct": float(100 * swings.max() / BATT_CAP_MWH)}
        if alpha is not None:
            row["alpha_frac_active"] = float((alpha > 1e-6).mean())
            row["alpha_max"] = float(alpha.max())
        rows.append(row)
        print(f"  TF {name:22s} WAPE={row['WAPE']:.4f} nd_WAPE={row['nd_WAPE']:.4f} "
              f"bal_own={row['bal_own_max_mw']:.3f} dem_act={row['n_demand_act']} "
              f"({row['mismatch_act_pct']:.2f}%) neg={row['n_neg']} ramp={n_ramp} "
              f"soc_day={row['soc_day_feasible_pct']:.0f}%", flush=True)
    return rows


# ----------------------------- CL stress rollout -----------------------------

def load_episodes(data, min_steps=1):
    eps = json.loads((HERE / "stress_episodes.json").read_text())
    idx, lb = data["test_index"], data["lookback_steps"]
    out = []
    for e in eps:
        if e["split"] != "test":
            continue
        i0 = int(idx.searchsorted(pd.Timestamp(e["start"], tz=idx.tz)))
        i1 = int(idx.searchsorted(pd.Timestamp(e["end_excl"], tz=idx.tz)))
        if i1 - i0 < min_steps:
            continue
        span = idx[max(i0 - lb, 0):i1]
        if not (np.diff(span.values) == np.timedelta64(5, "m")).all():
            print(f"  [skip] episode {e['id']}: gap in test index")
            continue
        out.append({**e, "i0": i0, "i1": i1})
    return out


def cl_eval(entries, data, device, episodes, eta_rt, max_steps=None):
    xs, ys = data["x_scaler"], data["y_scaler"]
    nd_idx = data["feat_cols"].index("net_demand")
    lb = data["lookback_steps"]
    z = np.load(pipeline.PREPROCESSED_DIR / "hist" / "5min" / "net_dispatch_totdem"
                / "prepared.npz", allow_pickle=False)
    Xte_flat, Yte_flat = z["Xte_flat"], z["Yte_flat"]
    tfi = np.array(M.TARGET_FEAT_IDX)
    x_mu, x_sd = xs.mean_[tfi], xs.scale_[tfi]
    ramp_up = np.array([RAMPS[t][1] for t in TARGETS])
    ramp_dn = np.array([abs(RAMPS[t][0]) for t in TARGETS])
    steps = [min(e["i1"] - e["i0"], max_steps or 10 ** 9) for e in episodes]
    smax = max(steps)
    actual = np.concatenate([ys.inverse_transform(Yte_flat[lb + e["i0"]:lb + e["i0"] + n])
                             for e, n in zip(episodes, steps)]).astype(np.float64)
    segs = np.concatenate([np.full(n, k) for k, n in enumerate(steps)])
    prev0 = {k: (Xte_flat[e["i0"] + lb - 1, tfi] * x_sd + x_mu).astype(np.float64)
             for k, e in enumerate(episodes)}

    rows = []
    for name, n_out, model in entries:
        is_fd = "rayenfd" in name        # own D = the exogenous nd(t-1) the plane is pinned to
        blocks = [Xte_flat[e["i0"]:e["i0"] + lb + n].copy() for e, n in zip(episodes, steps)]
        preds = [np.empty((n, 6), dtype=np.float64) for n in steps]
        d_owns = [np.empty(n, dtype=np.float64) for n in steps]
        alphas = []
        with torch.no_grad():
            for s in range(smax):
                act = [k for k, n in enumerate(steps) if s < n]
                xb = torch.from_numpy(np.stack([blocks[k][s:s + lb] for k in act])).to(device)
                out = model(xb).cpu().numpy()
                if getattr(model, "last_alpha", None) is not None:
                    alphas.append(model.last_alpha.cpu().numpy())
                p6_scaled = out[:, 1:] if n_out == 7 else out
                mw = ys.inverse_transform(p6_scaled)
                for j, k in enumerate(act):
                    preds[k][s] = mw[j]
                    if n_out == 7:
                        d_owns[k][s] = out[j, 0] * xs.scale_[nd_idx] + xs.mean_[nd_idx]
                    elif is_fd:
                        d_owns[k][s] = (blocks[k][s + lb - 1, nd_idx] * xs.scale_[nd_idx]
                                        + xs.mean_[nd_idx])
                    else:
                        d_owns[k][s] = mw[j] @ SIGN
                    blocks[k][s + lb, tfi] = (mw[j] - x_mu) / x_sd   # feed back own dispatch
        pred = np.concatenate(preds)
        d_own = np.concatenate(d_owns)
        nd_act = actual @ SIGN

        # per-target WAPE with a denominator floor: float32 scaler round trips
        # turn gas_steam's exact zeros into +-1e-4 MW, and over short episodes
        # that dust is the whole denominator -- require >= 1 MW mean |actual|
        wapes = []
        for i in range(6):
            denom = np.abs(actual[:, i]).sum()
            if denom >= len(actual):
                wapes.append(np.abs(pred[:, i] - actual[:, i]).sum() / denom)
        wape = float(np.mean(wapes))
        n_ramp, ex_sum = 0, 0.0
        for k, n in enumerate(steps):
            c, s_ = _ramp_counts(preds[k], prev0[k], ramp_up, ramp_dn)
            n_ramp += c; ex_sum += s_
        resid_act = pred @ SIGN - nd_act
        bal = np.abs(pred @ SIGN - d_own)
        swings = _soc_swings(pred, segs, eta_rt, RES_HOURS["5min"])
        row = {"model": name, "n": len(pred), "n_episodes": len(episodes),
               "cl_WAPE": wape,
               "n_ramp": n_ramp,
               "ramp_excess_pct": float(100 * ex_sum / np.abs(pred).sum()),
               "bal_own_max_mw": float(bal.max()),
               "n_demand_act": int((np.abs(resid_act) > DEMAND_TOL).sum()),
               "mismatch_act_pct": float(100 * np.abs(resid_act).sum() / np.abs(nd_act).sum()),
               "n_neg": int((pred < -0.1).sum()),
               "soc_feasible_eps": int((swings <= BATT_CAP_MWH).sum()),
               "soc_worst_ep_pct": float(100 * swings.max() / BATT_CAP_MWH)}
        if alphas:
            a = np.concatenate(alphas)
            row["alpha_frac_active"] = float((a > 1e-6).mean())
            row["alpha_max"] = float(a.max())
        rows.append(row)
        print(f"  CL {name:22s} WAPE={wape:.4f} ramp={n_ramp} "
              f"({row['ramp_excess_pct']:.4f}%) dem_act={row['n_demand_act']} "
              f"({row['mismatch_act_pct']:.2f}%) neg={row['n_neg']} "
              f"SOC {row['soc_feasible_eps']}/{len(episodes)}", flush=True)
    return rows


# ----------------------------- reporting -----------------------------

def write_md(path: Path, title: str, note: str, rows: list[dict], cols: list[str]):
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)
    lines = [f"# {title}\n", note + "\n",
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


def write_per_channel_md(path: Path, rows: list[dict]):
    """Per-channel WAPE table (study problem 1): where does each model lose to
    persistence? Delta column = macro WAPE - persistence macro WAPE."""
    pers = next((r for r in rows if r["model"] == "persistence"), None)
    cols = ["model"] + TARGETS + ["macro_WAPE", "delta_vs_persistence"]
    lines = ["# Hist teacher-forced per-channel WAPE\n",
             "Same run as hist_tf.md. delta_vs_persistence = macro WAPE minus the "
             "persistence row's. Persistence is the h=1 floor; the study gate asks "
             "for parity (<= +0.005 macro) with the residual gap isolated to the "
             "battery channels.\n",
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        pt = r.get("per_target_WAPE")
        if pt is None:
            continue
        delta = r["WAPE"] - pers["WAPE"] if pers else float("nan")
        cells = [r["model"]] + [f"{pt[t]:.4f}" for t in TARGETS] \
            + [f"{r['WAPE']:.4f}", f"{delta:+.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(ROOT / "ml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eta-rt", type=float, default=0.834,
                    help="battery round-trip efficiency (last365 calibration)")
    ap.add_argument("--tf-stride", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None, help="cap CL episode steps (smoke)")
    ap.add_argument("--skip-tf", action="store_true")
    ap.add_argument("--skip-cl", action="store_true")
    ap.add_argument("--rayenfd-steam-pt", choices=["on", "off"], default="on",
                    help="force the gas_steam persistence passthrough on rayenfd rows "
                         "(study ladder A-prime; 'off' evaluates the head as trained)")
    args = ap.parse_args()
    device = ev.pick_device(args.device)

    data = pipeline.load_prepared("5min", 24, 1, "net_dispatch_totdem", "hist")
    entries = load_entries(Path(args.ckpt_dir), data, args.seed, device,
                           rayenfd_steam_pt=args.rayenfd_steam_pt == "on")
    print(f"models: {[n for n, _, _ in entries]}  device={device}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    if not args.skip_tf:
        rows = tf_eval(entries, data, device, args.tf_stride, args.eta_rt)
        for r in rows:
            (RESULTS / f"hist_tf_{r['model'].replace('+', '_')}.json").write_text(json.dumps(r, indent=2))
        write_md(RESULTS / "hist_tf.md", "Hist teacher-forced test (demand balance)",
                 f"Test = Jan-Jul 2026 (stride {args.tf_stride}). @own = vs the model's own "
                 f"D_t (mechanism check, threshold {DEMAND_TOL} MW); @act = vs actual nd(t). "
                 "nd_WAPE = D_t forecast quality. SOC per calendar day, "
                 f"eta_rt={args.eta_rt}, cap {BATT_CAP_MWH:.0f} MWh.",
                 rows, ["model", "WAPE", "R2", "nd_WAPE", "bal_own_max_mw", "n_demand_own",
                        "n_demand_act", "mismatch_act_pct", "n_neg", "n_ramp_vs_prev",
                        "soc_day_feasible_pct", "soc_worst_day_pct"])
        write_per_channel_md(RESULTS / "hist_tf_per_channel.md", rows)

    if not args.skip_cl:
        episodes = load_episodes(data)
        print(f"CL: {len(episodes)} test-split stress episodes "
              f"({sum(e['i1'] - e['i0'] for e in episodes):,} steps)")
        rows = cl_eval(entries, data, device, episodes, args.eta_rt, args.max_steps)
        for r in rows:
            (RESULTS / f"hist_cl_{r['model'].replace('+', '_')}.json").write_text(json.dumps(r, indent=2))
        write_md(RESULTS / "hist_cl_stress.md", "Hist closed-loop stress episodes (ramp rate)",
                 "Autoregressive rollout over TEST-split stress episodes; models feed back "
                 "their own dispatch, exogenous features stay actual. Ramps on the model's own "
                 f"trajectory (asymmetric data limits + {RAMP_TOL} MW tol); ramp_excess_pct = "
                 "100*sum(MW beyond limit)/sum(|pred|). SOC per episode.",
                 rows, ["model", "cl_WAPE", "n_ramp", "ramp_excess_pct", "bal_own_max_mw",
                        "n_demand_act", "mismatch_act_pct", "n_neg", "soc_feasible_eps",
                        "soc_worst_ep_pct"])


if __name__ == "__main__":
    main()
