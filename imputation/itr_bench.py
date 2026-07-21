"""D1 tables for the transformer-imputer ablation.

  --stage1   training table: T1-T4 test MAE per gap length (3h/6h/12h/blackout,
             raw fills) + each arm's val blackout-MAE; picks T* = lowest val
             blackout-MAE. Ties: arms within 5%% relative of the winner are
             FLAGGED AS TIES, not wins (single-seed runs vary a few %%).
  --stage2   constraint table: C0 (T* raw) / C1 (best-lambda soft, raw) /
             C2 (T* + posthoc Π, guarantee) / C3 (rayen arm through its ray-shoot,
             guarantee) + reference rows: causal (T* with the right flank masked
             too -- the information set of a causal closed-loop forecaster) and
             pro-rata (share_i x net_demand, zero-learning). MAE + violation
             counts; the best GUARANTEED row (C2 vs C3, blackout MAE, 5%% tie
             rule) is written to stage2_selection.json for D2/D3. Footnote: the
             feasibility floor F = MAE(map(truth)) -- what each guarantee
             mechanism costs a PERFECT prediction (Π: ~0; rayen: its single
             global step size is the known throttle).

    python3 imputation/itr_bench.py --stage1
    python3 imputation/itr_bench.py --stage2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap_data import load_flats, SIGN, ND_COL                                 # noqa: E402
import constraints as C                                                       # noqa: E402
import itr_data as D                                                          # noqa: E402
from itr_model import ITransformerImputer, gap_slice, apply_gap_map           # noqa: E402

OUT = HERE / "results" / "itr"
LENGTHS = [36, 72, 144, 288]
LNAME = {36: "3h", 72: "6h", 144: "12h", 288: "blackout"}
TIE_BAND = 0.05                                     # <5% relative MAE = tie, not a win
TOL_BAL, TOL_RAMP, TOL_NEG = 1.0, 0.6, -0.1


def test_sets(f, smoke=False):
    n = {36: 24, 72: 24, 144: 24, 288: 12} if smoke else {36: 256, 72: 256, 144: 256, 288: 160}
    return {L: D.sample_windows(f, "test", n[L], glen_fixed=L, seed=123 + L) for L in LENGTHS}


def load_model(stem, device):
    p = OUT / f"{stem}.pt"
    if not p.exists():
        return None
    m = ITransformerImputer().to(device)
    m.load_state_dict(torch.load(p, map_location=device, weights_only=True)); m.eval()
    return m


def fill_windows(model, f, rec, device, batch=128, causal=False):
    """Model fills for a fixed-glen record set -> (n,T,6) MW plus batch dicts."""
    fills, metas = [], []
    with torch.no_grad():
        for glen, idx in D.glen_groups(rec, batch):
            b = D.build_batch(f, "test", rec, idx)
            X, M = b["X"].copy(), b["M"].copy()
            if causal and glen < D.T:                    # hide the right flank too
                for k in range(len(idx)):
                    hide = np.arange(b["g0"][k], D.T)
                    M[k, hide, 0] = 0.0
                    X[k][np.ix_(hide, D._TFI)] = 0.0
            xt = torch.from_numpy(X).to(device)
            mt = torch.from_numpy(M).to(device)
            out = model(xt, mt).cpu().numpy().astype(np.float64)
            fills.append(out * f.y_scale + f.y_mean)     # (B,T,6) MW
            metas.append(b)
    return fills, metas


def score(fills, metas, f, map_name):
    """MAE over gap cells + violation counts through a constraint map."""
    tot = n = 0.0
    v = {"bal_steps": 0, "ramp_cells": 0, "neg_cells": 0, "soc_days": 0}
    for F_, b in zip(fills, metas):
        glen = b["glen"]
        mapped = apply_gap_map(map_name, F_, b["g0"], glen, b["pL"], b["pR"], b["nd"])
        G = gap_slice(mapped, b["g0"], glen)
        truth = gap_slice(b["Y"].astype(np.float64), b["g0"], glen) * f.y_scale + f.y_mean
        tot += np.abs(G - truth).sum(); n += G.size
        bal = np.abs((G * SIGN).sum(-1) - b["nd"])
        v["bal_steps"] += int((bal > TOL_BAL).sum())
        full = np.concatenate([b["pL"][:, None], G, b["pR"][:, None]], axis=1)
        d = np.diff(full, axis=1)
        v["ramp_cells"] += int(((d > C.R_UP + TOL_RAMP) | (d < -(C.R_DN + TOL_RAMP))).sum())
        v["neg_cells"] += int((G < TOL_NEG).sum())
        for k in range(len(G)):
            if C._soc_swing_mwh(G[k]) > C.BATT_CAP_MWH + 1e-6:
                v["soc_days"] += 1
    return tot / max(n, 1), v


def prorata_fills(f, rec):
    """share_i x net_demand(t): zero-learning, balance-exact by construction of the
    shares (Σ SIGN·share = 1 on the train flat, where nd == Σ SIGN·P exactly)."""
    share = f.y_to_mw(f.Ytr).mean(0) / f.col_mw(f.Xtr, ND_COL).mean()
    fills, metas = [], []
    for glen, idx in D.glen_groups(rec, 128):
        b = D.build_batch(f, "test", rec, idx)
        F_ = b["Y"].astype(np.float64) * f.y_scale + f.y_mean            # start from truth...
        nd_full = np.zeros((len(idx), D.T))
        pos = b["g0"][:, None] + np.arange(glen)[None, :]
        F_[np.arange(len(idx))[:, None], pos] = b["nd"][:, :, None] * share[None, None, :]
        fills.append(F_); metas.append(b)                                # ...gap = pro-rata
        del nd_full
    return fills, metas


def tie_note(vals, best_key):
    ties = [k for k, v in vals.items()
            if k != best_key and v is not None
            and abs(v - vals[best_key]) / max(vals[best_key], 1e-9) < TIE_BAND]
    return ties


def stage1(f, device, smoke):
    sets = test_sets(f, smoke)
    sfx = "_smoke" if smoke else ""
    rows, blk = {}, {}
    for arm in D.ARMS:
        meta_p = OUT / f"itr_{arm}{sfx}.json"
        model = load_model(f"itr_{arm}{sfx}", device)
        if model is None:
            rows[arm] = None; continue
        rows[arm] = {}
        for L in LENGTHS:
            fills, metas = fill_windows(model, f, sets[L], device)
            rows[arm][L], _ = score(fills, metas, f, "raw")
        blk[arm] = json.loads(meta_p.read_text())["val_blackout_mae_mw"] if meta_p.exists() else None
    lines = ["# D1 — training-method ablation (T1–T4, raw fills, MAE in MW)", "",
             "| arm | " + " | ".join(f"MAE {LNAME[L]}" for L in LENGTHS)
             + " | val blackout-MAE (T* metric) |",
             "| --- | " + " | ".join("---" for _ in LENGTHS) + " | --- |"]
    for arm in D.ARMS:
        if rows[arm] is None:
            lines.append(f"| {arm} | _not trained_ | | | | |"); continue
        tail = f" | {blk[arm]:.1f} |" if blk[arm] is not None else " | — |"
        lines.append(f"| {arm} | " + " | ".join(f"{rows[arm][L]:.1f}" for L in LENGTHS) + tail)
    have = {k: v for k, v in blk.items() if v is not None}
    sel = {}
    if have:
        tstar = min(have, key=have.get)
        ties = tie_note(have, tstar)
        sel = {"tstar": tstar, "val_blackout_mae_mw": have[tstar],
               "ties_within_5pct": ties, "smoke": smoke}
        lines += ["", f"**T\\* = {tstar}** (val blackout-MAE {have[tstar]:.1f} MW)."
                  + (f" ⚠ TIE within 5%: {ties} — treat as equivalent; rerun with seeds "
                     "only if the choice matters." if ties else "")]
        (OUT / f"stage1_selection{sfx}.json").write_text(json.dumps(sel, indent=2))
    (OUT / f"d1_training_table{sfx}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return sel


def stage2(f, device, smoke):
    sfx = "_smoke" if smoke else ""
    sel_p = OUT / f"stage1_selection{sfx}.json"
    if not sel_p.exists():
        raise SystemExit("run --stage1 first (needs stage1_selection.json)")
    tstar = json.loads(sel_p.read_text())["tstar"]
    sets = test_sets(f, smoke)
    m_star = load_model(f"itr_{tstar}{sfx}", device)
    # C1: best lambda on val gap-MAE among the trained soft runs
    lam_best, lam_val = None, float("inf")
    for lam in (0.1, 1.0, 10.0):
        p = OUT / f"itr_{tstar}_soft{lam:g}{sfx}.json"
        if p.exists():
            v = json.loads(p.read_text())["best_val_gap_mae_mw"]
            if v < lam_val:
                lam_best, lam_val = lam, v
    m_soft = load_model(f"itr_{tstar}_soft{lam_best:g}{sfx}", device) if lam_best else None
    m_ray = load_model(f"itr_{tstar}_rayen{sfx}", device)

    rows = []   # (name, guarantee, model, map, causal)
    rows.append((f"C0 none (= {tstar})", "no", m_star, "raw", False))
    rows.append((f"C1 soft λ={lam_best:g}" if lam_best else "C1 soft", "no", m_soft, "raw", False))
    rows.append(("C2 posthoc Π(C0)", "YES", m_star, "proj", False))
    rows.append(("C3 RAYEN fixed-D", "YES", m_ray, "rayen", False))
    rows.append(("ref: causal (left ctx only)", "no", m_star, "raw", True))
    rows.append(("ref: pro-rata share×nd", "no", None, "prorata", False))

    hdr = ("| method | guarantee | " + " | ".join(f"MAE {LNAME[L]}" for L in LENGTHS)
           + " | bal>1MW steps | ramp cells | neg cells | SOC days |")
    lines = [f"# D1 — constraint-method ablation (training = {tstar})", "", hdr,
             "| --- | --- | " + " | ".join("---" for _ in LENGTHS) + " | --- | --- | --- | --- |"]
    blk_mae = {}
    for name, guar, model, mp, causal in rows:
        if model is None and mp != "prorata":
            lines.append(f"| {name} | {guar} | _not trained_ | | | | | | |"); continue
        maes, vt = {}, {"bal_steps": 0, "ramp_cells": 0, "neg_cells": 0, "soc_days": 0}
        for L in LENGTHS:
            if mp == "prorata":
                fills, metas = prorata_fills(f, sets[L])
                mae, v = score(fills, metas, f, "raw")
            else:
                fills, metas = fill_windows(model, f, sets[L], device, causal=causal)
                mae, v = score(fills, metas, f, mp)
            maes[L] = mae
            for k in vt:
                vt[k] += v[k]
        if guar == "YES":
            blk_mae[name] = maes[288]
        note = " (≡ C0 for blackout: no flanks to mask)" if causal else ""
        lines.append(f"| {name}{note} | {guar} | "
                     + " | ".join(f"{maes[L]:.1f}" for L in LENGTHS)
                     + f" | {vt['bal_steps']} | {vt['ramp_cells']} | {vt['neg_cells']} "
                     f"| {vt['soc_days']} |")

    # feasibility floor F: what each guarantee map costs a PERFECT prediction
    Fnote = []
    rec = sets[288]
    b = D.build_batch(f, "test", rec, np.arange(len(rec["s"])))
    truth = b["Y"].astype(np.float64) * f.y_scale + f.y_mean
    for nm, mp in (("Π", "proj"), ("rayen", "rayen")):
        mapped = apply_gap_map(mp, truth.copy(), b["g0"], b["glen"], b["pL"], b["pR"], b["nd"])
        Fnote.append(f"F_{nm} = {np.abs(mapped - truth).mean():.2f} MW")
    lines += ["", f"_Feasibility floor (blackout, map applied to the TRUTH): "
              + "; ".join(Fnote) + ". Π costs a perfect prediction ~nothing; a large "
              "F_rayen explains a large C3 MAE as the mechanism's conservatism, "
              "not the network._"]

    sel = {}
    if blk_mae:
        bestg = min(blk_mae, key=blk_mae.get)
        ties = tie_note(blk_mae, bestg)
        map_name = "proj" if "C2" in bestg else "rayen"
        stem = f"itr_{tstar}{sfx}" if "C2" in bestg else f"itr_{tstar}_rayen{sfx}"
        sel = {"tstar": tstar, "best_guaranteed": bestg, "map": map_name,
               "ckpt_stem": stem, "blackout_mae_mw": blk_mae[bestg],
               "ties_within_5pct": ties, "smoke": smoke}
        (OUT / f"stage2_selection{sfx}.json").write_text(json.dumps(sel, indent=2))
        lines += ["", f"**Best guaranteed-feasible: {bestg}** (blackout MAE "
                  f"{blk_mae[bestg]:.1f} MW)."
                  + (f" ⚠ TIE within 5%: {ties} — if the C2-vs-C3 choice matters for the "
                     "claim, rerun that pair with 2–3 seeds." if ties else "")]
    (OUT / f"d1_constraint_table{sfx}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             ("mps" if torch.backends.mps.is_available() else "cpu"))
    f = load_flats()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage1 or not args.stage2:
        stage1(f, device, args.smoke)
    if args.stage2:
        stage2(f, device, args.smoke)


if __name__ == "__main__":
    main()
