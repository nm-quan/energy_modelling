"""Plan1 benchmark (plan1.md S4): one harness, same 400 test 3h gaps for every arm.

Rows: interp+map reference, baseline, cost, size_aware (+ optimization later).
Metrics: MAE (MW), MRE (%) = 100*Sigma|err|/Sigma|truth| -- per channel + aggregate;
dispatch cost $ (c_signed . E, battery charging negative) for prediction and truth;
feasibility counters vs the SUPPLY-side balance target (must be 0).

    python3 imputation/plan1_bench.py            # full weights
    python3 imputation/plan1_bench.py --smoke    # *_smoke.pt sanity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "prepared.npz"
from gap_data import (load_flats, sample_recon_windows, TARGETS, SIGN,  # noqa: E402
                      TARGET_FEAT_IDX)
import constraints as C                                                # noqa: E402
from model import BiLSTMImputer                                        # noqa: E402
from constraint_layers import rayen_traj_project                       # noqa: E402
from plan1_train import C_SIGNED                                       # noqa: E402

WEIGHTS = HERE / "results" / "plan1"          # checkpoints load from here
OUT = HERE / "results" / "plan1"              # bench written here (--out overrides)
ARMS = ["baseline", "cost", "size_aware"]
DT = 5.0 / 60.0


def fill_window(model, f, gw, device, context=48):
    """Raw fill (G,6) MW for one GapWindow: interp skeleton + model deviation."""
    tfi = np.asarray(TARGET_FEAT_IDX)
    g0, N = gw.gap_idx[0], len(gw.gap_idx)
    xs = f.Xte[g0 - context: g0 + N + context].copy()
    m = np.ones((xs.shape[0], 1), np.float32); m[context:context + N] = 0.0
    xs[context:context + N][:, tfi] = 0.0
    with torch.no_grad():
        dev = model(torch.from_numpy(xs[None].astype(np.float32)).to(device),
                    torch.from_numpy(m[None]).to(device))[0, context:context + N].cpu().numpy()
    tt = (np.arange(1, N + 1) / (N + 1))[:, None]
    pL_s = (gw.pL_mw - f.y_mean) / f.y_scale
    pR_s = (gw.pR_mw - f.y_mean) / f.y_scale
    interp = pL_s[None, :] + tt * (pR_s - pL_s)[None, :]
    return (interp + dev) * f.y_scale + f.y_mean


def score(name, fills, gws, size_aware):
    num = np.zeros(6); den = np.zeros(6); abs_err = np.zeros(6); n_cells = 0
    cost_p = cost_t = 0.0
    viol = {"bal>1MW": 0, "ramp": 0, "neg": 0, "SOC": 0}
    for fill, gw in zip(fills, gws):
        nd_bal = gw.truth_mw @ SIGN                       # supply-side target
        with torch.no_grad():
            # size-aware: the proportional POCS anchor converges slower than the
            # equal split -- 160 cycles drives the residual to ~0 (40 leaves a few
            # >1 MW cells). Training keeps 40 (soft-balance covers the slack).
            P = rayen_traj_project(torch.tensor(fill[None]),
                                   torch.tensor(gw.pL_mw[None]),
                                   torch.tensor(gw.pR_mw[None]),
                                   torch.tensor(nd_bal[None]),
                                   anchor_iters=160 if size_aware else 40,
                                   size_aware=size_aware)[0].numpy()
        e = np.abs(P - gw.truth_mw)
        num += e.sum(0); den += np.abs(gw.truth_mw).sum(0)
        abs_err += e.sum(0); n_cells += len(P)
        cost_p += float((P * C_SIGNED).sum()) * DT
        cost_t += float((gw.truth_mw * C_SIGNED).sum()) * DT
        viol["bal>1MW"] += int((np.abs((P * SIGN).sum(-1) - nd_bal) > 1.0).sum())
        full = np.vstack([gw.pL_mw[None], P, gw.pR_mw[None]])
        d = np.diff(full, axis=0)
        viol["ramp"] += int(((d > C.R_UP + 0.6) | (d < -(C.R_DN + 0.6))).sum())
        viol["neg"] += int((P < -0.1).sum())
        viol["SOC"] += int(C._soc_swing_mwh(P) > C.BATT_CAP_MWH + 1e-6)
    mae = abs_err / n_cells
    mre = 100.0 * num / np.clip(den, 1e-9, None)
    return {"name": name,
            "mae": {t: float(mae[i]) for i, t in enumerate(TARGETS)},
            "mae_agg": float(abs_err.sum() / (n_cells * 6)),
            "mre": {t: float(mre[i]) for i, t in enumerate(TARGETS)},
            "mre_agg": float(100.0 * num.sum() / den.sum()),
            "cost_pred_usd": cost_p, "cost_truth_usd": cost_t,
            "cost_delta_usd": cost_p - cost_t, "violations": viol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="output dir for bench.md/json (default results/plan1)")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out)
    sfx = "_smoke" if args.smoke else ""
    f = load_flats()
    n_feat = len(f.feat_cols)
    gws = sample_recon_windows(f, "test", n=args.n, context=48, gap=36, seed=123)
    print(f"bench: {len(gws)} test 3h gaps, {n_feat} features")

    rows = []
    # reference: interp through the equal map (no model)
    tt = None
    interp_fills = []
    for gw in gws:
        N = len(gw.gap_idx)
        t_ = (np.arange(1, N + 1) / (N + 1))[:, None]
        interp_fills.append(gw.pL_mw[None, :] + t_ * (gw.pR_mw - gw.pL_mw)[None, :])
    rows.append(score("interp+map (ref)", interp_fills, gws, size_aware=False))

    for arm in ARMS:
        p = WEIGHTS / f"{arm}{sfx}.pt"
        if not p.exists():
            print(f"  [skip] {p} missing"); continue
        model = BiLSTMImputer(n_features=n_feat).to(args.device)
        model.load_state_dict(torch.load(p, map_location=args.device, weights_only=True))
        model.eval()
        fills = [fill_window(model, f, gw, args.device) for gw in gws]
        rows.append(score(arm, fills, gws, size_aware=(arm == "size_aware")))
        print(f"  scored {arm}")

    lines = [f"# plan1 benchmark — {len(gws)} test 3h gaps (seed 123), map balance = supply-side nd", "",
             "| model | MAE agg (MW) | MRE agg (%) | cost pred ($) | Δcost vs truth ($) | bal>1MW | ramp | neg | SOC |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        v = r["violations"]
        lines.append(f"| {r['name']} | {r['mae_agg']:.1f} | {r['mre_agg']:.2f} | "
                     f"{r['cost_pred_usd']:,.0f} | {r['cost_delta_usd']:+,.0f} | "
                     f"{v['bal>1MW']} | {v['ramp']} | {v['neg']} | {v['SOC']} |")
    lines += ["", "## per-channel MRE (%)", "",
              "| model | " + " | ".join(TARGETS) + " |",
              "| --- |" + " --- |" * 6]
    for r in rows:
        lines.append(f"| {r['name']} | " + " | ".join(f"{r['mre'][t]:.1f}" for t in TARGETS) + " |")
    lines += ["", "## per-channel MAE (MW)", "",
              "| model | " + " | ".join(TARGETS) + " |",
              "| --- |" + " --- |" * 6]
    for r in rows:
        lines.append(f"| {r['name']} | " + " | ".join(f"{r['mae'][t]:.1f}" for t in TARGETS) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"bench{sfx}.md").write_text("\n".join(lines) + "\n")
    (OUT / f"bench{sfx}.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))
    print(f"\nwrote {OUT}/bench{sfx}.md")


if __name__ == "__main__":
    main()
