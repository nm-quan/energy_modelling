"""Train the bi-LSTM gap imputer on 5 years of masked windows, evaluate vs the
interpolation baseline on the real test 11:00-14:00 windows.

Training: random-position masked gaps over the 394k-row train flat.
Loss = masked reconstruction + soft balance (+ optional L2 on the deviation).
NOTE --perturb stays OFF by default: perturbing demand while keeping the real
breakdown as the target teaches the model to IGNORE demand (measured); demand
responsiveness comes from natural 5-yr variation + the balance term/projection.

Early stopping: on MIDDAY-MATCHED val windows (gaps pinned to 11:00-14:00 in the
held-out val split -- same task as test). The test set is scored exactly ONCE,
after training, on the val-selected best model. Leakage history in
gap_data.sample_val_midday_windows.

Eval (deployment-matched): mask the real 11-14 window, impute, PROJECT to hard
feasibility (constraints.project_gap), score reconstruction WAPE vs measured truth,
and check violations incl. both seams.

    python3 imputation/train.py --epochs 100 --patience 10   # full run (GPU)
    python3 imputation/train.py --smoke                      # tiny, CPU, sanity
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap_data import (load_flats, sample_train_windows, sample_recon_windows,   # noqa: E402
                      TARGETS, SIGN, TARGET_FEAT_IDX, ND_COL, DEM_COL)
import constraints as C                                                        # noqa: E402
from model import BiLSTMImputer, masked_loss                                   # noqa: E402
from constraint_layers import project_in_graph, deployed_fill                  # noqa: E402

OUT = HERE / "results"


def perturb_demand(X, mask, f, frac: float, rng, max_g: float = 30.0):
    """On a `frac` share of windows, scale demand+net_demand INSIDE the gap by a
    random +g% (like the deployment counterfactual). Known-subspace cols only, so
    the target (real breakdown) is unchanged -- teaching the model that a raised
    total should change the fill, not be ignored. Returns X (copy modified)."""
    X = X.copy()
    B, W, _ = X.shape
    gmask = (mask[:, :, 0] == 0)                              # (B,W) gap steps
    pick = rng.random(B) < frac
    g = rng.uniform(0, max_g, B) / 100.0
    for col in (ND_COL, DEM_COL):
        mw = X[:, :, col] * f.x_scale[col] + f.x_mean[col]
        mw = np.where(pick[:, None] & gmask, mw * (1 + g[:, None]), mw)
        X[:, :, col] = (mw - f.x_mean[col]) / f.x_scale[col]
    return X


# hour-of-day buckets for the general eval; "midday(11-14)" is the deployment slice
HOUR_BUCKETS = [("night(0-6)", 0, 6), ("morning(6-11)", 6, 11), ("midday(11-14)", 11, 14),
                ("afternoon(14-18)", 14, 18), ("evening(18-24)", 18, 24)]


def evaluate(model, f, gws, device, context, mode="posthoc"):
    """Reconstruction WAPE on GENERAL test gaps (all hours) + a per-hour breakdown
    incl. the midday(11-14) deployment slice + violation magnitudes. Scores
    Π(F(x)): the mode's own deployed forward (rayen shot for rayen_traj, identity
    otherwise -- see constraint_layers docstring) followed by the shared exact
    projection Π, so every mode is evaluated through the map it was trained with."""
    tfi = np.asarray(TARGET_FEAT_IDX)
    num = np.zeros(6); den = np.zeros(6)
    bnum = {b[0]: np.zeros(6) for b in HOUR_BUCKETS}
    bden = {b[0]: np.zeros(6) for b in HOUR_BUCKETS}
    ramp_over = 0.0; neg = 0; bal_max = 0.0
    model.eval()
    with torch.no_grad():
        for gw in gws:
            g0 = gw.gap_idx[0]; N = len(gw.gap_idx)
            xs = f.Xte[g0 - context: g0 + N + context].copy()   # (W,17)
            m = np.ones((xs.shape[0], 1), np.float32); m[context:context + N] = 0.0
            xs[context:context + N][:, tfi] = 0.0
            xb = torch.from_numpy(xs[None].astype(np.float32)).to(device)
            mb = torch.from_numpy(m[None]).to(device)
            dev = model(xb, mb)[0, context:context + N].cpu().numpy()   # (N,6) y-scaled deviation
            # residual: fill = linear-interp skeleton + learned deviation (y-scaled)
            tt = (np.arange(1, N + 1) / (N + 1))[:, None]
            pL_s, pR_s = f.Yte[g0 - 1], f.Yte[gw.gap_idx[-1] + 1]
            interp = pL_s[None, :] + tt * (pR_s - pL_s)[None, :]
            fill = (interp + dev) * f.y_scale + f.y_mean
            fill = deployed_fill(fill, gw.pL_mw, gw.pR_mw, gw.nd_mw, mode)   # F(x)
            P, resid = C.project_gap(fill, gw.pL_mw, gw.pR_mw, gw.nd_mw)     # Π(F(x))
            e = np.abs(P - gw.truth_mw).sum(0); t = np.abs(gw.truth_mw).sum(0)
            num += e; den += t
            for name, lo, hi in HOUR_BUCKETS:
                if lo <= gw.hour < hi:
                    bnum[name] += e; bden[name] += t
            ramp_over = max(ramp_over, C._ramp_overshoot_mw(np.vstack([gw.pL_mw, P, gw.pR_mw])))
            neg += int((P < -0.1).sum()); bal_max = max(bal_max, float(resid.max()))
    per = num / np.clip(den, 1e-6, None)
    # per-hour uses MICRO WAPE (Σ|err| / Σ|truth| across ALL channels) not the
    # per-channel macro: in some hour bands a small channel (gas_steam) is ~entirely
    # off, so its per-channel denominator ~0 and macro-WAPE explodes. Total dispatch
    # is always large, so micro is stable and still says "how close is this hour band".
    per_hour = {name: float(bnum[name].sum() / max(bden[name].sum(), 1e-6))
                for name, _, _ in HOUR_BUCKETS if bden[name].sum() > 1e-6}
    return {"macro_WAPE": float(per.mean()),          # mean of per-channel WAPE (repo convention)
            "micro_WAPE": float(num.sum() / max(den.sum(), 1e-6)),   # Σerr/Σtruth, stable headline
            "per_channel_WAPE": {t: float(per[i]) for i, t in enumerate(TARGETS)},
            "per_hour_WAPE": per_hour,
            "ramp_overshoot_mw": ramp_over, "n_neg": neg, "balance_resid_max_mw": bal_max,
            "n_eval_windows": len(gws)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10,
                    help="early stop after this many epochs with no val-recon improvement")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-eval", type=int, default=800,
                    help="general (all-hours) test windows for the final reconstruction eval")
    ap.add_argument("--n-val", type=int, default=4000,
                    help="general (all-hours) val windows for early stopping")
    ap.add_argument("--context", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="share of windows with SMALL input-noise augmentation (default off; "
                         "counterfactual responsiveness comes from natural 5-yr demand variation, "
                         "NOT from perturbing demand with an unchanged target -- that teaches the "
                         "model to IGNORE demand)")
    ap.add_argument("--constraint-mode", choices=["posthoc", "unrolled", "rayen_traj"],
                    default="posthoc",
                    help="how constraints are enforced. posthoc: project only at eval "
                         "(train on soft-balance loss). unrolled: project IN-GRAPH each "
                         "training step via unrolled cyclic POCS (model trains inside the "
                         "constraints). rayen_traj: differentiable RAYEN ray-shoot over the "
                         "whole gap. All three are scored by the SAME posthoc eval projection.")
    ap.add_argument("--proj-iters", type=int, default=10,
                    help="unrolled-mode: POCS rounds inside the forward pass (small = cheaper "
                         "gradients; eval always uses the full 40-round projection)")
    ap.add_argument("--loss", choices=["mse", "mae", "wape"], default="mse",
                    help="reconstruction loss. mse (default, but z-scored so it under-weights "
                         "the volatile battery channel); mae; wape (MW per-channel Σ|err|/Σ|truth|, "
                         "matches the reported metric so batteries aren't down-weighted).")
    ap.add_argument("--lam-bal", type=float, default=0.1)
    ap.add_argument("--lam-dev", type=float, default=0.0,
                    help="L2 penalty on the deviation from interp (keeps smooth channels "
                         "at the interp baseline; sweep on GPU, e.g. 0.01-0.3)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT / "bilstm_imputer.pt"))
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_train, args.batch = 2, 2000, 64
        args.n_eval, args.n_val = 200, 500
        # never let a smoke run clobber ANY real checkpoint/json: whatever --out was
        # given, write to its _smoke sibling (bilstm_posthoc.pt -> bilstm_posthoc_smoke.pt),
        # so benchmark.py can never mistake a 2-epoch smoke for a trained mode.
        p = Path(args.out)
        if not p.stem.endswith("_smoke"):
            args.out = str(p.with_name(("bilstm_smoke" if p.stem == "bilstm_imputer"
                                        else p.stem + "_smoke") + p.suffix))
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"device={device} epochs={args.epochs} n_train={args.n_train} "
          f"constraint_mode={args.constraint_mode} perturb={args.perturb}")

    f = load_flats()
    # startup guard: surface the flat shapes so a stale/rebuilt prepared.npz is
    # obvious on line 1 (the Colab run that silently gave 3 val windows had a
    # different val flat). sample_recon_windows below also fails loud if <30.
    print(f"flats: Xtr={f.Xtr.shape} Xva={f.Xva.shape} Xte={f.Xte.shape} lb_carry={f.lb_carry}")
    if f.Xva.shape[0] < 5000:
        print(f"  WARNING: Xva_flat has only {f.Xva.shape[0]} rows (committed data ~53,280) "
              f"-- val may be unrepresentative; check prepared.npz.")
    # GENERAL test windows (all hours) for the final reconstruction eval + per-hour
    # breakdown incl. the midday(11-14) deployment slice (user concern #1: eval must
    # be general, not midday-only). Fixed seed -> reproducible test set.
    gws = sample_recon_windows(f, "test", n=args.n_eval, context=args.context, seed=123)
    hh = np.array([w.hour for w in gws])
    print(f"eval: {len(gws)} GENERAL test windows, gap-open hours {hh.min():.1f}-{hh.max():.1f} "
          f"({int(((hh >= 11) & (hh < 14)).sum())} in the midday slice)")
    rng = np.random.default_rng(args.seed)
    ys_mean = torch.tensor(f.y_mean, dtype=torch.float32, device=device)
    ys_scale = torch.tensor(f.y_scale, dtype=torch.float32, device=device)
    sign = torch.tensor(SIGN, dtype=torch.float32, device=device)
    nd_mean, nd_scale = float(f.x_mean[ND_COL]), float(f.x_scale[ND_COL])

    model = BiLSTMImputer(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    gs, ge = args.context, args.context + 36

    # Train set sampled ONCE (reused each epoch -- standard, avoids the per-epoch
    # resample). Early stopping runs on GENERAL (all-hours) windows from the held-out
    # VAL split -- now MATCHED to the general test task (both all-hours), so val
    # estimates the deployed quantity and can't mis-rank epochs. History: v1 selected
    # epochs on TEST = leakage; v2 random val vs midday test = a difficulty mismatch;
    # v3 both general = matched. Test is scored ONCE, after training.
    tr = sample_train_windows(f, args.n_train, context=args.context, seed=args.seed, split="train")
    if args.perturb > 0:
        tr["X"] = perturb_demand(tr["X"], tr["mask"], f, args.perturb, rng)
    va = sample_train_windows(f, args.n_val, context=args.context, seed=args.seed + 777, split="val")
    print(f"val: {len(va['X'])} GENERAL held-out val windows for early stopping")

    def val_recon_wape():
        """Deployment-matched val: score Π(F(x)) exactly like the final test eval --
        F = the mode's forward (rayen shot for rayen_traj, identity otherwise), Π =
        the shared cyclic projection. Ranking epochs on the same map that deployment
        uses is what makes early stopping trustworthy for the in-graph modes (their
        raw fill is trained to rely on being projected; scoring it raw could pick
        the wrong epoch). Batched float32, fewer Π iters than test -- consistency of
        the operator across epochs is what matters for ranking, not the last 1e-6."""
        model.eval()
        with torch.no_grad():
            num = np.zeros(6); den = np.zeros(6)
            for i in range(0, len(va["X"]), 512):
                sl = slice(i, i + 512)
                dev = model(torch.from_numpy(va["X"][sl]).to(device),
                            torch.from_numpy(va["mask"][sl]).to(device))[:, gs:ge]
                fill = (torch.from_numpy(va["interp"][sl]).to(device) + dev) * ys_scale + ys_mean
                pLb = torch.from_numpy(va["pL_mw"][sl]).to(device)
                pRb = torch.from_numpy(va["pR_mw"][sl]).to(device)
                ndb = torch.from_numpy(va["nd_mw"][sl]).to(device)
                if args.constraint_mode == "rayen_traj":                 # F(x)
                    fill = project_in_graph(fill, pLb, pRb, ndb, mode="rayen_traj")
                P = C.cyclic_project(fill, pLb, pRb, ndb, iters=15)      # Π(F(x))
                truth = torch.from_numpy(va["Y"][sl]).to(device) * ys_scale + ys_mean
                num += (P - truth).abs().sum((0, 1)).cpu().numpy()
                den += truth.abs().sum((0, 1)).cpu().numpy()
        return float(np.mean(num / np.clip(den, 1e-6, None)))

    best = float("inf"); best_state = None; waited = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train(); perm = rng.permutation(len(tr["X"])); tot = 0.0
        for i in range(0, len(tr["X"]), args.batch):
            j = perm[i:i + args.batch]
            xb = torch.from_numpy(tr["X"][j]).to(device)
            mb = torch.from_numpy(tr["mask"][j]).to(device)
            yb = torch.from_numpy(tr["Y"][j]).to(device)
            interp = torch.from_numpy(tr["interp"][j]).to(device)
            xnd = torch.from_numpy(tr["X"][j][:, gs:ge, ND_COL]).to(device)
            dev = model(xb, mb)[:, gs:ge]
            out = interp + dev                                # residual: interp + learned deviation
            if args.constraint_mode != "posthoc":
                # project IN-GRAPH so the model trains inside the constraints. Work in MW,
                # then map back to y-scaled for the loss (balance term ~0 after projection).
                fill_mw = out * ys_scale + ys_mean
                P_mw = project_in_graph(fill_mw,
                                        torch.from_numpy(tr["pL_mw"][j]).to(device),
                                        torch.from_numpy(tr["pR_mw"][j]).to(device),
                                        torch.from_numpy(tr["nd_mw"][j]).to(device),
                                        mode=args.constraint_mode, iters=args.proj_iters)
                out = (P_mw - ys_mean) / ys_scale
            loss, rec, bal = masked_loss(out, yb, xnd, nd_mean, nd_scale,
                                         ys_mean, ys_scale, sign, args.lam_bal, loss=args.loss)
            loss = loss + args.lam_dev * (dev ** 2).mean()    # keep dev small: stay near interp
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(j)
        vw = val_recon_wape()
        stop = False
        if vw < best - 1e-5:
            best = vw; waited = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1; stop = waited >= args.patience
        print(f"  ep{ep:03d} loss={tot/len(tr['X']):.4f} val_recon_WAPE={vw:.4f} "
              f"(best {best:.4f}, waited {waited}/{args.patience}) ({time.time()-t0:.0f}s)"
              f"{' *' if waited == 0 else ''}{'  EARLY STOP' if stop else ''}", flush=True)
        if stop:
            break

    if best_state:
        model.load_state_dict(best_state)
    ev = evaluate(model, f, gws, device, args.context, mode=args.constraint_mode)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), args.out)
    ev.update(method="bilstm", constraint_mode=args.constraint_mode, loss=args.loss,
              epochs=args.epochs, n_train=args.n_train, perturb=args.perturb,
              context=args.context, n_eval=args.n_eval, eval_seed=123)
    # results json follows the checkpoint name, so --smoke (-> bilstm_smoke.pt)
    # writes bilstm_smoke_recon.json and never clobbers the real bilstm_recon.json
    recon_name = "bilstm_recon.json" if Path(args.out).stem == "bilstm_imputer" \
        else Path(args.out).stem + "_recon.json"
    (OUT / recon_name).write_text(json.dumps(ev, indent=2))
    print(f"\nBEST general recon: macro_WAPE={ev['macro_WAPE']:.4f}  micro_WAPE={ev['micro_WAPE']:.4f}  "
          f"ramp_overshoot={ev['ramp_overshoot_mw']:.2e} MW  neg={ev['n_neg']}  "
          f"bal_max={ev['balance_resid_max_mw']:.2e} MW  (n={ev['n_eval_windows']})")
    for t in TARGETS:
        print(f"    {t:20s} {ev['per_channel_WAPE'][t]:.4f}")
    print("  per-hour WAPE:", {k: round(v, 3) for k, v in ev["per_hour_WAPE"].items()})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
