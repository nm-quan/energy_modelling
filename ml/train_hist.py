"""Train a forecaster on the hist dataset (2021-2026) -- local or Colab.

Loads data from the compressed prepared.npz that ships in the repo
(pipeline.load_prepared -- no parquets needed), trains with the checkpointed
loop (per-epoch save/resume, so Colab disconnects or local kills lose at most
one epoch), and writes weights + metrics JSON to --out.

  python3 ml/train_hist.py --arch itransformer
  python3 ml/train_hist.py --arch lstm_revin --seed 1
  python3 ml/train_hist.py --arch itransformer --out /content/drive/MyDrive/energy_runs

Constraint arms (plan.md):
  --arch {lstm,itransformer}_rayen   approach 1: hard RAYEN layer trained through
  --arch {lstm,itransformer}_task7   approach 2: 7-output task net (D_t, P_1..P_6)
        trained as usual; after training the decision-rules safe blend
        (lib/decision_rule.py) is fitted + evaluated post-hoc and reported as a
        second row in the metrics JSON ("dr").

Arch-specific recipes match the last365 reference models (apples-to-apples):
lstm*: epochs 40, patience 6, batch 128; itransformer: 60, 10, 64; lr 1e-4.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent          # ml/
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline                 # noqa: E402
import models as M              # noqa: E402
import evaluate as ev           # noqa: E402

RECIPES = {"itransformer": dict(epochs= 40, patience=10, batch=128)}
DEFAULT_RECIPE = dict(epochs=40, patience=10, batch=128)


def _val_mse(model, Xva, Yva, device, batch):
    mse = torch.nn.MSELoss(reduction="sum"); tot, n = 0.0, len(Xva)
    with torch.no_grad():
        model.eval()
        for i in range(0, n, batch):
            xb = torch.from_numpy(np.ascontiguousarray(Xva[i:i + batch])).to(device)
            yb = torch.from_numpy(np.ascontiguousarray(Yva[i:i + batch])).to(device)
            tot += mse(model(xb), yb).item()
    return tot / (n * Yva.shape[1])


def train_ckpt(model, Xtr, Ytr, Xva, Yva, device, epochs, patience, batch, lr, ckpt):
    """AdamW + MSE with early stopping; checkpoints every epoch, auto-resumes."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = torch.nn.MSELoss(); n = len(Xtr)
    best_val, best_state, waited, start_ep = float("inf"), None, 0, 1
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device)
        model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["opt_state"])
        best_val, best_state, waited = ck["best_val"], ck["best_state"], ck["waited"]
        start_ep = ck["epoch"] + 1
        print(f"  resumed at epoch {start_ep} (best_val={best_val:.5f})", flush=True)

    ep = start_ep - 1
    for ep in range(start_ep, epochs + 1):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            idx = np.sort(perm[i:i + batch])       # sorted gather is faster on views
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(Ytr[idx]).to(device)
            opt.zero_grad(); loss = mse(model(xb), yb); loss.backward(); opt.step()
        vmse = _val_mse(model, Xva, Yva, device, batch)
        stop = False
        if vmse < best_val - 1e-6:
            best_val = vmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            stop = waited >= patience
        torch.save({"epoch": ep, "best_val": best_val, "best_state": best_state,
                    "waited": waited, "model_state": model.state_dict(),
                    "opt_state": opt.state_dict()}, ckpt)
        print(f"  epoch {ep:03d}  val_mse={vmse:.5f}  ({time.time()-t0:.0f}s)"
              f"{'  early stop' if stop else ''}", flush=True)
        if stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    help="any make_neural arch, e.g. lstm, lstm_revin, itransformer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="hist")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="output dir (default ml/<arch>_<dataset>/)")
    ap.add_argument("--stride", type=int, default=1,
                    help="train/val window stride for cheap pilots (e.g. 12)")
    args = ap.parse_args()

    is_rayen = args.arch.endswith("_rayen")
    is_task7 = args.arch.endswith("_task7")     # decision-rules task net (7 outputs)
    base_arch = args.arch.rsplit("_", 1)[0] if (is_rayen or is_task7) else args.arch
    rec = dict(RECIPES.get(base_arch, DEFAULT_RECIPE))
    for k in ("epochs", "patience", "batch"):
        if getattr(args, k) is not None:
            rec[k] = getattr(args, k)
    device = ev.pick_device(args.device)
    out = Path(args.out) if args.out else HERE / f"{args.arch}_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arch}_{args.dataset}_s{args.seed}" + (f"_str{args.stride}" if args.stride > 1 else "")
    print(f"{tag}: device={device} recipe={rec}", flush=True)

    data = pipeline.load_prepared("5min", 24, 1, "net_dispatch_totdem", args.dataset)
    xs, ys = data["x_scaler"], data["y_scaler"]
    nd_idx = data["feat_cols"].index("net_demand")
    Xtr, Ytr = data["Xtr"][::args.stride], data["Ytr"][::args.stride]
    Xva, Yva = data["Xva"][::args.stride], data["Yva"][::args.stride]
    print(f"windows: train {len(Xtr):,}  val {len(Xva):,}  test {len(data['Xte']):,}", flush=True)

    SIGN = np.array([1, 1, 1, 1, -1, 1], dtype=np.float64)
    if is_rayen or is_task7:
        # 7-dim target: [scaled nd_t, 6 scaled dispatch] -- the layer's D output
        # is supervised against the actual net demand (SIGN . dispatch identity)
        def y7(Y):
            nd_mw = ys.inverse_transform(Y) @ SIGN
            nd_s = (nd_mw - xs.mean_[nd_idx]) / xs.scale_[nd_idx]
            return np.concatenate([nd_s[:, None], Y], axis=1).astype(np.float32)
        Ytr, Yva = y7(Ytr), y7(Yva)

    from check_caps import RAMPS
    ramp_dn = [abs(RAMPS[t][0]) for t in data["targets"]]
    ramp_up = [RAMPS[t][1] for t in data["targets"]]
    ev._seed_all(args.seed)
    if is_rayen:
        model = M.make_rayen(base_arch, xs, ys, ramp_up, ramp_dn,
                             nd_feat_idx=nd_idx, n_features=len(data["feat_cols"])).to(device)
    elif is_task7:
        model = M.make_task7(base_arch, n_features=len(data["feat_cols"]),
                             nd_feat_idx=nd_idx).to(device)
    else:
        model = M.make_neural(args.arch, n_features=len(data["feat_cols"]), n_targets=6).to(device)
    t0 = time.time()
    model, best_val, ep = train_ckpt(model, Xtr, Ytr, Xva, Yva,
                                     device, rec["epochs"], rec["patience"], rec["batch"],
                                     args.lr, out / f"{tag}_ckpt.pt")
    torch.save(model.state_dict(), out / f"{tag}.pt")

    true = ys.inverse_transform(data["Yte"])
    nd_true = true @ SIGN
    prev = data["Xte"][:, -1, M.TARGET_FEAT_IDX] * xs.scale_[M.TARGET_FEAT_IDX] \
        + xs.mean_[M.TARGET_FEAT_IDX]

    def seven_extras(pred_raw, label):
        """(N,7) [scaled D, scaled P] -> (pred MW, constraint self-checks):
        balance vs own D, ramps vs prev actual (tol = eps-anchor slack +
        noise floor), floors, D forecast quality."""
        d_mw = pred_raw[:, 0] * xs.scale_[nd_idx] + xs.mean_[nd_idx]
        pred = ys.inverse_transform(pred_raw[:, 1:])
        resid = np.abs(pred @ SIGN - d_mw)
        delta = pred - prev
        tol = 0.6
        n_ramp = int(((delta > np.array(ramp_up) + tol)
                      | (delta < -(np.array(ramp_dn) + tol))).sum())
        nd_wape = float(np.abs(d_mw - nd_true).sum() / np.abs(nd_true).sum())
        extras = {"balance_resid_max_mw": float(resid.max()), "n_ramp_vs_prev": n_ramp,
                  "n_neg": int((pred < -0.1).sum()), "nd_WAPE": nd_wape}
        print(f"  {label}: |SIGN.P - D| max={resid.max():.4f} MW, "
              f"ramp_viol={n_ramp}, neg={extras['n_neg']}, nd_WAPE={nd_wape:.4f}", flush=True)
        return pred, extras

    pred_raw = ev.predict_neural(model, data["Xte"], device, batch=256)
    extras, dr_result = {}, None
    if is_rayen:
        pred, extras = seven_extras(pred_raw, "rayen self-check")
    elif is_task7:
        pred, extras = seven_extras(pred_raw, "task7 bare")
        # post-hoc decision-rules arm: safe-net LP once, wrap, re-predict
        import decision_rule as dr
        from check_caps import CAPS
        nd_col = data["Xtr"][:, -1, nd_idx] * xs.scale_[nd_idx] + xs.mean_[nd_idx]
        span = nd_col.max() - nd_col.min()
        fit = dr.fit_safe_F(ramp_up, ramp_dn,
                            [1.3 * CAPS[t] for t in data["targets"]],
                            float(nd_col.min() - 0.25 * span),
                            float(nd_col.max() + 0.25 * span))
        print(f"  safe LP: t*={fit['t']:.2f} MW  mc_min_slack={fit['mc_min_slack']:.2f} "
              f"mc_eq_max={fit['mc_eq_max']:.2e}", flush=True)
        np.savez(out / f"{tag}_safeF.npz", **{k: v for k, v in fit.items() if k != "include_floor"})
        wrapped = dr.DecisionRuleHead(model, fit["F"], xs.mean_, xs.scale_,
                                      ys.mean_, ys.scale_, ramp_up, ramp_dn,
                                      nd_feat_idx=nd_idx).to(device).eval()
        outs, alphas, sn_mins = [], [], []
        with torch.no_grad():
            for i in range(0, len(data["Xte"]), 256):
                xb = torch.from_numpy(np.ascontiguousarray(data["Xte"][i:i + 256])).to(device)
                outs.append(wrapped(xb).cpu().numpy())
                alphas.append(wrapped.last_alpha.cpu().numpy())
                sn_mins.append(wrapped.last_sn_min.cpu().numpy())
        dr_raw = np.concatenate(outs); alpha = np.concatenate(alphas)
        sn_min = np.concatenate(sn_mins)
        dr_pred, dr_extras = seven_extras(dr_raw, "task7 +DR")
        dr_extras.update(safe_t=fit["t"], n_outside_X=int((sn_min < -1e-6).sum()),
                         alpha_frac_active=float((alpha > 1e-6).mean()),
                         alpha_mean_active=float(alpha[alpha > 1e-6].mean()) if (alpha > 1e-6).any() else 0.0,
                         alpha_max=float(alpha.max()))
        dr_met = ev.compute_metrics(true, dr_pred, data["targets"])
        dr_result = {"metrics": dr_met, **dr_extras}
        print(f"  task7 +DR: WAPE={dr_met['average']['WAPE']:.4f} "
              f"alpha active {100 * dr_extras['alpha_frac_active']:.2f}% "
              f"(mean {dr_extras['alpha_mean_active']:.3f}, max {dr_extras['alpha_max']:.3f}), "
              f"outside-X {dr_extras['n_outside_X']}", flush=True)
    else:
        pred = ys.inverse_transform(pred_raw)
    met = ev.compute_metrics(true, pred, data["targets"])
    avg = met["average"]
    result = {"tag": tag, "arch": args.arch, "seed": args.seed, "dataset": args.dataset,
              "device": device, "recipe": rec, "lr": args.lr, "epochs_run": ep,
              "stride": args.stride, "best_val_mse": best_val,
              "secs": time.time() - t0, "metrics": met, **extras}
    if dr_result is not None:
        result["dr"] = dr_result
    (out / f"{tag}_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"{tag}: WAPE={avg['WAPE']:.4f} R2={avg['R2']:.4f} "
          f"({ep} epochs, {(time.time()-t0)/60:.0f} min)")
    print("wrote", out / f"{tag}_metrics.json")


if __name__ == "__main__":
    main()
