"""Train a forecaster on the hist dataset (2021-2026) -- local or Colab.

Loads data from the compressed prepared.npz that ships in the repo
(pipeline.load_prepared -- no parquets needed), trains with the checkpointed
loop (per-epoch save/resume, so Colab disconnects or local kills lose at most
one epoch), and writes weights + metrics JSON to --out.

  python3 ml/train_hist.py --arch itransformer
  python3 ml/train_hist.py --arch lstm_revin --seed 1
  python3 ml/train_hist.py --arch itransformer --out /content/drive/MyDrive/energy_runs

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

RECIPES = {"itransformer": dict(epochs=60, patience=10, batch=64)}
DEFAULT_RECIPE = dict(epochs=40, patience=6, batch=128)


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
    args = ap.parse_args()

    rec = dict(RECIPES.get(args.arch, DEFAULT_RECIPE))
    for k in ("epochs", "patience", "batch"):
        if getattr(args, k) is not None:
            rec[k] = getattr(args, k)
    device = ev.pick_device(args.device)
    out = Path(args.out) if args.out else HERE / f"{args.arch}_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arch}_{args.dataset}_s{args.seed}"
    print(f"{tag}: device={device} recipe={rec}", flush=True)

    data = pipeline.load_prepared("5min", 24, 1, "net_dispatch_totdem", args.dataset)
    print(f"windows: train {len(data['Xtr']):,}  val {len(data['Xva']):,}  "
          f"test {len(data['Xte']):,}", flush=True)

    ev._seed_all(args.seed)
    model = M.make_neural(args.arch, n_features=len(data["feat_cols"]), n_targets=6).to(device)
    t0 = time.time()
    model, best_val, ep = train_ckpt(model, data["Xtr"], data["Ytr"], data["Xva"], data["Yva"],
                                     device, rec["epochs"], rec["patience"], rec["batch"],
                                     args.lr, out / f"{tag}_ckpt.pt")
    torch.save(model.state_dict(), out / f"{tag}.pt")

    ys = data["y_scaler"]
    pred = ys.inverse_transform(ev.predict_neural(model, data["Xte"], device, batch=256))
    true = ys.inverse_transform(data["Yte"])
    met = ev.compute_metrics(true, pred, data["targets"])
    avg = met["average"]
    result = {"tag": tag, "arch": args.arch, "seed": args.seed, "dataset": args.dataset,
              "device": device, "recipe": rec, "lr": args.lr, "epochs_run": ep,
              "best_val_mse": best_val, "secs": time.time() - t0, "metrics": met}
    (out / f"{tag}_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"{tag}: WAPE={avg['WAPE']:.4f} R2={avg['R2']:.4f} "
          f"({ep} epochs, {(time.time()-t0)/60:.0f} min)")
    print("wrote", out / f"{tag}_metrics.json")


if __name__ == "__main__":
    main()
