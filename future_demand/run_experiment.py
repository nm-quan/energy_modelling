"""Does giving the LSTM AEMO's own future-demand forecast improve WAPE?

Very simple A/B: train the no-RevIN LSTM (lib.models.make_neural("lstm", ...))
twice on the identical 45-day window (future_demand_45d dataset -- see
lib/pipeline.py DATASETS and script/pull_p5min_demand_fcst.py for how the
AEMO forecast was obtained and joined in), same architecture/seed/recipe,
differing only in one input channel:

  baseline  input_mode=net_dispatch_totdem       (no forward-looking info)
  +futuredem input_mode=net_dispatch_totdem_fcst  (+ AEMO P5MIN 1-step demand
                                                    forecast, known at time t,
                                                    for the exact t+1 target
                                                    the LSTM predicts)

Both datasets must already be prepared (run this first if data/preprocessed/
future_demand_45d/5min/<mode>/ is missing):
    python3 -c "import sys; sys.path.insert(0,'lib'); import pipeline as p; \
      p.prepare('5min', 24, 1, 'net_dispatch_totdem', dataset='future_demand_45d'); \
      p.prepare('5min', 24, 1, 'net_dispatch_totdem_fcst', dataset='future_demand_45d')"

Stability notes (all needed to get a trustworthy comparison on this small,
45-day dataset -- none of these matter on the 4.75-year "hist" runs, which
dilute the same issues across thousands of batches/epoch instead of ~68):
  - price_aud_per_mwh has a genuine ~90-sigma outlier (a real AEMO price-cap
    event, not a data bug -- the full hist dataset has ~65-sigma outliers of
    the same kind). Unclipped, this exploded gradients (train_mse=inf most
    epochs). Backward grad-norm clipping alone wasn't enough (the forward
    pass through an under-trained LSTM already overflows on a raw 90-sigma
    input) -- fixed by winsorizing the *scaled* inputs to +/-8 sigma
    (INPUT_CLIP_SIGMA) in addition to grad clipping.
  - Apple's MPS backend produced NaN losses on this dataset's batch scale
    even with both of the above fixes; CPU training was stable. --device
    defaults to cpu here for that reason -- don't switch back to mps/auto
    without re-checking for NaN losses in the log.

Usage:
  python3 future_demand/run_experiment.py
  python3 future_demand/run_experiment.py --epochs 40 --patience 10 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import pipeline               # noqa: E402
import models as M            # noqa: E402
import evaluate as ev         # noqa: E402

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATASET = "future_demand_45d"
ARCH = "lstm"                 # no-RevIN LSTM per README's stated default model
GRAD_CLIP_NORM = 1.0
# price_aud_per_mwh has ~90-sigma price-cap-event outliers (2 of 12,959 rows,
# see run log) -- backward clipping alone isn't enough because the forward
# pass through an under-trained LSTM already overflows on a raw 90-sigma
# input. Winsorizing the *scaled* inputs is the standard fix for this exact
# electricity-price-cap situation; applied locally here only (not touched in
# lib/pipeline.py) since the 4.75-year "hist" runs never hit this at their
# batch scale and shouldn't change.
INPUT_CLIP_SIGMA = 8.0


def train_clipped(model: nn.Module, data: dict, spec: M.TrainSpec, device: str, name: str):
    """Same recipe as lib.evaluate.train_neural (AdamW + MSE, early stop on
    val MSE), plus gradient-norm clipping. Kept local to this experiment
    rather than touching lib/evaluate.py (shared by other, already-recorded
    runs) -- only this 45-day dataset is small enough (~68 batches/epoch) for
    a couple of ~90-sigma price-spike rows to blow up a whole epoch's mean
    loss and destabilize training; the 4.75-year "hist" runs dilute the same
    outliers across thousands of batches per epoch and never need clipping.
    """
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    loss_fn = nn.MSELoss()
    tr = DataLoader(TensorDataset(torch.from_numpy(data["Xtr"]), torch.from_numpy(data["Ytr"])),
                    batch_size=spec.batch, shuffle=True, pin_memory=True)
    va = DataLoader(TensorDataset(torch.from_numpy(data["Xva"]), torch.from_numpy(data["Yva"])),
                    batch_size=spec.batch * 4, shuffle=False, pin_memory=True)

    history = {"train_mse": [], "val_mse": []}
    best_val, best_state, waited, epochs_run = float("inf"), None, 0, 0
    for epoch in range(1, spec.epochs + 1):
        model.train()
        tl = []
        for xb, yb in tr:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            tl.append(loss.item())
        tloss = float(np.mean(tl))
        model.eval()
        with torch.no_grad():
            vl = []
            for xb, yb in va:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                vl.append(loss_fn(model(xb), yb).item())
            vloss = float(np.mean(vl))
        history["train_mse"].append(tloss)
        history["val_mse"].append(vloss)
        epochs_run = epoch
        print(f"  [{name}] epoch {epoch:03d}  train_mse={tloss:.5f}  val_mse={vloss:.5f}")
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= spec.patience:
                print(f"  [{name}] early stop at epoch {epoch} (best val_mse={best_val:.5f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_mse"] = best_val
    history["epochs_run"] = epochs_run
    return model, history


def run_arm(input_mode: str, name: str, spec: M.TrainSpec, seed: int, device: str) -> dict:
    data = pipeline.load_prepared("5min", 24, 1, input_mode, DATASET)
    for k in ("Xtr", "Xva", "Xte"):
        data[k] = np.clip(data[k], -INPUT_CLIP_SIGMA, INPUT_CLIP_SIGMA)
    print(f"[{name}] input_mode={input_mode}  features={data['feat_cols']}")
    print(f"[{name}] windows: train {len(data['Xtr']):,}  val {len(data['Xva']):,}  "
          f"test {len(data['Xte']):,}", flush=True)

    ev._seed_all(seed)
    model = M.make_neural(ARCH, n_features=len(data["feat_cols"]), n_targets=6)
    t0 = time.time()
    model, history = train_clipped(model, data, spec, device, name)
    secs = time.time() - t0

    pred_scaled = ev.predict_neural(model, data["Xte"], device, batch=256)
    ys = data["y_scaler"]
    pred = ys.inverse_transform(pred_scaled)
    true = ys.inverse_transform(data["Yte"])
    metrics = ev.compute_metrics(true, pred, data["targets"])

    print(f"[{name}] WAPE(avg)={metrics['average']['WAPE']:.4f}  "
          f"R2(avg)={metrics['average']['R2']:.4f}  "
          f"({history['epochs_run']} epochs, {secs:.0f}s)")
    return {
        "input_mode": input_mode, "n_features": len(data["feat_cols"]),
        "feat_cols": data["feat_cols"], "seed": seed,
        "epochs_run": history["epochs_run"], "best_val_mse": history["best_val_mse"],
        "secs": secs, "metrics": metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu",
                    help="default cpu: MPS produced NaN losses on this small "
                         "(~68 batches/epoch) dataset even with grad clipping "
                         "+ input winsorizing -- see module docstring")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    device = args.device or ev.pick_device()
    spec = M.TrainSpec(epochs=args.epochs, patience=args.patience,
                       batch=args.batch, lr=args.lr)
    print(f"arch={ARCH} dataset={DATASET} device={device} recipe={spec}", flush=True)

    baseline = run_arm("net_dispatch_totdem", "baseline (no future demand)", spec, args.seed, device)
    withfcst = run_arm("net_dispatch_totdem_fcst", "+AEMO future demand", spec, args.seed, device)

    b_wape = baseline["metrics"]["average"]["WAPE"]
    f_wape = withfcst["metrics"]["average"]["WAPE"]
    delta = f_wape - b_wape
    pct = 100 * delta / b_wape if b_wape else float("nan")

    print("\n=== comparison (test WAPE, lower is better) ===")
    print(f"  baseline        : {b_wape:.4f}")
    print(f"  +AEMO future dem: {f_wape:.4f}")
    print(f"  delta           : {delta:+.4f}  ({pct:+.1f}%)")
    print("\n  per-target WAPE:")
    for t in baseline["metrics"]["per_target"]:
        bw = baseline["metrics"]["per_target"][t]["WAPE"]
        fw = withfcst["metrics"]["per_target"][t]["WAPE"]
        print(f"    {t:<20} baseline={bw:.4f}  +fcst={fw:.4f}  delta={fw - bw:+.4f}")

    result = {
        "dataset": DATASET, "arch": ARCH, "recipe": vars(spec), "seed": args.seed,
        "baseline": baseline, "with_future_demand": withfcst,
        "wape_delta": delta, "wape_delta_pct": pct,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
