"""Training loop + evaluation metrics for the neural forecasters.

Extracted from the original benchmark harness so the clean folder has a single,
self-contained train/evaluate module. Used by ml/train_lstm.py.

  compute_metrics  per-target MAE / RMSE / WAPE / R^2 + macro mean
  train_neural     AdamW + MSE, early stopping on validation MSE
  predict_neural   batched forward pass (scaled outputs)
  _seed_all        reproducible seeding
  count_params     trainable parameter count

`spec` is any object exposing .lr .weight_decay .batch .epochs .patience
(e.g. models.TrainSpec).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- metrics -----------------------------

def compute_metrics(true: np.ndarray, pred: np.ndarray, targets: list[str]) -> dict:
    """Per-target MAE/RMSE/WAPE/R^2 + a macro mean across targets.

    WAPE per target = sum(|err|)/sum(|y|); the macro average is a simple mean
    across targets (not pooled) so every fuel is weighted equally.
    """
    per, maes, rmses, wapes, r2s = {}, [], [], [], []
    for i, t in enumerate(targets):
        yt, yp = true[:, i], pred[:, i]
        err = yp - yt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        denom = float(np.sum(np.abs(yt)))
        wape = float(np.sum(np.abs(err)) / denom) if denom > 0 else float("nan")
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        per[t] = {"MAE": mae, "RMSE": rmse, "WAPE": wape, "R2": r2}
        maes.append(mae); rmses.append(rmse); wapes.append(wape); r2s.append(r2)
    return {
        "per_target": per,
        "average": {
            "MAE": float(np.mean(maes)),
            "RMSE": float(np.mean(rmses)),
            "WAPE": float(np.mean(wapes)),
            "R2": float(np.mean(r2s)),
        },
    }


# ----------------------------- training -----------------------------

def _seed_all(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(override: str | None = None) -> str:
    """cuda (Colab/GPU box) > mps (local Mac) > cpu."""
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_neural(model: nn.Module, data: dict, spec, device: str,
                 name: str) -> tuple[nn.Module, dict]:
    """AdamW + MSE training with early stopping on validation MSE.

    Returns (best_model, history) where history has per-epoch train/val MSE.
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


def predict_neural(model: nn.Module, X: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    """Batched forward pass -> scaled (N, n_targets) array (inverse-transform outside)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)
