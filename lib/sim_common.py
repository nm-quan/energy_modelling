"""Shared pieces for the demand simulation + behaviour scripts.

Loads the dispatch model (iTransformer or the no-RevIN LSTM) on the shared
5-min net_dispatch_totdem pipeline, and provides the rollouts used downstream:

  predict            one-shot (teacher-forced) ReLU-clamped prediction
  ar_free_rollout    closed-loop free-window rollout (the responsive path)
  response_mask      mask of the response region (window edge inside free window)

The iTransformer is RevIN-invariant to demand level and collapses in closed loop
(see behaviour/), so the demand simulation uses the LSTM through the selfsum
ar_free_rollout. Both models share the model(window) -> (B, 6) scaled interface.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent          # lib/
ROOT = HERE.parent                               # energy_modelling/
sys.path.insert(0, str(HERE))                    # pipeline / models / plotting are siblings
import pipeline       # noqa: E402
import models as M    # noqa: E402

# default (iTransformer) and responsive (no-RevIN LSTM) weights, both bundled under ml/
ARCH = "itransformer"
WEIGHTS = ROOT / "ml" / "itransformer_totdem" / "itransformer_totdem.pt"
LSTM_ARCH = "lstm"
LSTM_WEIGHTS = ROOT / "ml" / "lstm_5min_mse" / "lstm_5min_mse.pt"

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]
SIGN = np.array([1, 1, 1, 1, -1, 1], dtype=np.float64)   # net_demand = sum(sign * target)
FREE_HOURS = (11, 14)                                     # 11am-2pm default free window
PRED_TO_FEAT = [0, 2, 3, 1, 4, 5]   # pred (TARGETS order) -> feature energy positions 0..5
ND_FEAT = 6                          # net_demand feature index (net_dispatch_totdem layout)


def net_demand(arr: np.ndarray) -> np.ndarray:
    return arr @ SIGN


def predict(model, Xw, y_scaler, device, batch=1024):
    """One-shot forward pass over windows -> MW, with the non-negativity ReLU clamp."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xw), batch):
            xb = torch.from_numpy(Xw[i:i + batch]).to(device)
            outs.append(model(xb).cpu().numpy())
    pred = y_scaler.inverse_transform(np.concatenate(outs, 0))
    return np.maximum(pred, 0.0)


def load_model_and_data(device, arch=ARCH, weights=WEIGHTS):
    """Prepare scalers/windows, load the dispatch model, and build the unscaled
    feature frame (val tail + test) used to re-window scenarios.

    Defaults to the iTransformer; pass arch=LSTM_ARCH, weights=LSTM_WEIGHTS for
    the responsive no-RevIN LSTM. Returns a dict with: data (pipeline output),
    model, lb, full, test_df, test_index, feat_cols, x_scaler, y_scaler.
    """
    data = pipeline.prepare(resolution="5min", lookback=24, horizon=1,
                            input_mode="net_dispatch_totdem", save=False)
    lb = data["lookback_steps"]
    model = M.make_neural(arch, len(data["feat_cols"]), 6).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    df = pipeline.build_table("5min")
    val_df = df[(df.index > pipeline.TRAIN_END) & (df.index <= pipeline.VAL_END)]
    test_df = df[df.index > pipeline.VAL_END]
    full = pd.concat([val_df.tail(lb), test_df])
    return {"data": data, "model": model, "lb": lb,
            "full": full, "test_df": test_df,
            "test_index": pd.DatetimeIndex(data["test_index"]),
            "feat_cols": data["feat_cols"],
            "x_scaler": data["x_scaler"], "y_scaler": data["y_scaler"]}


def load_lstm(name, n_feats, device):
    """Load a plain no-RevIN LSTM from ml/<name>/<name>.pt."""
    model = M.make_neural("lstm", n_features=n_feats, n_targets=6)
    model.load_state_dict(torch.load(ROOT / "ml" / name / f"{name}.pt", map_location=device))
    return model.to(device).eval()


def response_mask(target_index, h, free_hours):
    """True where the inference window's recent edge (target - h steps) is in the free window."""
    end = pd.DatetimeIndex(target_index) - pd.Timedelta(minutes=5 * h)
    return np.asarray((end.hour >= free_hours[0]) & (end.hour < free_hours[1]))


def ar_free_rollout(model, fs_base, fs_scen, lb, free_test, x_scaler, y_scaler,
                    device, nd_mode="selfsum"):
    """5-min autoregressive free-window rollout for base + scenario together.

    Returns (base_pred, scen_pred), each (N, 6) MW aligned to the test index.

    Outside the free window every step is a one-step prediction reseeded from the
    real/scenario feature window. Inside the free window the rollout goes closed
    loop: the six dispatch-history channels are fed back from the model's own
    predictions, and (nd_mode="selfsum") net_demand is replaced by the signed sum
    of the predicted dispatch -- so the demand shift reaches the model only through
    demand_mw + price. nd_mode="scenario" keeps the prescribed net_demand instead.
    """
    fs = np.stack([fs_base, fs_scen])                                  # (2, L, F)
    fs_t = torch.from_numpy(fs).to(device)
    ymean = torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device)
    yscale = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=device)
    xmean = torch.tensor(x_scaler.mean_, dtype=torch.float32, device=device)
    xscale = torch.tensor(x_scaler.scale_, dtype=torch.float32, device=device)
    xmean_e, xscale_e = xmean[:6], xscale[:6]
    p2f = torch.tensor(PRED_TO_FEAT, dtype=torch.long, device=device)
    sign = torch.tensor(SIGN, dtype=torch.float32, device=device)

    free_test = np.asarray(free_test)
    N = len(free_test)
    preds = torch.zeros((2, N, 6), dtype=torch.float32, device=device)
    window, free_prev = None, False
    with torch.no_grad():
        for i in range(N):
            free_now = bool(free_test[i])
            if not (free_now and free_prev):                # reseed from real/scenario data
                window = fs_t[:, i:i + lb, :].clone()
            out = model(window)
            pred_mw = torch.clamp(out * yscale + ymean, min=0.0)
            preds[:, i, :] = pred_mw
            if free_now:                                    # autoregress inside the free window
                newrow = fs_t[:, i + lb, :].clone()         # scenario demand_mw / price / net_demand
                newrow[:, :6] = (pred_mw[:, p2f] - xmean_e) / xscale_e
                if nd_mode == "selfsum":
                    nd_pred = pred_mw @ sign
                    newrow[:, ND_FEAT] = (nd_pred - xmean[ND_FEAT]) / xscale[ND_FEAT]
                window = torch.cat([window[:, 1:, :], newrow[:, None, :]], dim=1)
            free_prev = free_now
    p = preds.cpu().numpy()
    return p[0], p[1]
