"""Requested Colab metrics for itransformer_rayenfd.

Evaluates the fixed-demand Rayen iTransformer, whose hard layer pins demand to
the input window's latest net_demand: nD(t-1).

Outputs exactly:
  regular_test:   WAPE, n_ramp_violations, n_netdemand_balance_violations,
                  SOC_possible_starting
  simulated_test: n_ramp_violations, n_netdemand_balance_violations,
                  SOC_possible_starting

The simulated test is the constraint-study closed-loop rollout over the
TEST-split stress episodes in constraints/stress_episodes.json. It intentionally
does not report WAPE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "ml"))

import evaluate as ev  # noqa: E402
import models as M  # noqa: E402
import pipeline  # noqa: E402
from check_caps import BATT_CAP_MWH, RAMPS, RES_HOURS  # noqa: E402

TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]
SIGN = np.array([1, 1, 1, 1, -1, 1], dtype=np.float64)
RAMP_TOL = 0.6
DEMAND_TOL = 10.0
ETA_RT = 0.834


def _find_weight(ckpt_dir: Path, seed: int) -> Path:
    name = f"itransformer_rayenfd_hist_s{seed}.pt"
    candidates = [
        ckpt_dir / name,
        ckpt_dir / "itransformer_rayenfd_hist" / name,
        ROOT / "ml" / "itransformer_rayenfd_hist" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find itransformer_rayenfd weights. Expected one of:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def load_model(data: dict, ckpt_dir: Path, seed: int, device: str) -> torch.nn.Module:
    xs, ys = data["x_scaler"], data["y_scaler"]
    nd_idx = data["feat_cols"].index("net_demand")
    ramp_dn = [abs(RAMPS[t][0]) for t in TARGETS]
    ramp_up = [RAMPS[t][1] for t in TARGETS]
    model = M.make_rayen(
        "itransformer", xs, ys, ramp_up, ramp_dn,
        nd_feat_idx=nd_idx, n_features=len(data["feat_cols"]),
        fix_demand=True, passthrough_idx=(2,),
    )
    state = torch.load(_find_weight(ckpt_dir, seed), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _predict(model, X: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def _ramp_violations(pred: np.ndarray, prev: np.ndarray) -> int:
    ramp_up = np.array([RAMPS[t][1] for t in TARGETS], dtype=np.float64)
    ramp_dn = np.array([abs(RAMPS[t][0]) for t in TARGETS], dtype=np.float64)
    delta = pred - prev
    return int(((delta > ramp_up + RAMP_TOL) | (delta < -(ramp_dn + RAMP_TOL))).sum())


def _ramp_violations_trajectory(pred: np.ndarray, prev_first: np.ndarray) -> int:
    ramp_up = np.array([RAMPS[t][1] for t in TARGETS], dtype=np.float64)
    ramp_dn = np.array([abs(RAMPS[t][0]) for t in TARGETS], dtype=np.float64)
    delta = np.diff(np.concatenate([prev_first[None], pred]), axis=0)
    return int(((delta > ramp_up + RAMP_TOL) | (delta < -(ramp_dn + RAMP_TOL))).sum())


def _soc_possible_starting(pred: np.ndarray, segs: np.ndarray) -> bool:
    """True iff every segment has some SOC0 in [0, capacity].

    For regular_test, segments are calendar days. For simulated_test, segments
    are stress episodes. In both cases this is the same physical existence test:
    max(cumulative reservoir energy) - min(cumulative reservoir energy) <= cap.
    """
    eta = np.sqrt(ETA_RT)
    chg = np.clip(pred[:, TARGETS.index("battery_charging")], 0, None)
    dis = np.clip(pred[:, TARGETS.index("battery_discharging")], 0, None)
    d_e = (chg * eta - dis / eta) * RES_HOURS["5min"]
    for seg in np.unique(segs):
        cum = np.concatenate([[0.0], np.cumsum(d_e[segs == seg])])
        if float(cum.max() - cum.min()) > BATT_CAP_MWH + 1e-6:
            return False
    return True


def regular_test_metrics(model, data: dict, device: str) -> dict:
    xs, ys = data["x_scaler"], data["y_scaler"]
    nd_idx = data["feat_cols"].index("net_demand")
    Xte, Yte = data["Xte"], data["Yte"]
    true = ys.inverse_transform(Yte).astype(np.float64)
    pred = ys.inverse_transform(_predict(model, Xte, device)).astype(np.float64)

    prev = (Xte[:, -1, M.TARGET_FEAT_IDX] * xs.scale_[M.TARGET_FEAT_IDX]
            + xs.mean_[M.TARGET_FEAT_IDX]).astype(np.float64)
    nd_ref = (Xte[:, -1, nd_idx] * xs.scale_[nd_idx] + xs.mean_[nd_idx]).astype(np.float64)
    bal = np.abs(pred @ SIGN - nd_ref)
    days = pd.DatetimeIndex(data["test_index"]).normalize()
    segs = np.unique(days, return_inverse=True)[1]

    return {
        "WAPE": float(ev.compute_metrics(true, pred, TARGETS)["average"]["WAPE"]),
        "n_ramp_violations": _ramp_violations(pred, prev),
        "n_netdemand_balance_violations": int((bal > DEMAND_TOL).sum()),
        "SOC_possible_starting": _soc_possible_starting(pred, segs),
    }


def _load_test_episodes(data: dict) -> list[dict]:
    episodes = json.loads((HERE / "stress_episodes.json").read_text())
    idx, lb = data["test_index"], data["lookback_steps"]
    out = []
    for episode in episodes:
        if episode["split"] != "test":
            continue
        i0 = int(idx.searchsorted(pd.Timestamp(episode["start"], tz=idx.tz)))
        i1 = int(idx.searchsorted(pd.Timestamp(episode["end_excl"], tz=idx.tz)))
        if i1 <= i0:
            continue
        span = idx[max(i0 - lb, 0):i1]
        if not (np.diff(span.values) == np.timedelta64(5, "m")).all():
            continue
        out.append({**episode, "i0": i0, "i1": i1})
    return out


def simulated_test_metrics(model, data: dict, device: str, max_steps: int | None = None) -> dict:
    xs, ys = data["x_scaler"], data["y_scaler"]
    nd_idx = data["feat_cols"].index("net_demand")
    lb = data["lookback_steps"]
    z = np.load(
        pipeline.PREPROCESSED_DIR / "hist" / "5min" / "net_dispatch_totdem" / "prepared.npz",
        allow_pickle=False,
    )
    Xte_flat = z["Xte_flat"]
    tfi = np.array(M.TARGET_FEAT_IDX)
    x_mu, x_sd = xs.mean_[tfi], xs.scale_[tfi]
    episodes = _load_test_episodes(data)
    if not episodes:
        raise RuntimeError("No TEST-split stress episodes found.")

    all_pred, all_ref, all_seg = [], [], []
    n_ramp = 0
    with torch.no_grad():
        for k, episode in enumerate(episodes):
            n = episode["i1"] - episode["i0"]
            if max_steps is not None:
                n = min(n, max_steps)
            block = Xte_flat[episode["i0"]:episode["i0"] + lb + n].copy()
            pred = np.empty((n, 6), dtype=np.float64)
            d_ref = np.empty(n, dtype=np.float64)
            for s in range(n):
                xb = torch.from_numpy(np.ascontiguousarray(block[s:s + lb][None])).to(device)
                p_scaled = model(xb).cpu().numpy()[0]
                mw = ys.inverse_transform(p_scaled[None])[0].astype(np.float64)
                pred[s] = mw
                d_ref[s] = block[s + lb - 1, nd_idx] * xs.scale_[nd_idx] + xs.mean_[nd_idx]
                block[s + lb, tfi] = (mw - x_mu) / x_sd
            prev0 = (Xte_flat[episode["i0"] + lb - 1, tfi] * x_sd + x_mu).astype(np.float64)
            n_ramp += _ramp_violations_trajectory(pred, prev0)
            all_pred.append(pred)
            all_ref.append(d_ref)
            all_seg.append(np.full(n, k, dtype=int))

    pred = np.concatenate(all_pred)
    d_ref = np.concatenate(all_ref)
    segs = np.concatenate(all_seg)
    bal = np.abs(pred @ SIGN - d_ref)
    return {
        "n_ramp_violations": n_ramp,
        "n_netdemand_balance_violations": int((bal > DEMAND_TOL).sum()),
        "SOC_possible_starting": _soc_possible_starting(pred, segs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=str(ROOT / "ml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(HERE / "results" / "rayenfd_requested_metrics.json"))
    parser.add_argument("--sim-max-steps", type=int, default=None,
                        help="Optional smoke-test cap per stress episode.")
    args = parser.parse_args()

    device = ev.pick_device(args.device)
    data = pipeline.load_prepared("5min", 24, 1, "net_dispatch_totdem", "hist")
    model = load_model(data, Path(args.ckpt_dir), args.seed, device)
    result = {
        "regular_test": regular_test_metrics(model, data, device),
        "simulated_test": simulated_test_metrics(model, data, device, args.sim_max_steps),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
