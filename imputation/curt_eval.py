"""MAE / MRE on random test-set 3h gaps, per output channel, for the two
curtailment-predicting imputers.

Rows
  interp       linear interpolation between the pinned boundaries, for all 8
               channels. The no-model reference.
  plain        BiLSTM -> 8 raw channels, no constraints.
  recursive    BiLSTM -> recursive RAYEN head, AS TRAINED. Its balance target is
               nd_eff = nd - (curt_actual - curt_pred), which uses the TRUE
               curtailment inside the gap.
  recursive*   the same weights with the credit disabled (nd_eff = nd). See below.

THE LEAK, AND WHY BOTH ROWS ARE REPORTED. Coupling curtailment into the balance
target is legitimate for a COUNTERFACTUAL -- there the base curtailment is
historical fact and the model predicts the shifted one. It is NOT legitimate for
IMPUTATION scoring: curtailment inside the gap is masked out of the inputs
precisely because it is unknown, so feeding curt_actual back in through nd_eff
hands the head information the task says it does not have. `recursive` is the
as-trained map (leaky); `recursive*` is what the same weights do without it.
The gap between the two rows is the size of the leak.

Metrics, per channel, over all gap cells:
  MAE = mean |pred - truth|                      (MW)
  MRE = 100 * sum|pred - truth| / sum|truth|     (%)
MRE is unstable for channels that are mostly zero -- the denominator collapses --
so the curtailment and gas_steam columns should be read with that in mind.

    python3 imputation/curt_eval.py --n 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data" / "preprocessed" / "hist" / "5min" / "net_dispatch_ren" / "prepared.npz"
from gap_data import load_flats, TARGETS, SIGN, TARGET_FEAT_IDX        # noqa: E402
import constraints as C                                                # noqa: E402
from model import BiLSTMImputer                                        # noqa: E402
from recursive_head import recursive_rayen                             # noqa: E402
from curt_train import curt_activation                                 # noqa: E402

OUT = HERE / "results" / "plan1"
CURT_COLS = [19, 20]
CTX, GAP = 48, 36
CHANNELS = TARGETS + ["wind_curtailment", "solar_curtailment"]
SHORT = {"coal_brown": "coal", "gas_ocgt": "gas_ocgt", "gas_steam": "gas_steam",
         "hydro": "hydro", "battery_charging": "batt_chg",
         "battery_discharging": "batt_dis",
         "wind_curtailment": "wind_curt", "solar_curtailment": "solar_curt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    f = load_flats()
    nfeat = len(f.feat_cols)
    W = 2 * CTX + GAP
    lb = f.lb_carry
    rng = np.random.default_rng(args.seed)
    # gap must sit in genuine test rows (past the val-tail prefix) and the whole
    # window must fit
    lo = max(0, lb - CTX)
    starts = rng.integers(lo, f.Xte.shape[0] - W, size=args.n)
    print(f"{args.n} random test 3h gaps (seed {args.seed}), {nfeat} features")

    tfi = np.asarray(TARGET_FEAT_IDX)
    cm, cs = f.x_mean[CURT_COLS], f.x_scale[CURT_COLS]

    X = np.stack([f.Xte[s:s + W] for s in starts]).astype(np.float32)
    Yz = np.stack([f.Yte[s + CTX:s + CTX + GAP] for s in starts])
    Ymw = Yz * f.y_scale + f.y_mean                                       # (n,G,6)
    Cmw = X[:, CTX:CTX + GAP][:, :, CURT_COLS] * cs + cm                  # (n,G,2)
    truth = np.concatenate([Ymw, Cmw], -1)                                # (n,G,8)
    pL = f.y_to_mw(f.Yte[starts + CTX - 1])
    pR = f.y_to_mw(f.Yte[starts + CTX + GAP])
    cL = f.Xte[starts + CTX - 1][:, CURT_COLS] * cs + cm
    cR = f.Xte[starts + CTX + GAP][:, CURT_COLS] * cs + cm
    nd = (Ymw @ SIGN).astype(np.float64)                                  # supply-side
    mask = np.ones((args.n, W, 1), np.float32); mask[:, CTX:CTX + GAP] = 0.0
    X[:, CTX:CTX + GAP, tfi] = 0.0
    X[:, CTX:CTX + GAP, CURT_COLS] = 0.0

    preds = {}
    # ---- interp reference, all 8 channels ----
    tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
    preds["interp"] = np.concatenate(
        [pL[:, None] + tt * (pR - pL)[:, None],
         cL[:, None] + tt * (cR - cL)[:, None]], -1)

    for arm, n_out in (("plain", 8), ("recursive", 9)):
        m = BiLSTMImputer(n_features=nfeat, n_targets=n_out)
        m.load_state_dict(torch.load(OUT / f"curt_{arm}.pt", map_location="cpu",
                                     weights_only=True))
        m.eval()
        outs = {arm: [], f"{arm}*": []}
        for i in range(0, args.n, 64):
            sl = slice(i, i + 64)
            with torch.no_grad():
                raw = m(torch.from_numpy(X[sl]),
                        torch.from_numpy(mask[sl]))[:, CTX:CTX + GAP].double()
                curt = curt_activation(raw[..., -2:], torch.tensor(cm), torch.tensor(cs))
                if arm == "plain":
                    P = raw[..., :6].numpy() * f.y_scale + f.y_mean
                    outs[arm].append(np.concatenate([P, curt.numpy()], -1))
                    continue
                pLt, pRt = torch.tensor(pL[sl]).double(), torch.tensor(pR[sl]).double()
                ndt = torch.tensor(nd[sl])
                # as trained: credit uses the TRUE curtailment (leaky for imputation)
                nd_eff = ndt - (torch.tensor(Cmw[sl]).double() - curt).sum(-1)
                P = recursive_rayen(raw[..., :7], pLt, pRt, nd_eff)
                outs[arm].append(np.concatenate([P.numpy(), curt.numpy()], -1))
                # leak-free: no credit at all
                P2 = recursive_rayen(raw[..., :7], pLt, pRt, ndt)
                outs[f"{arm}*"].append(np.concatenate([P2.numpy(), curt.numpy()], -1))
        for k, v in outs.items():
            if v:
                preds[k] = np.concatenate(v, 0)

    rows = ["interp", "plain", "recursive", "recursive*"]
    L = [f"# MAE / MRE per output channel — {args.n} random test 3h gaps (seed {args.seed})",
         "", f"Both models trained **1 epoch**. Channels 1–6 are dispatch, 7–8 are the "
         "new curtailment outputs.", "",
         "`recursive` is the as-trained map, whose balance target "
         "`nd_eff = nd − (curt_actual − curt_pred)` uses the TRUE in-gap curtailment. "
         "That is fine for a counterfactual but is **leakage** when scoring imputation, "
         "since curtailment is masked out of the inputs. `recursive*` is the same "
         "weights with the credit removed. The gap between those two rows is the leak.",
         "", "## MAE (MW)", "",
         "| model | " + " | ".join(SHORT[c] for c in CHANNELS) +
         " | disp agg | curt agg |",
         "| --- |" + " ---: |" * (len(CHANNELS) + 2)]
    mae, mre = {}, {}
    for r in rows:
        e = np.abs(preds[r] - truth)
        mae[r] = e.reshape(-1, 8).mean(0)
        mre[r] = 100 * e.reshape(-1, 8).sum(0) / np.abs(truth).reshape(-1, 8).sum(0)
        L.append(f"| {r} | " + " | ".join(f"{v:,.1f}" for v in mae[r]) +
                 f" | **{mae[r][:6].mean():,.1f}** | **{mae[r][6:].mean():,.1f}** |")
    L += ["", "## MRE (%)  =  100 · Σ|err| / Σ|truth|", "",
          "| model | " + " | ".join(SHORT[c] for c in CHANNELS) + " |",
          "| --- |" + " ---: |" * len(CHANNELS)]
    for r in rows:
        L.append(f"| {r} | " + " | ".join(f"{v:,.1f}" for v in mre[r]) + " |")
    L += ["", "_MRE is unstable where the truth is mostly zero — the denominator "
          "collapses. Share of gap cells at or below 1 MW: " +
          ", ".join(f"{SHORT[c]} {100*(np.abs(truth.reshape(-1,8)[:,k])<=1).mean():.0f}%"
                    for k, c in enumerate(CHANNELS)) + "._", ""]

    # feasibility of each row
    L += ["## Feasibility of the filled window", "",
          "| model | worst balance resid (MW) | ramp overshoot (MW) | most-negative (MW) |",
          "| --- | ---: | ---: | ---: |"]
    for r in rows:
        P = preds[r][..., :6]
        b = np.abs((P * SIGN).sum(-1) - nd).max()
        fullp = np.concatenate([pL[:, None], P, pR[:, None]], 1)
        d = np.diff(fullp, axis=1)
        ro = float((np.maximum(d - C.R_UP, 0) + np.maximum(-d - C.R_DN, 0)).max())
        L.append(f"| {r} | {b:,.2f} | {ro:,.2f} | {min(0.0, P.min()):,.2f} |")
    L += ["", "_`recursive` is audited against nd; its residual is the curtailment "
          "credit, not a solver error. `recursive*` has no credit, so its residual is "
          "the head's true balance error._", ""]

    (OUT / "curt_eval.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT/'curt_eval.md'}")


if __name__ == "__main__":
    main()
