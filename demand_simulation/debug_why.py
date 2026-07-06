"""Debug, visually, WHY the models do/don't respond to demand.

Two panels you can read at a glance:

LEFT  — Demand response curves. Take real test windows, sweep ONLY the demand
        channels (demand_mw + net_demand) over the recent 3h by a multiplier, and
        plot each model's predicted net_demand vs that multiplier. A model that USES
        demand slopes up; a model that IGNORES it is a flat horizontal line.

RIGHT — Why they're allowed to ignore it: a parameter-free PERSISTENCE baseline
        (predict next 5-min dispatch = current dispatch) vs the trained models, on
        test WAPE. If persistence is already competitive, demand is not needed for
        accuracy, so a model is free to ignore it.

Prints the redundancy fact: corr(net_demand input, signed sum of dispatch targets).

    python demand_simulation/debug_why.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import pipeline                 # noqa: E402
import sim_common as sc         # noqa: E402
import models as M              # noqa: E402
import evaluate as ev           # noqa: E402

OUT = Path(__file__).resolve().parent / "sweep_eqnd" / "figure"
RECENT = 36   # last 3h (free-window length) of the lookback to perturb


def load(arch, w, nf, dev):
    m = M.make_neural(arch, nf, 6); m.load_state_dict(torch.load(w, map_location=dev))
    return m.to(dev).eval()


def main():
    dev = "cpu"
    data = pipeline.prepare("5min", 24, 1, "net_dispatch_totdem", save=False)
    fc, xs, ys = data["feat_cols"], data["x_scaler"], data["y_scaler"]
    nf = len(fc); nd_i, dem_i = fc.index("net_demand"), fc.index("demand_mw")

    R = Path(".").resolve()
    models = {
        "LSTM no-RevIN":         load("lstm", R/"ml"/"lstm_5min_mse"/"lstm_5min_mse.pt", nf, dev),
        "LSTM +RevIN":           load("lstm_revin", R/"ml"/"lstm_revin_5min"/"lstm_revin_5min.pt", nf, dev),
        "iTransformer +RevIN":   load("itransformer", R/"ml"/"itransformer_totdem"/"itransformer_totdem.pt", nf, dev),
        "iTransformer no-RevIN": load("itransformer_norevin", R/"ml"/"itransformer_norevin_5min"/"itransformer_norevin_5min.pt", nf, dev),
    }

    # windows in the response region (free window) — where a demand shift would act
    ti = pipeline.build_table("5min")
    test_index = data["test_index"]
    mask = sc.response_mask(test_index, 1, (11, 14))
    X0 = data["Xte"][mask][:512].copy()

    # raw = scaled*scale + mean ; multiply raw by alpha on the recent steps ->
    # new_scaled = alpha*scaled + (alpha-1)*mean/scale
    mu = xs.mean_; sd = xs.scale_
    alphas = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    curves = {name: [] for name in models}
    for a in alphas:
        X = X0.copy()
        for c in (nd_i, dem_i):
            X[:, -RECENT:, c] = a * X[:, -RECENT:, c] + (a - 1.0) * mu[c] / sd[c]
        for name, m in models.items():
            pred = sc.predict(m, X, ys, dev)
            curves[name].append(sc.net_demand(pred).mean())
    # normalize each curve to its alpha=1 value (% change)
    base_i = list(alphas).index(1.0)

    # persistence baseline vs models (test WAPE)
    true = ys.inverse_transform(data["Yte"])
    pers = true[:-1]; pers_true = true[1:]
    wape = {"persistence (next=current)":
            ev.compute_metrics(pers_true, pers, sc.TARGETS)["average"]["WAPE"]}
    for name, m in models.items():
        pred = ys.inverse_transform(ev.predict_neural(m, data["Xte"], dev))
        wape[name] = ev.compute_metrics(true, pred, sc.TARGETS)["average"]["WAPE"]

    # redundancy fact
    nd_raw = ys.inverse_transform(data["Yte"]) @ sc.SIGN   # signed sum of targets (=net_demand by constr.)
    # net_demand input channel (unscaled) at the target step is not directly in Yte;
    # show corr of the input net_demand vs signed sum of dispatch history instead:
    hist_sum = (data["Xte"][:, -1, :6] * sd[:6] + mu[:6]) @ sc.SIGN[[0, 2, 3, 1, 4, 5]]
    nd_in = data["Xte"][:, -1, nd_i] * sd[nd_i] + mu[nd_i]
    corr = np.corrcoef(nd_in, hist_sum)[0, 1]
    print(f"redundancy: corr(net_demand input , signed sum of dispatch history) = {corr:.4f}")
    print("test WAPE:")
    for k, v in wape.items():
        print(f"  {k:28s} {v:.4f}")

    # ---- figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    colors = {"LSTM no-RevIN": "#1f77b4", "LSTM +RevIN": "#17becf",
              "iTransformer +RevIN": "#d62728", "iTransformer no-RevIN": "#ff7f0e"}
    for name in models:
        y = np.array(curves[name]); y = 100 * (y / y[base_i] - 1)
        axL.plot(alphas, y, marker="o", color=colors[name], label=name)
    axL.axvline(1.0, color="k", lw=0.6, ls="--")
    axL.axhline(0.0, color="k", lw=0.6)
    axL.set_xlabel("demand input multiplier (recent 3h)  — sweeping ONLY demand_mw + net_demand")
    axL.set_ylabel("predicted net_demand change (%)")
    axL.set_title("Response curve: flat line = model ignores demand")
    axL.legend(); axL.grid(True, alpha=0.3)

    names = list(wape.keys()); vals = [wape[n] for n in names]
    bar_c = ["gray"] + [colors[n] for n in names[1:]]
    axR.barh(names, vals, color=bar_c)
    for i, v in enumerate(vals):
        axR.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    axR.invert_yaxis()
    axR.set_xlabel("test WAPE (lower = better)")
    axR.set_title("Why they can ignore demand:\npersistence already competitive")
    fig.suptitle("Debugging the demand (non-)response: response curves + persistence baseline")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "debug_why_demand_response.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
