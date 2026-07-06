"""Autoregressive (closed-loop) behaviour of itransformer_totdem on the 6 energies.

Two rollouts over the test set, no demand shift:
  - predicted (one-step): each step sees the TRUE 288-step history (one-step
    ahead). This is the standard "running on the test set" model prediction.
  - autoregressive: seed the window once at the start of the test set, then feed
    the model's own (ReLU-clamped) predictions back into the 6 energy channels and
    recompute net_demand each step. Exogenous inputs (demand_mw, price, calendar)
    stay on real data. Pure free-running, no reseeding.

Outputs to behaviour/figure/:
  - ar_stacked_full.png / ar_stacked_week.png / ar_stacked_day.png
    (3 panels: actual | teacher-forced | autoregressive)
And a per-energy comparison table to behaviour/document_ar.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import sim_common as sc          # noqa: E402
from plotting import COLORS, STACK_POS, STACK_NEG  # noqa: E402

FIG = HERE / "figure"
FIG.mkdir(parents=True, exist_ok=True)

PRED_TO_FEAT = [0, 2, 3, 1, 4, 5]   # TARGETS order -> feature energy positions 0..5
ND_FEAT = 6                          # net_demand feature index (17-feature totdem layout)


def teacher_forced(model, Xte, ys, device):
    """One-step predictions on the true test windows, ReLU-clamped (MW)."""
    return sc.predict(model, Xte, ys, device)


def autoregressive(model, s, device):
    """Closed-loop rollout seeded once at the test start. Returns (N,6) MW."""
    feat_cols, xs, ys, lb = s["feat_cols"], s["x_scaler"], s["y_scaler"], s["lb"]
    full, test_index = s["full"], s["test_index"]
    N, L = len(test_index), len(full)

    fs = xs.transform(full[feat_cols].values).astype(np.float32)
    fs_t = torch.from_numpy(fs).to(device)
    ymean = torch.tensor(ys.mean_, dtype=torch.float32, device=device)
    yscale = torch.tensor(ys.scale_, dtype=torch.float32, device=device)
    xmean = torch.tensor(xs.mean_, dtype=torch.float32, device=device)
    xscale = torch.tensor(xs.scale_, dtype=torch.float32, device=device)
    xmean_e, xscale_e = xmean[:6], xscale[:6]
    sign_t = torch.tensor(sc.SIGN, dtype=torch.float32, device=device)
    p2f = torch.tensor(PRED_TO_FEAT, dtype=torch.long, device=device)

    preds = torch.zeros((N, 6), dtype=torch.float32, device=device)
    window = fs_t[None, :lb, :].clone()      # (1, lb, F), seeded once
    with torch.no_grad():
        for i in range(N):
            out = model(window)                               # (1,6) scaled
            pred_mw = torch.clamp(out * yscale + ymean, min=0.0)
            preds[i] = pred_mw[0]
            newrow = fs_t[None, i + lb, :].clone()            # real exo row
            newrow[:, :6] = (pred_mw[:, p2f] - xmean_e) / xscale_e
            nd = pred_mw @ sign_t
            newrow[:, ND_FEAT] = (nd - xmean[ND_FEAT]) / xscale[ND_FEAT]
            window = torch.cat([window[:, 1:, :], newrow[:, None, :]], dim=1)
            if (i + 1) % 2000 == 0:
                print(f"  AR step {i + 1}/{N}")
    return preds.cpu().numpy()


# ----------------------------- metrics -----------------------------

def metrics(true, pred):
    err = pred - true
    mae = np.abs(err).mean(0)
    rmse = np.sqrt((err ** 2).mean(0))
    wape = np.abs(err).sum(0) / np.abs(true).sum(0)
    bias = pred.mean(0) - true.mean(0)
    return mae, rmse, wape, bias


def report(true, tf, ar, targets, out):
    mae_t, rmse_t, wape_t, bias_t = metrics(true, tf)
    mae_a, rmse_a, wape_a, bias_a = metrics(true, ar)
    lines = ["Autoregressive vs predicted (itransformer_totdem, test set)", ""]
    lines.append("Per energy: one-step prediction vs pure autoregressive rollout.")
    lines.append("WAPE = sum|err| / sum|actual|. Bias = mean(pred) - mean(actual), MW.")
    lines.append("")
    lines.append("| energy | MAE pred | MAE ar | RMSE pred | RMSE ar | WAPE pred | WAPE ar | bias pred | bias ar |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, t in enumerate(targets):
        lines.append(f"| {t} | {mae_t[i]:.1f} | {mae_a[i]:.1f} | {rmse_t[i]:.1f} | {rmse_a[i]:.1f} "
                     f"| {wape_t[i]:.4f} | {wape_a[i]:.4f} | {bias_t[i]:+.1f} | {bias_a[i]:+.1f} |")
    lines.append(f"| AVG | {mae_t.mean():.1f} | {mae_a.mean():.1f} | {rmse_t.mean():.1f} | {rmse_a.mean():.1f} "
                 f"| {wape_t.mean():.4f} | {wape_a.mean():.4f} | {bias_t.mean():+.1f} | {bias_a.mean():+.1f} |")
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    print("\n" + text)


# ----------------------------- plotting -----------------------------

def draw_panel(ax, t, arr, ti):
    pos = [p for p in STACK_POS if p in ti]
    ax.stackplot(t, *[arr[:, ti[p]] for p in pos], labels=pos,
                 colors=[COLORS[p] for p in pos], alpha=0.88)
    ax.fill_between(t, 0, -arr[:, ti[STACK_NEG]], color=COLORS[STACK_NEG],
                    alpha=0.6, label=STACK_NEG + " (load)")
    ax.axhline(0, color="k", lw=0.6)
    ax.margins(x=0)
    ax.grid(True, axis="y", alpha=0.3)


def stacked3(t, true, tf, ar, targets, outpath, day_axis=False):
    ti = {t_: i for i, t_ in enumerate(targets)}
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, sharey=True)
    for ax, (name, a) in zip(axes, [("Actual", true),
                                    ("Predicted (one-step)", tf),
                                    ("Autoregressive (closed-loop)", ar)]):
        draw_panel(ax, t, a, ti)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_ylabel("MW")
    if day_axis:
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved {outpath}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    s = sc.load_model_and_data(device)
    model, data = s["model"], s["data"]
    ys = s["y_scaler"]
    targets = sc.TARGETS
    ti_local = pd.DatetimeIndex(s["test_index"]).tz_localize(None)

    true = ys.inverse_transform(data["Yte"])
    print("teacher-forced...")
    tf = teacher_forced(model, data["Xte"], ys, device)
    print("autoregressive...")
    ar = autoregressive(model, s, device)

    report(true, tf, ar, targets, HERE / "document_ar.md")

    # save trajectories for replication
    out = pd.DataFrame({"interval": pd.DatetimeIndex(s["test_index"]).astype(str)})
    for i, t in enumerate(targets):
        out[f"{t}_actual"] = true[:, i]
        out[f"{t}_pred"] = tf[:, i]
        out[f"{t}_ar"] = ar[:, i]
    out.to_csv(HERE / "predictions_ar.csv", index=False)

    # collapse timescale: first step where AR total generation stays below
    # half the actual total generation for the rest of the rollout
    gen_act = true[:, [0, 1, 2, 3, 5]].sum(1)      # exclude battery_charging (load)
    gen_ar = ar[:, [0, 1, 2, 3, 5]].sum(1)
    below = gen_ar < 0.5 * gen_act.mean()
    collapse = next((i for i in range(len(below)) if below[i:].all()), None)
    if collapse is not None:
        hrs = collapse * 5 / 60
        print(f"AR collapse: step {collapse} (~{hrs:.1f} h into rollout); "
              f"steady AR coal ~{ar[collapse:, 1].mean():.0f} MW vs actual ~{true[:, 1].mean():.0f} MW")

    # full test set
    stacked3(ti_local, true, tf, ar, targets, FIG / "ar_stacked_full.png")

    # week + day from the START of the test set, where the AR rollout is still
    # alive and the decay/collapse is visible (collapse completes ~day 4).
    df_idx = pd.Series(np.arange(len(ti_local)), index=ti_local)
    start = ti_local.normalize().unique()[0]
    wk = df_idx[(ti_local >= start) & (ti_local < start + pd.Timedelta(days=7))].values
    stacked3(ti_local[wk], true[wk], tf[wk], ar[wk], targets, FIG / "ar_stacked_week.png")
    # 2nd full day: AR has structure but is visibly drifting from actual
    day0 = start + pd.Timedelta(days=1)
    dd = df_idx[(ti_local >= day0) & (ti_local < day0 + pd.Timedelta(days=1))].values
    stacked3(ti_local[dd], true[dd], tf[dd], ar[dd], targets, FIG / "ar_stacked_day.png", day_axis=True)


if __name__ == "__main__":
    main()
