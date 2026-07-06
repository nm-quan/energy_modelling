"""Reusable figures for the forecasting studies: full test, random day, dispatch stack."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# dispatch stack: generation stacks above 0, battery charging drawn below 0 (load)
STACK_POS = ["coal_brown", "gas_steam", "gas_ocgt", "hydro", "battery_discharging"]
STACK_NEG = "battery_charging"
COLORS = {
    "coal_brown": "saddlebrown",
    "gas_steam": "#d62728",        # red (was orange, clashed with solar gold)
    "gas_ocgt": "#ff7f0e",         # orange (was gold, clashed with solar gold)
    "hydro": "royalblue",
    "battery_discharging": "#9467bd",  # purple (was seagreen, clashed with wind green)
    "battery_charging": "dimgray",
}


def plot_all(true, pred, targets, index, fig_dir: Path, res: str, title_tag: str = "",
             history: dict | None = None):
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_full(true, pred, targets, index, fig_dir, res, title_tag)
    steps_per_day = 24 if res == "1h" else 288
    if len(index) > steps_per_day:
        _plot_random_day(true, pred, targets, fig_dir, res, steps_per_day, title_tag)
    if history is not None:
        plot_loss(history, fig_dir, title_tag)


def plot_loss(history: dict, fig_dir: Path, tag: str = ""):
    """Plot train and val MSE per epoch. If history also contains
    train_penalty / val_penalty (composite loss), plot the penalty on a
    second y-axis so both components are visible on one figure."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_mse"]) + 1)
    has_pen = "train_penalty" in history and "val_penalty" in history

    f, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, history["train_mse"], label="train MSE", lw=1.2, color="C0")
    ax.plot(epochs, history["val_mse"], label="val MSE", lw=1.2, color="C1")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled)")
    ax.set_title(f"{tag} — training loss".strip(" —"))

    if has_pen:
        ax2 = ax.twinx()
        ax2.plot(epochs, history["train_penalty"], label="train penalty", lw=1.0,
                 color="C2", linestyle="--")
        ax2.plot(epochs, history["val_penalty"], label="val penalty", lw=1.0,
                 color="C3", linestyle="--")
        ax2.set_ylabel("penalty (normalised)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    else:
        ax.legend(loc="upper right", fontsize=9)
    f.tight_layout(); f.savefig(fig_dir / "loss_curve.png", dpi=120); plt.close(f)


def _plot_full(true, pred, targets, index, fig_dir, res, tag):
    n = len(targets)
    f, axes = plt.subplots(n, 1, figsize=(14, 2.4 * n), sharex=True)
    for i, (ax, t) in enumerate(zip(axes, targets)):
        ax.plot(index, true[:, i], label="actual", lw=0.8)
        ax.plot(index, pred[:, i], label="pred", lw=0.8, alpha=0.8)
        ax.set_ylabel(t); ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{tag} — full test set ({res})".strip(" —"))
    f.tight_layout(); f.savefig(fig_dir / "test_full.png", dpi=120); plt.close(f)


def _plot_random_day(true, pred, targets, fig_dir, res, steps_per_day, tag):
    rng = np.random.default_rng(42)
    start = int(rng.integers(0, len(true) - steps_per_day))
    sl = slice(start, start + steps_per_day)
    x = np.arange(1, steps_per_day + 1)              # 1,2,3,...,24 (or ...,288)
    xlabel = "hour of day" if res == "1h" else "5-min step of day"

    n = len(targets)
    f, axes = plt.subplots(n, 1, figsize=(12, 2.4 * n), sharex=True)
    for i, (ax, t) in enumerate(zip(axes, targets)):
        ax.plot(x, true[sl, i], "o-", ms=3, label="actual")
        ax.plot(x, pred[sl, i], "o-", ms=3, alpha=0.8, label="pred")
        ax.set_ylabel(t); ax.legend(loc="upper right", fontsize=8)
    if res == "1h":
        axes[-1].set_xticks(x)
    axes[-1].set_xlabel(xlabel)
    axes[0].set_title(f"{tag} — random test day ({res})".strip(" —"))
    f.tight_layout(); f.savefig(fig_dir / "test_random_day.png", dpi=120); plt.close(f)

    _plot_stacked(true[sl], pred[sl], targets, x, xlabel, res, fig_dir, tag)


def _plot_stacked(true, pred, targets, x, xlabel, res, fig_dir, tag):
    idx = {t: targets.index(t) for t in targets}
    pos = [t for t in STACK_POS if t in targets]
    f, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, data, name in [(axes[0], true, "actual"), (axes[1], pred, "predicted")]:
        ax.stackplot(x, *[data[:, idx[t]] for t in pos], labels=pos,
                     colors=[COLORS[t] for t in pos], alpha=0.85)
        if STACK_NEG in targets:
            ax.fill_between(x, 0, -data[:, idx[STACK_NEG]],
                            color=COLORS[STACK_NEG], alpha=0.6, label=STACK_NEG + " (load)")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(name); ax.set_xlabel(xlabel)
        if res == "1h":
            ax.set_xticks(x)
    axes[0].set_ylabel("MW")
    axes[1].legend(loc="upper right", fontsize=8)
    f.suptitle(f"{tag} — dispatch stack ({res}) — battery charging shown negative".strip(" —"))
    f.tight_layout(); f.savefig(fig_dir / "test_stacked_day.png", dpi=120); plt.close(f)
