"""Plot per-epoch training/val loss for the imputation benchmark models.

Single-axis facets (never dual-axis): BiLSTM and iTransformer use different train
losses (MSE+balance vs MAE) and different early-stop metrics (recon-WAPE vs
gap-MAE-MW), so each metric gets its own panel/scale.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SB = Path(sys.argv[1])
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
runs = json.loads((SB / "loss_curves.json").read_text())
R = {r["tag"]: r for r in runs}

# validated categorical palette (light mode), fixed order
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUT = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e7e6e2"; SURF = "#ffffff"

BILSTM = ["none", "posthoc", "unrolled", "rayen_traj"]      # constraint modes
ITR    = ["T1", "T2", "T3", "T4"]                           # training arms

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUT, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUT, "ytick.color": MUT,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))
fig.subplots_adjust(left=0.07, right=0.98, top=0.905, bottom=0.165, hspace=0.36, wspace=0.19)

LABELS = {"none": "none (pure)", "posthoc": "posthoc Π", "unrolled": "unrolled POCS",
          "rayen_traj": "rayen-traj",
          "T1": "T1 mixed+aux", "T2": "T2 +flank-aug", "T3": "T3 blackout",
          "T4": "T4 −aux (T*)"}


def draw(a, tags, key, title, ylab, mark_best, legloc="lower left"):
    for i, tag in enumerate(tags):
        r = R.get(tag)
        if not r:
            continue
        c = PAL[i]
        ep, y = r["epoch"], r[key]
        a.plot(ep, y, color=c, lw=2.0, solid_capstyle="round", zorder=3,
               label=LABELS.get(tag, tag))
        if mark_best:
            bi = min(range(len(y)), key=lambda k: y[k])
            a.scatter([ep[bi]], [y[bi]], s=66, facecolor=c, edgecolor=SURF,
                      linewidth=1.5, zorder=6)
    a.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
    a.set_xlabel("epoch"); a.set_ylabel(ylab)
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    a.margins(x=0.05)
    a.legend(loc=legloc, frameon=False, fontsize=9, handlelength=1.4,
             labelcolor=INK2, borderaxespad=0.4)


draw(ax[0, 0], BILSTM, "loss", "BiLSTM · training loss  (masked MSE + 0.1·balance)",
     "train loss", mark_best=False)
draw(ax[0, 1], BILSTM, "val", "BiLSTM · early-stop metric  (val recon-WAPE, through Π)",
     "val recon-WAPE  (dimensionless)", mark_best=True)
draw(ax[1, 0], ITR, "loss", "iTransformer · training loss  (masked MAE + aux)",
     "train loss", mark_best=False)
draw(ax[1, 1], ITR, "val", "iTransformer · early-stop metric  (val gap-MAE)",
     "val gap-MAE  (MW)", mark_best=True, legloc="upper right")

fig.suptitle("Imputation models — training dynamics  (BiLSTM vs iTransformer)",
             x=0.07, ha="left", fontsize=15, fontweight="bold", color=INK, y=0.965)
fig.text(0.07, 0.085,
         "Filled dot on the right panels = best epoch (the checkpoint actually saved).  The two right panels use different val metrics on different scales — compare within a family, not across.",
         ha="left", va="bottom", fontsize=9, color=INK2)
fig.text(0.07, 0.048,
         "SHARED harness — same 5-yr flats · n_train 40,000 · AdamW lr 1e-3, wd 1e-5 · batch 128 · max 100 epochs, patience 10 · seed 0 · best-val checkpoint.",
         ha="left", va="bottom", fontsize=9, color=INK)
fig.text(0.07, 0.014,
         "DIFFERENT — loss (MSE+balance vs MAE+aux) · early-stop metric (recon-WAPE vs gap-MAE-MW) · output (residual-over-interp vs direct fill) · gap task (fixed 3 h vs mixed/blackout).",
         ha="left", va="bottom", fontsize=9, color="#b34a2f")

png = OUT / "loss_curves.png"
fig.savefig(png, dpi=150)
print("wrote", png)
