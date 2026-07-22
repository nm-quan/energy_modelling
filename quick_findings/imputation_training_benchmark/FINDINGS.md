# Are the imputation models trained comparably enough to benchmark? — and how they train

**Scope:** the two learned gap-imputers — the **BiLSTM** (`imputation/train.py` →
`imputation/results/bilstm_{none,posthoc,rayen_traj}.pt`) and the **iTransformer**
(`imputation/itr_train.py` → `imputation/results/itr/itr_{T1..T4}.pt`). Loss curves in
`loss_curves.png`; the per-epoch numbers behind them are in `loss_curves_data.json`.

## TL;DR

**Same training *harness*, different training *objective* and *evaluation protocol*.**
They share the optimizer, schedule, data, and seed, so their **convergence dynamics are
directly comparable** (that's what the figure shows). But the **absolute quality numbers
are not yet apples-to-apples**: the two families use different losses, different
early-stopping metrics, and — critically — are scored in **two different benchmark
harnesses on different gap tasks and different metrics**. As configured today you can
say *"both converge cleanly under the same recipe"*; you **cannot** yet say *"model A
beats model B"* from their reported numbers.

## What is genuinely shared (the fair part)

| knob | BiLSTM | iTransformer |
|---|---|---|
| data source | `gap_data.load_flats()` — same 5-yr prepared.npz | same |
| train windows | **40,000** | **40,000** |
| optimizer | AdamW, **lr 1e-3, wd 1e-5** | AdamW, **lr 1e-3, wd 1e-5** |
| batch | **128** | **128** |
| budget | **max 100 epochs, patience 10** | **max 100 epochs, patience 10** |
| seed | **0** | **0** |
| checkpoint | best-val state restored at end | best-val state restored at end |

*(The stale `bilstm_recon.json` on disk says `n_train=30000`; that is an **earlier** run.
The committed `bilstm_*.pt` were trained with the default 40,000 — verified from the Colab
training commands — so the two families now match on train size.)*

## What differs (why the reported numbers aren't directly comparable)

| axis | BiLSTM | iTransformer | benchmark impact |
|---|---|---|---|
| **training loss** | masked **MSE** + `0.1·`soft-balance (`none` arm: no balance term) | masked **MAE** (+ `0.1·`aux recon of observed dispatch) | different objective → different bias (MSE under-weights the volatile battery channel; MAE doesn't) |
| **early-stop metric** | **val recon-WAPE** through the projection Π | **val gap-MAE (MW)** | the two right-hand panels are on different scales — **do not read across them** |
| **output param** | residual over a **linear-interp skeleton** | **direct** fill | different inductive bias for smooth channels |
| **gap task** | fixed **36-step (3 h)** gaps | **mixed 3–12 h + whole-day blackout** | not the same prediction problem |
| **eval harness** | WAPE @ 800 all-hour windows (`benchmark.py`) | MAE(MW) @ 3h/6h/12h/blackout (`itr_bench.py`) | **separate tables** — see below |
| **ablation axis** | varies the **constraint mode** (none/posthoc/unrolled/rayen) | varies the **training method** T1–T4, *then* the constraint C0–C3 | the two families ablate different things |

**The smoking gun:** `imputation/results/benchmark.md` (the unified WAPE table that already
holds interp+projection and the classical baselines) lists **every bilstm mode as
`_not run yet_`**, and the iTransformer never appears in it at all — it lives in its own
MAE table (`itr/d1_*`). So today there is **no single row-for-row comparison** of the two.

## How they get trained (walkthrough)

Both follow the same loop: sample 40k masked gap windows once, then each epoch shuffle →
forward on the gap cells → backward (AdamW) → score a held-out val set → keep the best-val
weights → stop after 10 epochs without improvement.

- **BiLSTM** predicts a *deviation* added to a linear-interp skeleton; the loss adds a soft
  balance penalty; early stopping ranks epochs by recon-WAPE **after** the same exact
  projection Π used at deployment (so in-graph modes aren't mis-ranked). Runs stopped at
  14–19 epochs.
- **iTransformer** predicts the fill directly; loss is masked MAE (+ SAITS-style aux recon
  for the T1–T3 arms); early stopping ranks by gap-MAE in MW through each arm's deployed
  map. Runs stopped at 25–61 epochs (slower, deeper convergence).

## Reading the loss curves

- **Both families converge cleanly and monotonically** under the shared recipe — evidence
  the harness itself is sound and comparable.
- **BiLSTM `rayen-traj` is the outlier** (flat train loss ~0.39, val-WAPE ~0.39): training
  *inside* the differentiable RAYEN ray-shoot barely learns — matching the iTransformer's
  C3 RAYEN result and the repo's standing note that RAYEN's conservatism, not the network,
  drives its error. `unrolled`/`posthoc`/`none` all converge to ~0.11 train / ~0.30 val.
- **iTransformer T1–T4 are within a whisker of each other** (val gap-MAE 88–100 MW), which
  is exactly why `d1_training_table.md` flags T2/T3/T4 as a **<5 % tie**. Caveat: each arm
  validates on **its own gap distribution** (T3 on blackout), so even these four per-epoch
  curves aren't perfectly co-scaled — the honest cross-arm number is the *common* val
  blackout-MAE, not these curves.

## To make them head-to-head benchmarkable

Score **both** trained models through **one** harness: the same held-out gap set (same
lengths + blackout), the same metric (WAPE **and** MAE-MW reported together), and the same
projection Π. Concretely: run the bilstm modes into `benchmark.py` (fill the `_not run yet_`
rows) and add an iTransformer adapter to that same table, or extend `itr_bench.py` to also
ingest the bilstm checkpoints. Then the comparison is legitimate.

## Provenance & caveats

Neither training script persists a per-epoch history to disk (both only `print` it and save
final `best_val`). The curves here were **extracted from the committed Colab output cells**
(`colab/imputation_gapfill.ipynb`, `colab/transformer_imputer.ipynb`) — they are the real
runs that produced the committed weights (epoch counts match the `*_metrics/*.json`), not a
re-run. Regenerate with `python3 plot_loss.py <scratchdir> <outdir>` (data JSON alongside).
