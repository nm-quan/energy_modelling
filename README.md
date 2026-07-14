energy_modelling — portable research snapshot

Self-contained slice of the VIC NEM demand->dispatch study: the data, the
forecasting pipeline, the no-RevIN LSTM dispatch model + evaluation, the demand
simulation, and the model-behaviour experiment. Everything runs from this folder
alone (no reference to the parent repo).

Layout

  data/                       energy fuels (generation), demand+price (market),
                              import/export (interconnector), curtailment, plus
                              capacity_constraints.json / empirical_bounds.json and
                              the preprocessed net_dispatch table.
  CONSTRAINT.md               capacity / ramp / battery-balance constraint spec.
  lib/                        shared importable modules:
      pipeline.py             load data -> feature/target table -> scaled windows.
      models.py               model architectures (LSTM used here; iTransformer for behaviour).
      evaluate.py             train loop (AdamW+MSE, early stop) + MAE/RMSE/WAPE/R^2.
      plotting.py             loss curve + actual-vs-pred + dispatch-stack figures.
      stack_plots.py          all-energy stacked dispatch graph (with renewables/curtailment).
      sim_common.py           model+data loader and the selfsum/scenario free-window rollout.
      shift_model.py          FixedPercentageShift demand model.
  ml/
      train_lstm.py           train + evaluate the LSTM (reproduces lstm_5min_mse/).
      check_caps.py           verify a predictions CSV against CONSTRAINT.md.
      lstm_5min_mse/          bundled weights, scalers, metrics, predictions, figures.
      itransformer_totdem/    bundled iTransformer weights + predictions (for behaviour).
  demand_simulation/
      sweep_eqnd.py           rebound sweep, demand-side net_demand through the LSTM.
      sweep_eqnd/             bundled sweep results + figure.
  behaviour/                  one-step vs autoregressive behaviour of the iTransformer
                              (report.md + figures + scripts).

Setup

  python -m pip install numpy pandas scikit-learn torch matplotlib pyarrow
  (xgboost is optional; only the unused XGBoost wrapper in models.py needs it.)
  A GPU is optional — everything falls back to CPU.

Run

  ML
    python ml/train_lstm.py        retrain the LSTM (optional; weights are bundled).
    python ml/check_caps.py        verify ml/lstm_5min_mse/predictions.csv vs constraints.
                                   (--csv <file> --suffix _pred to check another CSV.)

  Demand simulation
    python demand_simulation/sweep_eqnd.py
                                   demand-side net_demand rebound sweep through the LSTM;
                                   writes sweep_eqnd/{result_sweep.md, sweep.csv, figure/}.

  Bottleneck study (constraints + WAPE + demand response; colab/study_bottlenecks.ipynb)
    python constraints/eval_hist_models.py --ckpt-dir weights
                                   TF + CL tables incl. rayenfd(+steam passthrough),
                                   per-channel WAPE, per-day SOC.
    python demand_simulation/study_shift.py --scenario increase --g 10
                                   simulated increased-load rollout (parquet-free);
                                   also --scenario reshape (legacy-comparable).
    python constraints/study_report.py
                                   A/B/C ladder gates -> study_summary.md,
                                   study_verdicts.json, trade-off figure.

  Behaviour
    python behaviour/run_ar_behaviour.py   one-step vs closed-loop rollout (iTransformer).
    python behaviour/plot_behaviour.py     stacked generation + curtailment.
    python behaviour/plot_coal_pred.py     coal predicted-vs-actual stacked view.

Notes

  - The dispatch model is the no-RevIN LSTM: the iTransformer is RevIN-invariant to
    demand level and collapses in closed loop (see behaviour/report.md), so it does
    not respond to a demand shift. The LSTM does.
  - Data is 5-min VIC, ~1 year. Chronological split: train to 2026-02-28, val to
    2026-04-11, test to 2026-05-18.
  - All bundled artifacts can be regenerated from the scripts above.
