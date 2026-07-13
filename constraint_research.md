# Constraint research — feasible dispatch forecasts at minimal WAPE cost

Milestone goal: the model's predictions should satisfy the physical constraints
(CONSTRAINT.md) — box caps, ramp rates, demand-generation balance, battery SOC
feasibility — while keeping WAPE as low as possible. All experiments live under
`constraints/`; this file is the living plan + status tracker.

## Hypotheses

- H0 (original): the more constraints a model must satisfy, the higher its WAPE.
- H1 (refined, from the demand-anchoring milestone in demand_simulation/findings.md):
  the WAPE cost is **mechanism-dependent, not count-dependent**. Evidence so far:
  inference-time projection cost ~0 WAPE (clamping even improved it: 0.1132 -> 0.1068;
  demand anchoring 0.1068 -> 0.1075), training *through* a hard layer cost +0.21 WAPE,
  and soft penalties (lambda_eb sweep) paid WAPE without ever reaching 0 violations.
  Central claim to test: projection-family methods sit on the Pareto frontier.

## Metrics (constraints/harness.py :: constraint_report)

| metric | definition |
| --- | --- |
| WAPE, R2 | macro mean over the 6 targets (lib/evaluate.py), unchanged |
| n_neg, n_cap | cells with pred < 0; cells with pred > per-target data-max cap |
| n_ramp | consecutive-pair cells outside the per-target asymmetric data ramps, on the predicted trajectory (pred[t] - pred[t-1]) |
| n_demand, mean_demand_mismatch | steps with abs(sum(SIGN*pred) - nd_ref) > 10 MW; sum(abs resid)/sum(abs nd_ref) x100. Reported against BOTH references (below) |
| battery_feasibility | feasible SOC0 window as % of 4735.75 MWh nameplate (eta_rt calibrated 0.834), `None` if the reservoir swing exceeds capacity |

**Demand reference decision**: dual-report. `@in` = nd(t-1) read off the input
window (what the model is actually given; the anchor is float-exact against
this). `@act` = actual nd(t) (production honesty; against this the anchor's
irreducible residual is the 5-min nd ramp, std ~96 MW). An `nd_next` arm
(scenario/forecast demand at t, readable by the head only — no backbone
retraining) should drive `@act` mismatch to ~0 and is scheduled in Stage 1.

## Methods x constraints matrix

| constraint | soft (penalty, lambda-sweep) | hard @ inference | hard trained-through |
| --- | --- | --- | --- |
| box (>=0, caps) | relu(-pred)^2 + relu(pred-cap)^2 | clamp to [0, cap] | sigmoid-scaled output |
| ramp | relu(abs(dpred) - ramp)^2 vs prev actual | clip to [prev-r_dn, prev+r_up] | pred = prev + r_dn + (r_up-r_dn)*sigmoid(z) |
| demand balance | (sum(SIGN*pred) - nd)^2 (= lambda_eb loss, data exists) | anchor rescale (DONE: WAPE 0.1075, exact) | anchored head trained-through (DONE: failed, 0.34) |
| battery SOC | skip in training loss (path constraint, not window-local) | stateful swing-budget clip in closed loop | — |

Joint hard projection: clamp -> ramp-clip -> demand-rescale can undo each other;
iterate the cascade 2-3x (Dykstra-style) or solve the tiny per-step QP (box
intersect hyperplane in 6 dims). Battery SOC is enforced statefully in closed
loop only, and *reported* for teacher-forced runs.

## Stages

- [x] **Stage 0 — harness + baselines** (`constraints/harness.py`,
      `constraints/stage0_baselines.py`): constraint_report() refactored from
      ml/check_caps.py; score actuals and every existing checkpoint
      (bare / +clamp / +anchor). Output: results/stage0.md. (2026-07-06;
      data/gru_predictions_5min.csv referenced by CONSTRAINT.md is not in the
      repo, so the GRU row is absent.) Findings:
      - Ramp constraints are already ~free: 0-1 violations for every model.
      - Caps: 0 violations everywhere; negativity is fully fixed by clamp.
      - **lstm_revin's battery predictions are SOC-INFEASIBLE** (reservoir
        swing 112% of nameplate; anchor trims it to a knife-edge 100%);
        itransformer stays feasible in all variants ([43%,58%] window). New
        argument for the itransformer backbone.
      - dem@act has a floor: the anchor is exact @in but its @act mismatch
        (1.81%) is the nd ramp — and the BARE itransformer is slightly closer
        to nd(t) (1.74%) because the backbone implicitly forecasts the ramp.
        So the nd_next arm is the only route to ~0 @act mismatch.
      - **5-min persistence WAPE 0.0956 beats every learned model on every
        channel** (stage0_persistence.py), is exactly @in-balanced (its sum IS
        nd(t-1) by the identity), ramp- and SOC-feasible. Consequences:
        (a) one-step WAPE alone is near-degenerate at 5 min — the learned
        models are noisy persistence; value must be shown on responsiveness,
        closed-loop stability, horizon > 1. (b) **anchor(persistence)** —
        rescale the last dispatch mix to the prescribed demand — is a
        zero-parameter simulator with WAPE ~0.0956 + full structural response;
        it becomes the mandatory baseline every method must beat, and Stage 1
        must measure its closed-loop mix-staleness (where neural backbones
        should finally win).
- [ ] **Stage 1 — full hard-projection cascade at inference**: box + ramp +
      demand (+ nd_next arm) teacher-forced; stateful SOC clip closed-loop.
      Expected: all violations -> 0 at WAPE ~0.108-0.115. The "hardest" point
      on the graph and predicted Pareto anchor.
      - [x] Closed-loop persistence test (stage1_persistence_cl.py, 2026-07-06):
        **anchor(persistence) wins closed-loop too** — cl_WAPE 0.4749 vs prod
        LSTM 0.4971 vs anchored iTransformer 0.7015, tracking 70 MW, response
        +179.2% exact. VERDICT: on this benchmark (h=1 + 3h closed loop) the
        neural backbones add nothing over structural persistence+anchor; the
        h=1 MSE objective trains them to imitate persistence, and their
        closed-loop mix drifts. **persistence+anchor is the reference
        simulator**; the gate for any neural model is now 0.0956 one-step /
        0.4749 closed-loop while constraint-clean. A neural component must be
        trained for what persistence can't do (multi-step mix shares, learned
        response allocation) to earn a place.
- [ ] **Stage 2 — soft-penalty lambda grids**: lambda in {0,1,3,10,30,100} per
      constraint family, strided (12x) for curve shape (~5 min/point), full
      convergence spot-checks at 2-3 chosen lambdas ONLY (soft compliance
      decays with convergence — strided numbers must be validated before the
      final graph). Plus one adaptive-lambda (Lagrangian dual ascent) run per
      family. Reuse demand_simulation/sweep_lambda_eb.py scaffolding + the
      checkpoint/resume train loop from test_demand_anchored_full.py.
- [ ] **Stage 3 — hard trained-through variants** (optional, completeness):
      ramp-sigmoid head; demand head revisit with chg-detach. Strided pilots
      first; full run only if a pilot beats the Stage 1 frontier.
- [ ] **Stage 4 — graphs + writeup**: see below.
- [ ] **Stage 5 — literature review** (parallel): `constraints/lit_review.md`.
      Anchor list written (5 themes, papers + search keywords); verification /
      expansion by web search pending.

## The graph

Main figure: WAPE (y) vs total violations (x, log scale; zero-violation points
on a broken "0" axis), one series per mechanism family, lambda annotated along
soft curves. Secondary per-family figure: lambda on x, WAPE + violations on
twin y axes. "Hardness" is thus measured by achieved violation rate, which is
comparable across mechanisms, rather than by lambda, which is not.

## Literature themes (Stage 5, anchors to verify + expand by search)

1. Hard-constraint layers: DC3 (Donti, Rolnick & Kolter, ICLR 2021 — equality
   completion + inequality correction; our anchor IS a completion layer),
   OptNet (Amos & Kolter 2017), cvxpylayers (Agrawal et al. 2019), gauge-map
   methods (LOOP-LC line).
2. Soft/Lagrangian: Fioretto et al. — Lagrangian duality for constrained deep
   learning (power-systems applications); PINN-style penalties.
3. Power-system ML with feasibility: DeepOPF (predict-then-reconstruct),
   feasibility repair layers for OPF proxies.
4. Forecast reconciliation: MinT (Wickramasuriya/Hyndman) — projection onto
   linear constraints with proofs it reduces expected error; the theory behind
   "projection is free".
5. Safety layers / action projection: Dalal et al. 2018 (per-step analytic QP)
   — template for the closed-loop SOC clip.

## Data

Historical pull DONE (2026-07-06, `script/pull_data_hist.py`, resumable chunk
cache in data/pull_cache/): **2021-10-01 -> 2026-07-05, 500,832 5-min intervals**
(0 missing market, 45 missing flow intervals, 0 failed chunks) as
`vic_{generation,market,interconnector}_20211001_20260706.parquet`. Probe
findings: API serves 5-min back to 1999, but batteries/solar_utility only exist
from ~2021 — pre-2021 cannot support the 6-target model (cutoff validated).
Ignore the `battery` combined fueltech column (60% NaN, redundant with
charging/discharging which are ~99.5% complete).

Per-year maxima (fleet growth + stress preview; 2021 = Oct-Dec, 2026 = Jan-Jul):

| year | batt_dis max | demand max | price max | intervals price>$1k |
| --- | --- | --- | --- | --- |
| 2021 | 303 | 7,547 | 455 | 0 |
| 2022 | 323 | 8,406 | 15,100 | 105 |
| 2023 | 409 | 8,949 | 14,522 | 24 |
| 2024 | 465 | 9,887 | 17,500 | 104 |
| 2025 | 1,687 | 9,490 | 17,500 | 166 |
| 2026 | 1,297 | 10,784 | 3,026 | 9 |

Implications: battery fleet grew 5.6x over the span -> **time-varying caps are
mandatory** for era-appropriate violation counting (single CONSTRAINT.md caps
would miscount 2021-24). Natural stress candidates confirmed: June 2022 market
crisis, Jan 2024 + 2025 price-cap events, Jan 2026 record demand (10,784 MW).

**Pipeline integration (2026-07-06)**: `pipeline.prepare(..., dataset="hist")`.
- **VIC-island demand**: on hist, demand_mw := demand_mw - net_import_mw (the
  load VIC generators actually serve; project scoped to VIC only). Downstream
  scripts must NOT subtract net_import again on hist. last365 is frozen with
  legacy semantics for reproducibility of all existing results.
- Splits: train -> 2025-06-30, val -> 2025-12-31, test 2026-01..07 (record
  peak + winter finally inside the test set). 393,984 / 52,992 / 53,568 windows.
- Memory: features scaled to float32 BEFORE windowing so windows are zero-copy
  strided views -- peak RSS 1.25 GB for the full prepare (vs ~10 GB
  materialized). The API's combined `battery` fueltech is dropped in
  build_table (60% NaN; would have wiped rows at dropna).
- **Local training is feasible** (measured, MPS, batch 128, full 394k windows):
  iTransformer 1.9 min/epoch (~1 h to 30 epochs); LSTM 5.7 min/epoch (~3 h to
  30 epochs, use the checkpoint/resume loop); stride-12 pilots are seconds/epoch.
- CALIBRATION ITEM: island demand-side nd vs dispatch-sum nd shows a systematic
  ~-197 MW mean gap (mean|r| 223 MW, ~4% of level) = rooftop/losses/non-scheduled
  supply outside the 6 targets. Any anchor fed scenario demand-side nd inherits
  this bias; needs an offset calibration (or anchor to dispatch-sum-consistent
  signals) before hist scenario sims.

**Hist baselines (stage0_hist rows, 2026-07-06)**: persistence on the hist test
set (Jan-Jul 2026, 53,568 windows) = **WAPE 0.1084**, R2 0.9669 (per-target:
hydro .053, coal .008, steam .036, ocgt .034, batt_chg .249, batt_dis .270) —
the bar every hist-trained model must beat; harder than last365's 0.0956 almost
entirely via the batteries (bigger fleet + record summer in test). Two metric
findings from the actuals row, both harness TODOs:
1. RESOLVED: the 2026 actuals contained 60,391 negative cells (min -7 MW,
   station-load readings on idle units). Fix (user decision): clip hist targets
   >= 0 in build_table (net_demand computed after -> identity exact; last365
   frozen raw) + box_tol=0.1 MW noise floor in constraint_report (float32
   scaler round trips turn exact zeros into +-1e-4 MW). Result: actuals AND
   persistence n_neg = 0, WAPE unchanged.
2. RESOLVED (2026-07-12): the "SOC fails even for the ACTUALS" reading (swing
   115% of nameplate) was a **metric artifact**, not physics. `constraint_report`
   measured cumulative reservoir swing over one continuous 6-month segment
   (`seg` broke on data gaps only), so a ~1% eta bias drifts unbounded over 52k
   steps. Batteries cycle daily, so the physical feasibility unit is the per-day
   swing. Fix: `constraint_report(soc_period="day")` (default) breaks segments at
   calendar-day boundaries too. Result on hist actuals: **feasible every one of
   186 days**, worst-day swing 79.4% of nameplate (Jan 1 summer peak), SOC0
   window [21%, 42%]. eta needed NO recalibration — the preprocessed table is
   already the hist era (whole-table == hist-test-era == 0.8341); segmentation
   was the entire fix. `soc_period="window"` restores the legacy 114.7% number.
   Diagnostic: scratchpad/soc_diag.py.
3. **Era caps**: actuals exceed the last365-era CONSTRAINT.md caps on 46 cells
   in the 2026 test window (new records: Jan peak, Jun winter). Harness needs
   caps recomputed from the training era (with the time-varying-caps handling
   the 5.6x battery growth already requires); until then 46 is the actuals-row
   floor for n_cap on hist.

## Conventions

- Scripts: `constraints/stage<N>_<what>.py`; one JSON per run in
  `constraints/results/`, markdown tables rebuilt from the JSONs (re-runs never
  clobber other rows). Same test split and metrics everywhere.
- Protocol per method: implement -> strided pilot (~5 min sanity) -> full run
  only if the pilot earns it -> constraint_report -> point on the graph.
- 8 GB machine notes: free data["Xtr"/"Xva"] right after pipeline.prepare when
  only evaluating; closed-loop rollouts on CPU (sequential batch-2; MPS
  watermark OOMs under browser load); full training runs need the per-epoch
  checkpoint/resume loop.
- Responsiveness (+179.2% structural via the anchor) is tracked in
  demand_simulation/findings.md and is NOT re-measured per constraint run; any
  method that alters the demand path must re-verify it there.
