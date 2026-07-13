# Demand-shift simulation — `itransformer_rayen`

**Model:** iTransformer backbone + RAYEN hard-constraint output layer (`lib/models.py::RayenHead`).
**Scenario:** `FixedPercentageShift`, rebound = reduction = **2%**, free window## Ramp-rate violations (the ask: mean magnitude, not just the count)

| metric | value |
| --- | --- |
| violations (target-step cells) | **43** |
| **mean excess per violation** | **27.7 MWh**  (= 332.0 MW over the ramp cap) |
| worst single violation | 1478.2 MW over cap (≈ 123 MWh) |
| total excess energy over rollout | 1189.5 MWh |
| excess as % of dispatched energy | 0.0063% |
| negative-dispatch cells | 0 |

43 violations out of 53,568 steps × 6 targets ≈ 321k consecutive-pair cells = **0.013%** of cells.

## Change in energy dispatch (+2% midday shift, free-window response region)

| channel | base → scenario response |
| --- | --- |
| demand input (exogenous, driver) | **+18.8%** |
| net_demand (model's own forecast) | −0.008% |
| coal_brown | +0.05% |
| hydro | +0.76% |
| gas_ocgt | −1.27% |
| battery_discharging | −4.35% | 11:00–14:00.
**Protocol:** closed-loop free-window rollout over the full hist test set
(2026-01-01 → 2026-07-05, **53,568** 5-min steps). Exogenous `net_demand`/`demand_mw`/price
teacher-forced; the model feeds its own dispatch back inside the free window.
**Ramp metric:** per-target asymmetric data ramps (`CONSTRAINT.md`) + 0.6 MW tolerance, on the
*predicted* trajectory `Δ = pred[t] − pred[t−1]`. Excess MWh = excess MW × 5/60 h.



## Accuracy / feasibility

- base **WAPE 0.583**, R² 0.453 (no-shift rollout, response region)
- balance vs own D_t forecast: max **0.0011 MW** — exact by construction
- net-demand tracking vs scenario input: p50 331.7 MW, p95 730.1 MW; mismatch 9.41%
- SOC feasible: **no** (swing 2079% of pack — the battery ramp/SOC arm is loose here)

## Reading the numbers

1. **The 2% shift barely moves dispatch.** Exogenous midday demand rises +18.8%, but the model's
   own net-demand forecast moves −0.008% and the fleet stirs by ≤ ~1–4%. This is the known
   non-responsiveness of the RAYEN hist head (`plan.md` note 4: responsiveness to exogenous demand
   shifts is *not* structural here) — it anchors to the previous dispatch and forecasts its own D_t,
   which does not track the shifted input (hence the 331 MW median tracking gap).

   **Caveat — `net_demand` definition swap (train vs rollout).** The rollout feeds `net_demand =
   demand_mw − wind − solar_utility` (demand-side, `demand_side_nd_hist`), teacher-forced, so the
   demand shift propagates into the input. But the model was **trained** on `net_demand = Σ
   dispatchable generation` (`= SIGN·P`, supply-side; `pipeline.py:125`). On the test window the two
   differ by ~197 MW mean / 199 MAE / up to 629 MW (demand-side ~5% lower), so the input is ~5%
   off-distribution vs training. This is deliberate (a supply-side `net_demand` wouldn't move with a
   demand shift), but it inflates the 331 MW tracking gap and partly confounds the responsiveness
   read. The balance guarantee is unaffected (enforced against the model's own D_t).

2. **The ramp violations are the visible symptom of closed-loop collapse in the free window.**
   Localising the 43 violations by rollout position:

   | position | violations | of N steps |
   | --- | --- | --- |
   | closed-loop interior (anchor = prev prediction) | 0 | 6,510 |
   | first free step (re-reads actual) | 0 | 186 |
   | **free → teacher-forced seam (`TF-out`)** | **42** | 186 |
   | teacher-forced interior | 1 | 46,686 |

   Trace at the worst one (coal_brown, 2026-01-07 14:00): through the 11:00–14:00 **closed-loop**
   free window the prediction is **frozen flat at ~2108 MW** (Δ ≈ −0.2 MW/step) while actual coal
   climbs 3648 → 3880 MW — i.e. the iTransformer **collapses to a near-constant** under its own
   feedback (cf. the known closed-loop collapse). At 14:00 teacher-forcing resumes, the window
   re-reads actual dispatch, the anchor jumps 2107 → 3880, and the prediction snaps to 3854: a
   **+1747 MW one-step jump = the violation**. So the "violation" is the frozen closed-loop
   prediction reconnecting to reality.

   The 0-violations-in-closed-loop line is *not* reassurance: RAYEN starts at persistence
   (`s_bias=−4` → steps ~1.8% of the ramp) and takes tiny steps, so under feedback it freezes —
   ramp feasibility holds *because* it barely moves. The balance/floor guarantees do hold
   (bal 0.0011 MW, 0 negatives), but the closed-loop *trajectory* is broken.

3. **One failure, not two.** The free window is the only closed-loop segment, and it is where the
   demand shift is applied and where response is scored. A prediction frozen at ~2108 MW there
   ignores the +18.8% demand **and** the rising actual — so (a) dispatch barely responds
   (base-vs-scenario Δ 0.92 MW/cell) and (b) the frozen value snaps back at 14:00 (the ramp jump).
   Both are the **same closed-loop collapse**; they only look like separate problems because one is
   measured base-vs-scenario and the other over time. (The `net_demand` train/rollout definition
   swap in note 1 compounds the non-response but is not the primary cause — the collapse is.)

_Full metric row: `demand_simulation/sweep_eqnd/itr_rayen_reb2_red2.md` (regenerate with_
_`python3 demand_simulation/hist_constrained_shift.py --rebound 2 --reduction 2 --models itransformer_rayen`)._
